import base64
import json
import os
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional, Sequence

import requests
import yaml
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from openai import OpenAI


@dataclass
class RuntimeSettings:
    scopes: Sequence[str]
    client_secret_file: str
    gmail_query: str
    max_drafts: int
    critic_threshold: int
    max_retries: int
    token_file: str
    http_timeout: int
    openai_api_key: Optional[str]
    openai_api_key_env: str
    classify_model: str
    classify_max_tokens: int
    ticket_system: str
    freescout_url: str
    freescout_key: str
    draft_model: str
    draft_max_tokens: int
    draft_system_message: str


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg_path = Path(__file__).with_name("config.yaml")
    with cfg_path.open() as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> RuntimeSettings:
    cfg = load_config()
    ticket_cfg = cfg.get("ticket", {})
    openai_cfg = cfg.get("openai", {})
    gmail_cfg = cfg.get("gmail", {})
    limits = cfg.get("limits", {})
    thresholds = cfg.get("thresholds", {})

    return RuntimeSettings(
        scopes=gmail_cfg.get("scopes", []),
        client_secret_file=gmail_cfg.get("client_secret_file", ""),
        gmail_query=gmail_cfg.get("query", "is:unread"),
        max_drafts=limits.get("max_drafts", 100),
        critic_threshold=thresholds.get("critic_threshold", 7),
        max_retries=thresholds.get("max_retries", 3),
        token_file=gmail_cfg.get("token_file", "token.json"),
        http_timeout=cfg.get("http", {}).get("timeout", 15),
        openai_api_key=os.getenv(openai_cfg.get("api_key_env", "OPENAI_API_KEY")),
        openai_api_key_env=openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
        classify_model=openai_cfg.get("classify_model", "gpt-4.1-mini"),
        classify_max_tokens=openai_cfg.get("classify_max_tokens", 50),
        ticket_system=ticket_cfg.get("system", ""),
        freescout_url=os.getenv("FREESCOUT_URL")
        or ticket_cfg.get("freescout_url", ""),
        freescout_key=os.getenv("FREESCOUT_KEY")
        or ticket_cfg.get("freescout_key", ""),
        draft_model=openai_cfg.get("draft_model", "gpt-4.1"),
        draft_max_tokens=openai_cfg.get("draft_max_tokens", 16384),
        draft_system_message=openai_cfg.get("draft_system_message", ""),
    )


# ----- Gmail helpers -----


def _require_openai_key(settings: Optional[RuntimeSettings] = None) -> str:
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            f"Please set your {settings.openai_api_key_env} environment variable."
        )
    return settings.openai_api_key


def get_gmail_service(
    creds_filename: Optional[str] = None, token_filename: Optional[str] = None
):
    """Return an authenticated Gmail service instance."""

    settings = get_settings()
    creds_filename = creds_filename or settings.client_secret_file
    token_filename = token_filename or settings.token_file
    creds = None
    if os.path.exists(token_filename):
        with open(token_filename, "rb") as t:
            creds = pickle.load(t)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_filename, settings.scopes
            )
            creds = flow.run_local_server(port=0)
        with open(token_filename, "wb") as t:
            pickle.dump(creds, t)
    return build("gmail", "v1", credentials=creds)


def fetch_all_unread_messages(service, query):
    unread, token = [], None
    while True:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=query, pageToken=token)
            .execute()
        )
        unread.extend(resp.get("messages", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return unread


def create_base64_message(sender, to, subject, body):
    msg = MIMEText(body)
    msg["to"], msg["from"], msg["subject"] = to, sender, subject
    return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}


def create_draft(service, user_id, msg_body, thread_id=None):
    data = {"message": msg_body}
    if thread_id:
        data["message"]["threadId"] = thread_id
    return service.users().drafts().create(userId=user_id, body=data).execute()


def thread_has_draft(service, thread_id):
    data = service.users().threads().get(userId="me", id=thread_id).execute()
    return any("DRAFT" in (m.get("labelIds") or []) for m in data.get("messages", []))


def is_promotional_or_spam(message, body_text):
    settings = get_settings()

    labels = set(message.get("labelIds", []))
    promo_labels = {
        "SPAM",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_SOCIAL",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
    }

    if labels & promo_labels:
        return True
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in message.get("payload", {}).get("headers", [])
    }
    if "list-unsubscribe" in headers or "list-id" in headers:
        return True
    if "unsubscribe" in body_text.lower():
        return True
    return False


def critic_email(draft, original, settings: Optional[RuntimeSettings] = None):
    """Self-grade a draft reply using GPT-4.1."""

    settings = settings or get_settings()
    client = OpenAI(api_key=_require_openai_key(settings))
    resp = client.chat.completions.create(
        model=settings.classify_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY JSON {\"score\":1-10,\"feedback\":\"...\"} rating on correctness, tone, length."
                ),
            },
            {"role": "assistant", "content": draft},
            {"role": "user", "content": f"Original email:\n\n{original}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def classify_email(text, settings: Optional[RuntimeSettings] = None):
    """Classify an email and return a dict with type and importance."""

    settings = settings or get_settings()
    client = OpenAI(api_key=_require_openai_key(settings))
    try:
        resp = client.chat.completions.create(
            model=settings.classify_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Categorize the email as lead, customer, or other. Return ONLY JSON {\"type\":\"lead|customer|other\",\"importance\":1-10}. NO other text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=settings.classify_max_tokens,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


def create_ticket(
    subject,
    sender,
    body,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    settings: Optional[RuntimeSettings] = None,
):
    settings = settings or get_settings()
    if settings.ticket_system != "freescout":
        return

    if not settings.freescout_url or not settings.freescout_key:
        raise ValueError(
            "Please set FREESCOUT_URL and FREESCOUT_KEY via environment variables or config.yaml."
        )

    effective_timeout = timeout if timeout is not None else settings.http_timeout
    attempts = retries if retries is not None else max(1, settings.max_retries)

    url = f"{settings.freescout_url.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-FreeScout-API-Key": settings.freescout_key,
                },
                json=payload,
                timeout=effective_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == attempts:
                raise
            backoff = 2 ** (attempt - 1)
            print(f"Ticket creation error: {exc}. Retrying in {backoff}s...")
            time.sleep(backoff)
