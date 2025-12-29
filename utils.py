import base64
import json
import os
import pickle
import time
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI


CONFIG_PATH = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def _load_settings() -> Dict[str, Any]:
    """Load configuration and environment overrides lazily."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    return {
        "SCOPES": cfg["gmail"]["scopes"],
        "GMAIL_CLIENT_SECRET": cfg["gmail"]["client_secret_file"],
        "GMAIL_QUERY": cfg["gmail"].get("query", "is:unread"),
        "MAX_DRAFTS": cfg.get("limits", {}).get("max_drafts", 100),
        "CRITIC_THRESHOLD": cfg["thresholds"]["critic_threshold"],
        "MAX_RETRIES": cfg["thresholds"]["max_retries"],
        "GMAIL_TOKEN_FILE": cfg["gmail"]["token_file"],
        "HTTP_TIMEOUT": cfg.get("http", {}).get("timeout", 15),
        "OPENAI_API_KEY": os.getenv(cfg["openai"]["api_key_env"]),
        "OPENAI_API_KEY_ENV": cfg["openai"]["api_key_env"],
        "DRAFT_MODEL": cfg["openai"]["draft_model"],
        "DRAFT_MAX_TOKENS": cfg["openai"].get("draft_max_tokens", 16384),
        "DRAFT_SYSTEM_MSG": cfg["openai"].get("draft_system_message", ""),
        "CLASSIFY_MODEL": cfg["openai"]["classify_model"],
        "CLASSIFY_MAX_TOKENS": cfg["openai"].get("classify_max_tokens", 50),
        "PROMO_LABELS": {
            "SPAM",
            "CATEGORY_PROMOTIONS",
            "CATEGORY_SOCIAL",
            "CATEGORY_UPDATES",
            "CATEGORY_FORUMS",
        },
        "TICKET_SYSTEM": cfg["ticket"]["system"],
        "FREESCOUT_URL": os.getenv("FREESCOUT_URL")
        or cfg["ticket"].get("freescout_url", ""),
        "FREESCOUT_KEY": os.getenv("FREESCOUT_KEY")
        or cfg["ticket"].get("freescout_key", ""),
    }


def get_settings() -> Dict[str, Any]:
    """Public accessor for cached settings."""

    return _load_settings().copy()


def require_openai_api_key() -> str:
    """Return the OpenAI API key or raise a clear error when missing."""

    api_key = _load_settings()["OPENAI_API_KEY"]
    if not api_key:
        env_name = _load_settings()["OPENAI_API_KEY_ENV"]
        raise RuntimeError(
            f"Please set your {env_name} environment variable before calling OpenAI helpers."
        )
    return api_key


def require_ticket_settings() -> tuple[str, str]:
    """Return ticket URL/key or raise when configuration is incomplete."""

    settings = _load_settings()
    url, key = settings["FREESCOUT_URL"], settings["FREESCOUT_KEY"]
    if not url or not key:
        raise RuntimeError(
            "Please set FREESCOUT_URL and FREESCOUT_KEY via environment variables or config.yaml."
        )
    return url, key


# ----- Gmail helpers -----


def get_gmail_service(
    creds_filename: Optional[str] = None, token_filename: Optional[str] = None
):
    """Return an authenticated Gmail service instance."""

    settings = _load_settings()
    creds_filename = creds_filename or settings["GMAIL_CLIENT_SECRET"]
    token_filename = token_filename or settings["GMAIL_TOKEN_FILE"]
    creds = None
    if os.path.exists(token_filename):
        with open(token_filename, "rb") as t:
            creds = pickle.load(t)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_filename, settings["SCOPES"]
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
    settings = _load_settings()
    labels = set(message.get("labelIds", []))
    if labels & settings["PROMO_LABELS"]:
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


def critic_email(draft, original):
    """Self-grade a draft reply using GPT-4.1."""
    client = OpenAI(api_key=require_openai_api_key())
    resp = client.chat.completions.create(
        model=_load_settings()["CLASSIFY_MODEL"],
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


def classify_email(text):
    """Classify an email and return a dict with type and importance."""
    settings = _load_settings()
    client = OpenAI(api_key=require_openai_api_key())
    try:
        resp = client.chat.completions.create(
            model=settings["CLASSIFY_MODEL"],
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
            max_tokens=settings["CLASSIFY_MAX_TOKENS"],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


def create_ticket(
    subject: str,
    sender: str,
    body: str,
    timeout: Optional[int] = None,
    retries: int = 3,
):
    settings = _load_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return None

    url, key = require_ticket_settings()
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }

    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{url.rstrip('/')}/api/conversations",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-FreeScout-API-Key": key,
                },
                json=payload,
                timeout=http_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"Failed to create ticket after {retries} attempts: {exc}")
                return None
            delay = 2 ** (attempt - 1)
            print(f"Ticket creation error: {exc}. Retrying in {delay}s…")
            time.sleep(delay)
