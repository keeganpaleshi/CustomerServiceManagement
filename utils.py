import os
import json
import base64
import pickle
import time
from email.mime.text import MIMEText
from typing import Optional

import requests
import yaml
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from openai import OpenAI


CFG = None
SCOPES = None
GMAIL_CLIENT_SECRET = None
GMAIL_QUERY = None
MAX_DRAFTS = None
CRITIC_THRESHOLD = None
MAX_RETRIES = None
GMAIL_TOKEN_FILE = None
HTTP_TIMEOUT = None
OPENAI_API_KEY = None
CLASSIFY_MODEL = None
CLASSIFY_MAX_TOKENS = None
TICKET_SYSTEM = None
FREESCOUT_URL = None
FREESCOUT_KEY = None

PROMO_LABELS = {
    "SPAM",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}


def ensure_settings_loaded():
    """Load configuration and cached settings lazily.

    This avoids performing work (or requiring environment variables) during
    module import, which makes tooling happier in partially configured
    environments.
    """

    global CFG, SCOPES, GMAIL_CLIENT_SECRET, GMAIL_QUERY, MAX_DRAFTS
    global CRITIC_THRESHOLD, MAX_RETRIES, GMAIL_TOKEN_FILE, HTTP_TIMEOUT
    global OPENAI_API_KEY, CLASSIFY_MODEL, CLASSIFY_MAX_TOKENS
    global TICKET_SYSTEM, FREESCOUT_URL, FREESCOUT_KEY

    if CFG is not None:
        return

    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        CFG = yaml.safe_load(f)

    SCOPES = CFG["gmail"]["scopes"]
    GMAIL_CLIENT_SECRET = CFG["gmail"]["client_secret_file"]
    GMAIL_QUERY = CFG["gmail"].get("query", "is:unread")
    MAX_DRAFTS = CFG.get("limits", {}).get("max_drafts", 100)
    CRITIC_THRESHOLD = CFG["thresholds"]["critic_threshold"]
    MAX_RETRIES = CFG["thresholds"]["max_retries"]
    GMAIL_TOKEN_FILE = CFG["gmail"]["token_file"]
    HTTP_TIMEOUT = CFG.get("http", {}).get("timeout", 15)

    OPENAI_API_KEY = os.getenv(CFG["openai"]["api_key_env"])
    CLASSIFY_MODEL = CFG["openai"]["classify_model"]
    CLASSIFY_MAX_TOKENS = CFG["openai"].get("classify_max_tokens", 50)

    TICKET_SYSTEM = CFG["ticket"]["system"]
    FREESCOUT_URL = CFG["ticket"]["freescout_url"]
    FREESCOUT_KEY = CFG["ticket"]["freescout_key"]


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client, validating that an API key is configured."""

    ensure_settings_loaded()
    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError("Please set your OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=key)


# ----- Gmail helpers -----

def get_gmail_service(creds_filename=None, token_filename=None):
    """Return an authenticated Gmail service instance."""

    ensure_settings_loaded()
    creds_filename = creds_filename or GMAIL_CLIENT_SECRET
    token_filename = token_filename or GMAIL_TOKEN_FILE
    creds = None
    if os.path.exists(token_filename):
        with open(token_filename, "rb") as t:
            creds = pickle.load(t)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_filename, SCOPES)
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
    labels = set(message.get("labelIds", []))
    if labels & PROMO_LABELS:
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
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
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
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
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
            max_tokens=CLASSIFY_MAX_TOKENS,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


def create_ticket(subject, sender, body, timeout=None, retries=3):
    """Create a ticket in FreeScout with basic retry logic."""

    ensure_settings_loaded()
    timeout = timeout if timeout is not None else HTTP_TIMEOUT
    if TICKET_SYSTEM != "freescout":
        return None

    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-FreeScout-API-Key": FREESCOUT_KEY,
                },
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            sleep_time = 2 ** (attempt - 1)
            print(f"Ticket creation error: {exc}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
