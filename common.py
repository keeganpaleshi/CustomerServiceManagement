import base64
import json
import os
import pickle
from email.mime.text import MIMEText

import yaml
import requests
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI

# Load YAML configuration
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)

SCOPES = CFG["gmail"]["scopes"]
GMAIL_CLIENT_SECRET = CFG["gmail"]["client_secret_file"]
GMAIL_TOKEN_FILE = CFG["gmail"]["token_file"]

OPENAI_API_KEY = os.getenv(CFG["openai"]["api_key_env"])
CLASSIFY_MODEL = CFG["openai"]["classify_model"]
CLASSIFY_MAX_TOKENS = CFG["openai"].get("classify_max_tokens", 50)

CRITIC_THRESHOLD = CFG["thresholds"]["critic_threshold"]
MAX_RETRIES = CFG["thresholds"]["max_retries"]

TICKET_SYSTEM = CFG["ticket"]["system"]
FREESCOUT_URL = CFG["ticket"]["freescout_url"]
FREESCOUT_KEY = CFG["ticket"]["freescout_key"]

PROMO_LABELS = {
    "SPAM",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}


def get_gmail_service(creds_filename: str | None = None, token_filename: str | None = None):
    """Authenticate with Gmail API and return a service resource."""
    creds_filename = creds_filename or GMAIL_CLIENT_SECRET
    token_filename = token_filename or GMAIL_TOKEN_FILE
    creds = None
    if os.path.exists(token_filename):
        with open(token_filename, "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_filename, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_filename, "wb") as token:
            pickle.dump(creds, token)
    return build("gmail", "v1", credentials=creds)


def fetch_all_unread_messages(service):
    """Return all unread message references."""
    unread, token = [], None
    while True:
        resp = service.users().messages().list(userId="me", q="is:unread", pageToken=token).execute()
        unread.extend(resp.get("messages", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return unread


def create_base64_message(sender: str, to: str, subject: str, body: str):
    """Create a Gmail API message body."""
    msg = MIMEText(body)
    msg["to"], msg["from"], msg["subject"] = to, sender, subject
    raw_bytes = base64.urlsafe_b64encode(msg.as_bytes())
    return {"raw": raw_bytes.decode()}


def create_draft(service, user_id: str, msg_body, thread_id: str | None = None):
    """Create a draft email."""
    data = {"message": msg_body}
    if thread_id:
        data["message"]["threadId"] = thread_id
    return service.users().drafts().create(userId=user_id, body=data).execute()


def thread_has_draft(service, thread_id: str) -> bool:
    """Return True if the Gmail thread already contains a draft."""
    data = service.users().threads().get(userId="me", id=thread_id).execute()
    return any("DRAFT" in (m.get("labelIds") or []) for m in data.get("messages", []))


def is_promotional_or_spam(message, body_text: str) -> bool:
    """Return True if the message looks like a newsletter or spam."""
    labels = set(message.get("labelIds", []))
    if labels & PROMO_LABELS:
        return True
    headers = {h.get("name", "").lower(): h.get("value", "") for h in message.get("payload", {}).get("headers", [])}
    if "list-unsubscribe" in headers or "list-id" in headers:
        return True
    if "unsubscribe" in body_text.lower():
        return True
    return False


def critic_email(draft: str, original: str) -> dict:
    """Self-grade a draft reply using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return ONLY JSON {\"score\":1-10,\"feedback\":\"...\"} rating on correctness, tone, length.",
            },
            {"role": "assistant", "content": draft},
            {"role": "user", "content": f"Original email:\n\n{original}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def classify_email(text: str) -> dict:
    """Classify an email and return a dict with type and importance."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Categorize the email as lead, customer, or other. Return ONLY JSON {\"type\":\"lead|customer|other\",\"importance\":1-10}. NO other text.",
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=CLASSIFY_MAX_TOKENS,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        print(f"Error classifying email: {exc}")
        return {"type": "other", "importance": 0}


def create_ticket(subject: str, sender: str, body: str):
    """Create a ticket in FreeScout if ticketing is enabled."""
    if TICKET_SYSTEM != "freescout":
        return
    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }
    resp = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-FreeScout-API-Key": FREESCOUT_KEY,
        },
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
