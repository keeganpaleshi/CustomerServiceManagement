import os
import json
import base64
import pickle
from email.mime.text import MIMEText

import requests
import yaml
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from openai import OpenAI


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

PROMO_LABELS = {
    "SPAM",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}

TICKET_SYSTEM = CFG["ticket"]["system"]
FREESCOUT_URL = os.getenv("FREESCOUT_URL") or CFG["ticket"].get("freescout_url", "")
FREESCOUT_KEY = os.getenv("FREESCOUT_KEY") or CFG["ticket"].get("freescout_key", "")

if TICKET_SYSTEM == "freescout" and (not FREESCOUT_URL or not FREESCOUT_KEY):
    raise ValueError(
        "Please set FREESCOUT_URL and FREESCOUT_KEY via environment variables or config.yaml."
    )


# ----- Gmail helpers -----

def get_gmail_service(creds_filename=None, token_filename=None):
    """Return an authenticated Gmail service instance."""
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
    client = OpenAI(api_key=OPENAI_API_KEY)
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
    client = OpenAI(api_key=OPENAI_API_KEY)
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


def create_ticket(subject, sender, body, timeout=HTTP_TIMEOUT):
    if TICKET_SYSTEM != "freescout":
        return
    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }
    r = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-FreeScout-API-Key": FREESCOUT_KEY,
        },
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
