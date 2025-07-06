# Utility functions shared between gmail scripts
import os
import pickle
import base64
import json
from email.mime.text import MIMEText

import yaml
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI

# Load configuration
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)

SCOPES = CFG["gmail"]["scopes"]
CLIENT_SECRET = CFG["gmail"]["client_secret_file"]
TOKEN_FILE = CFG["gmail"]["token_file"]

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


# ---------------------- Gmail helpers ----------------------

def get_gmail_service(creds_filename: str | None = None, token_filename: str | None = None):
    """Authenticate with Gmail using OAuth2 and return a service object."""
    creds_filename = creds_filename or CLIENT_SECRET
    token_filename = token_filename or TOKEN_FILE
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
    """Return a list of all unread message references."""
    unread = []
    page_token = None
    while True:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread", pageToken=page_token)
            .execute()
        )
        unread.extend(resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return unread


# ---------------------- Email utilities ----------------------

def create_base64_message(sender, to, subject, body_text):
    msg = MIMEText(body_text)
    msg["to"] = to
    msg["from"] = sender
    msg["subject"] = subject
    raw_bytes = base64.urlsafe_b64encode(msg.as_bytes())
    return {"raw": raw_bytes.decode("utf-8")}


def create_draft(service, user_id, message_body, thread_id=None):
    """Create and insert a draft email."""
    try:
        data = {"message": message_body}
        if thread_id:
            data["message"]["threadId"] = thread_id
        return service.users().drafts().create(userId=user_id, body=data).execute()
    except Exception as error:
        print(f"An error occurred creating the draft: {error}")
        return None


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


# ---------------------- OpenAI helpers ----------------------

def classify_email(text: str) -> dict:
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

