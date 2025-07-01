import os

# additional imports
import json
import requests
from Draft_Replies import generate_ai_reply

from googleapiclient.discovery import build     # already imported in Draft_Replies, keep here too
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle, base64

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def get_gmail_service(creds_filename=None, token_filename=None):
    """Authenticate with Gmail using OAuth2.

    Filenames can be provided as arguments or via the ``GMAIL_CLIENT_SECRET``
    and ``GMAIL_TOKEN_FILE`` environment variables. Defaults are
    ``client_secret.json`` and ``token.pickle``.
    """
    creds_filename = creds_filename or os.getenv("GMAIL_CLIENT_SECRET", "client_secret.json")
    token_filename = token_filename or os.getenv("GMAIL_TOKEN_FILE", "token.pickle")
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

def fetch_all_unread_messages(service):
    unread, token = [], None
    while True:
        resp = service.users().messages().list(userId="me", q="is:unread", pageToken=token).execute()
        unread.extend(resp.get("messages", []))
        token = resp.get("nextPageToken")
        if not token: break
    return unread

def create_base64_message(sender, to, subject, body):
    from email.mime.text import MIMEText
    msg = MIMEText(body)
    msg["to"], msg["from"], msg["subject"] = to, sender, subject
    return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}

def create_draft(service, user_id, msg_body, thread_id=None):
    data = {"message": msg_body}
    if thread_id: data["message"]["threadId"] = thread_id
    return service.users().drafts().create(userId=user_id, body=data).execute()

def thread_has_draft(service, thread_id):
    data = service.users().threads().get(userId="me", id=thread_id).execute()
    return any(
        "DRAFT" in (m.get("labelIds") or []) for m in data.get("messages", [])
    )
# — model choices —
CLASSIFY_MODEL  = "gpt-4.1"
DRAFT_MODEL     = "o3"

# — routing & thresholds —
TICKET_SYSTEM   = "freescout"
FREESCOUT_URL   = os.getenv("FREESCOUT_URL", "")
FREESCOUT_KEY   = os.getenv("FREESCOUT_KEY", "")
CRITIC_THRESHOLD = 8.0
MAX_RETRIES      = 2


def create_ticket(subject: str, sender: str, body: str):
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
        timeout=15,
    )
    r.raise_for_status()


def process_messages(service, unread_messages):
    """Placeholder for the upcoming main loop rewrite."""
    for msg in unread_messages:
        email_type = "other"
        # classification will set email_type in Module G
        if email_type == "other":
            continue
        # further processing will be added later

