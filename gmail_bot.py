import os

# additional imports
import json
import requests

from googleapiclient.discovery import build     # already imported in Draft_Replies, keep here too
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle, base64

from openai import OpenAI
from Draft_Replies import generate_ai_reply
import yaml
import time

# Load YAML config
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)

# Gmail settings
SCOPES               = CFG["gmail"]["scopes"]
GMAIL_CLIENT_SECRET  = CFG["gmail"]["client_secret_file"]
GMAIL_TOKEN_FILE     = CFG["gmail"]["token_file"]

# OpenAI models & API key
OPENAI_API_KEY       = os.getenv(CFG["openai"]["api_key_env"])
CLASSIFY_MODEL       = CFG["openai"]["classify_model"]
CLASSIFY_MAX_TOKENS  = CFG["openai"].get("classify_max_tokens", 50)

# Critic settings
CRITIC_THRESHOLD     = CFG["thresholds"]["critic_threshold"]
MAX_RETRIES          = CFG["thresholds"]["max_retries"]

MAX_DRAFTS           = CFG.get("limits", {}).get("max_drafts", 100)

# Gmail label IDs that indicate promotional or spammy content. Messages with
# any of these labels will be skipped entirely.
PROMO_LABELS = {
    "SPAM",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}

# Ticketing
TICKET_SYSTEM        = CFG["ticket"]["system"]
FREESCOUT_URL        = CFG["ticket"]["freescout_url"]
FREESCOUT_KEY        = CFG["ticket"]["freescout_key"]

if not OPENAI_API_KEY:
    raise ValueError(f"Please set your {CFG['openai']['api_key_env']} environment variable.")

def get_gmail_service(creds_filename=None, token_filename=None):
    """Authenticate with Gmail using OAuth2.

    Filenames can be provided as arguments or will default to the values
    specified in ``config.yaml``.
    """
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
    """Self-grade a draft reply using GPT-4.1."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY JSON {\"score\":1-10,\"feedback\":\"...\"} "
                    "rating on correctness, tone, length."
                ),
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
                    "content": (
                        "Categorize the email as lead, customer, or other. Return ONLY JSON {\"type\":\"lead|customer|other\",\"importance\":1-10}. NO other text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=CLASSIFY_MAX_TOKENS,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


def poll_ticket_updates():
    """Poll FreeScout for recent ticket updates and log basic info."""
    if TICKET_SYSTEM != "freescout":
        return

    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations?modified=1h"
    try:
        resp = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "X-FreeScout-API-Key": FREESCOUT_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        for conv in resp.json().get("data", []):
            num = conv.get("number") or conv.get("id")
            status = conv.get("status")
            print(f"Ticket {num}: status={status}")
    except Exception as e:
        print(f"Error polling ticket updates: {e}")



def create_ticket(subject: str, sender: str, body: str, retries: int = 3) -> str | None:
    """Create a ticket in FreeScout and return its ID."""
    if TICKET_SYSTEM != "freescout":
        return None

    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }

    for attempt in range(retries):
        try:
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
            data = resp.json()
            return str(data.get("id") or data.get("conversation", {}).get("id"))
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to create ticket: {e}")
                return None
            wait = 2 ** attempt
            print(f"Error creating ticket (attempt {attempt + 1}): {e}; retrying in {wait}s")
            time.sleep(wait)

    return None


def route_email(service, msg):
    """Classify an email then create a ticket or info-request draft."""
    subject = next((h["value"] for h in msg["payload"]["headers"] if h["name"] == "Subject"), "")
    sender = next((h["value"] for h in msg["payload"]["headers"] if h["name"] == "From"), "")
    thread = msg["threadId"]

    part = msg["payload"]["parts"][0]["body"]["data"]
    body = base64.urlsafe_b64decode(part).decode("utf-8", "ignore")
    snippet = msg.get("snippet", "")

    if is_promotional_or_spam(msg, body):
        print(f"{msg['id'][:8]}… skipped promotional/spam")
        return

    cls = classify_email(f"Subject:{subject}\n\n{body}")
    email_type = cls["type"]
    importance = cls["importance"]

    if email_type in ("lead", "customer"):
        ticket_id = create_ticket(subject, sender, body)
        if ticket_id:
            print(f"{msg['id'][:8]}… ticket {ticket_id} created ({email_type}, imp={importance})")
            return
        else:
            print(f"{msg['id'][:8]}… ticket creation failed; drafting reply")

    # For other emails or failed ticket creation, draft a reply asking for more info
    if not thread_has_draft(service, thread):
        draft_text = generate_ai_reply(
            subject,
            sender,
            "Ask the sender for more information so we can assist further.",
            email_type,
        )
        msg_draft = create_base64_message("me", sender, f"Re: {subject}", draft_text)
        create_draft(service, "me", msg_draft, thread_id=thread)
    print(f"{msg['id'][:8]}… draft info-request imp={importance}")


def main():

    svc = get_gmail_service()
    poll_ticket_updates()
    for ref in fetch_all_unread_messages(svc)[:MAX_DRAFTS]:
        msg = (
            svc.users()
            .messages()
            .get(userId="me", id=ref["id"], format="full")
            .execute()
        )
        route_email(svc, msg)


if __name__ == "__main__":
    main()

