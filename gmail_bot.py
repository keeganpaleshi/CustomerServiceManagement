import os

# additional imports
import json
import requests
from Draft_Replies import generate_ai_reply, classify_email, critic_email

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


def main():
    svc = get_gmail_service()
    for ref in fetch_all_unread_messages(svc):
        msg = (
            svc.users()
            .messages()
            .get(userId="me", id=ref["id"], format="full")
            .execute()
        )
        subject = next(
            (h["value"] for h in msg["payload"]["headers"] if h["name"] == "Subject"),
            "",
        )
        sender = next(
            (h["value"] for h in msg["payload"]["headers"] if h["name"] == "From"),
            "",
        )
        thread = msg["threadId"]

        part = msg["payload"]["parts"][0]["body"]["data"]
        body = base64.urlsafe_b64decode(part).decode("utf-8", "ignore")
        snippet = msg.get("snippet", "")

        cls = classify_email(f"Subject:{subject}\n\n{body}")
        email_type, importance = cls["type"], cls["importance"]

        if email_type == "other":
            continue

        if not thread_has_draft(svc, thread):
            draft_text = generate_ai_reply(subject, sender, snippet, email_type)
            for _ in range(MAX_RETRIES):
                rating = critic_email(draft_text, body)
                if rating["score"] >= CRITIC_THRESHOLD:
                    break
                draft_text = generate_ai_reply(
                    subject,
                    sender,
                    f"{snippet}\n\nCritic feedback: {rating['feedback']}",
                    email_type,
                )
            msg_draft = create_base64_message("me", sender, f"Re: {subject}", draft_text)
            create_draft(svc, "me", msg_draft, thread_id=thread)

        create_ticket(subject, sender, body)

        print(f"{ref['id'][:8]}… {email_type:<8} imp={importance}")


if __name__ == "__main__":
    main()

