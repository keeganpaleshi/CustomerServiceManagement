import os

# additional imports
import json
import requests
import argparse
import time

from googleapiclient.discovery import build     # already imported in Draft_Replies, keep here too
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle, base64

from openai import OpenAI
from Draft_Replies import generate_ai_reply
import yaml


# Load YAML config
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)

# Gmail settings
SCOPES               = CFG["gmail"]["scopes"]
GMAIL_CLIENT_SECRET  = CFG["gmail"]["client_secret_file"]
GMAIL_TOKEN_FILE     = CFG["gmail"]["token_file"]
GMAIL_QUERY          = CFG["gmail"].get("query", "is:unread")

HTTP_TIMEOUT         = CFG.get("http", {}).get("timeout", 15)

# OpenAI models & API key
OPENAI_API_KEY       = os.getenv(CFG["openai"]["api_key_env"])
CLASSIFY_MODEL       = CFG["openai"]["classify_model"]
CLASSIFY_MAX_TOKENS  = CFG["openai"].get("classify_max_tokens", 50)

# Critic settings
CRITIC_THRESHOLD     = CFG["thresholds"]["critic_threshold"]
MAX_RETRIES          = CFG["thresholds"]["max_retries"]

MAX_DRAFTS           = CFG.get("limits", {}).get("max_drafts", 100)

def parse_args():
    parser = argparse.ArgumentParser(description="Process Gmail messages")
    parser.add_argument("--gmail-query", default=GMAIL_QUERY, help="Gmail search query")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP request timeout")
    return parser.parse_args()

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

def fetch_all_unread_messages(service, query: str = GMAIL_QUERY):
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



def create_ticket(subject: str, sender: str, body: str, timeout: int = HTTP_TIMEOUT, retries: int = 3):
    """Create a ticket in FreeScout with basic retry logic."""
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
        except requests.RequestException as e:
            if attempt == retries:
                print(f"Failed to create ticket after {retries} attempts: {e}")
                return None
            sleep_time = 2 ** (attempt - 1)
            print(f"Ticket creation error: {e}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def route_email(service, thread_id: str, subject: str, sender: str, body: str, snippet: str, timeout: int = HTTP_TIMEOUT):
    """Classify an email and either open a ticket or create a draft."""
    cls = classify_email(f"Subject:{subject}\n\n{body}")
    email_type = cls.get("type", "other")
    importance = cls.get("importance", 0)

    if email_type == "other":
        return "skipped", importance

    if importance >= 8:
        created = create_ticket(subject, sender, body, timeout=timeout)
        if created:
            return "ticket", importance

    if not thread_has_draft(service, thread_id):
        request_text = (
            "Thanks for reaching out. Could you provide more details about your request so we can assist?"
        )
        draft_msg = create_base64_message("me", sender, f"Re: {subject}", request_text)
        create_draft(service, "me", draft_msg, thread_id=thread_id)
        return "draft", importance

    return "none", importance


def poll_ticket_updates(limit: int = 10, timeout: int = HTTP_TIMEOUT):
    """Fetch recent ticket updates from FreeScout."""
    if TICKET_SYSTEM != "freescout":
        return []

    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    try:
        resp = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "X-FreeScout-API-Key": FREESCOUT_KEY,
            },
            params={"limit": limit},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.RequestException as e:
        print(f"Error polling FreeScout: {e}")
        return []


def main():
    args = parse_args()

    svc = get_gmail_service()
    for ref in fetch_all_unread_messages(svc, query=args.gmail_query)[:MAX_DRAFTS]:
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

        part = msg["payload"]["parts"][0]["body"].get("data", "")
        body = base64.urlsafe_b64decode(part).decode("utf-8", "ignore")
        snippet = msg.get("snippet", "")

        if is_promotional_or_spam(msg, body):
            print(f"{ref['id'][:8]}… skipped promotional/spam")
            continue

        action, importance = route_email(
            svc, thread, subject, sender, body, snippet, timeout=args.timeout
        )
        print(f"{ref['id'][:8]}… {action:<6} imp={importance}")

    updates = poll_ticket_updates()
    if updates:
        print(f"Fetched {len(updates)} ticket updates from FreeScout")


if __name__ == "__main__":
    main()

