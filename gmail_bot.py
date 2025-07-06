import os
import json
import argparse

import base64

import time
from datetime import datetime, timedelta


import requests

from Draft_Replies import generate_ai_reply
from email_utils import (
    CFG,
    CLASSIFY_MAX_TOKENS,
    CLASSIFY_MODEL,
    CRITIC_THRESHOLD,
    GMAIL_QUERY,
    HTTP_TIMEOUT,
    MAX_RETRIES,
    MAX_DRAFTS,
    OPENAI_API_KEY,
    create_base64_message,
    create_draft,
    create_ticket,
    critic_email,
    classify_email,
    fetch_all_unread_messages,
    get_gmail_service,
    is_promotional_or_spam,
    thread_has_draft,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Process Gmail messages")
    parser.add_argument("--gmail-query", default=GMAIL_QUERY, help="Gmail search query")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP request timeout")
    parser.add_argument(
        "--poll-freescout",
        action="store_true",
        help="Continuously poll FreeScout for updates",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Seconds between FreeScout polls",
    )
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


def route_email(
    service,
    subject: str,
    sender: str,
    body: str,
    thread_id: str,
    cls: dict,
    timeout: int = HTTP_TIMEOUT,
) -> None:
    """Route an email based on priority and information level.

    If the message is high priority or lacks sufficient information, open a
    ticket. Otherwise, create a draft requesting additional details.
    """

    email_type = cls.get("type")
    importance = cls.get("importance", 0)
    if email_type == "other":
        return

    high_priority = importance >= 8
    needs_info = False
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Assess priority and information sufficiency. Return ONLY "
                        "JSON {\"priority\":\"high|normal\",\"needs_more_info\":true|false}."
                    ),
                },
                {"role": "user", "content": f"Subject:{subject}\n\n{body}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        result = json.loads(resp.choices[0].message.content)
        high_priority = result.get("priority") == "high" or high_priority
        needs_info = result.get("needs_more_info", False)
    except Exception as e:
        print(f"Priority check failed: {e}")

    if high_priority or needs_info:
        create_ticket(subject, sender, body, timeout=timeout)
        return

    # Otherwise ask for more details
    if not thread_has_draft(service, thread_id):
        followup = (
            "Thank you for contacting us. Could you provide more details about "
            "your request so we can assist you?"
        )
        msg = create_base64_message("me", sender, f"Re: {subject}", followup)
        create_draft(service, "me", msg, thread_id=thread_id)



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


def fetch_recent_conversations(since_iso: str | None = None, timeout: int = HTTP_TIMEOUT):
    """Return list of recent FreeScout conversations since a given ISO time."""
    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    params = {"updated_since": since_iso} if since_iso else None
    resp = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "X-FreeScout-API-Key": FREESCOUT_KEY,
        },
        params=params,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json() or []


def ensure_label(service, name: str) -> str:
    """Return Gmail label ID, creating the label if needed."""
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for lab in labels:
        if lab.get("name") == name:
            return lab["id"]
    body = {
        "name": name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
    }
    created = service.users().labels().create(userId="me", body=body).execute()
    return created["id"]


def send_update_email(service, summary: str):
    msg = create_base64_message("me", "me", "FreeScout Updates", summary)
    service.users().messages().send(userId="me", body=msg).execute()


def poll_freescout_updates(service, interval: int = 300, timeout: int = HTTP_TIMEOUT):
    """Continuously poll FreeScout and email a summary of new conversations."""
    label_id = ensure_label(service, "FreeScout Updates")
    since = datetime.utcnow() - timedelta(minutes=5)
    while True:
        convs = fetch_recent_conversations(since.isoformat(), timeout=timeout)
        if convs:
            lines = [f"#{c.get('id')} {c.get('subject','')[:50]} [{c.get('status')}]" for c in convs]
            summary = "\n".join(lines)
            send_update_email(service, summary)
        since = datetime.utcnow()
        time.sleep(interval)


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

        # ---- classification ----
        cls = classify_email(f"Subject:{subject}\n\n{body}")
        email_type, importance = cls["type"], cls["importance"]

        # skip others
        if email_type == "other":
            continue

        # ---- draft creation with critic ----
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
            msg_draft = create_base64_message(
                "me", sender, f"Re: {subject}", draft_text
            )
            create_draft(svc, "me", msg_draft, thread_id=thread)

        # ---- ticket or follow-up ----
        route_email(
            svc,
            subject,
            sender,
            body,
            thread,
            cls,
            timeout=args.timeout,
        )

    updates = poll_ticket_updates()
    if updates:
        print(f"Fetched {len(updates)} ticket updates from FreeScout")

    if args.poll_freescout:
        poll_freescout_updates(
            svc,
            interval=args.poll_interval,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()

