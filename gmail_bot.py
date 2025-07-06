import argparse
import base64
import json
import time
from datetime import datetime, timedelta

import requests
from openai import OpenAI

from Draft_Replies import generate_ai_reply
from utils import (
    get_gmail_service,
    fetch_all_unread_messages,
    create_base64_message,
    create_draft,
    thread_has_draft,
    is_promotional_or_spam,
    critic_email,
    classify_email,
    GMAIL_QUERY,
    HTTP_TIMEOUT,
    MAX_DRAFTS,
    CRITIC_THRESHOLD,
    MAX_RETRIES,
    OPENAI_API_KEY,
    CLASSIFY_MODEL,
    TICKET_SYSTEM,
    FREESCOUT_URL,
    FREESCOUT_KEY,
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

# Ticketing constants are loaded from utils

if not OPENAI_API_KEY:
    raise ValueError(
        "Please set your OPENAI_API_KEY environment variable.")


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
    ensure_label(service, "FreeScout Updates")
    since = datetime.utcnow() - timedelta(minutes=5)
    while True:
        convs = fetch_recent_conversations(since.isoformat(), timeout=timeout)
        if convs:
            lines = [f"#{c.get('id')} {c.get('subject', '')[:50]} [{c.get('status')}]" for c in convs]
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

        # ---- classification ----
        cls = classify_email(f"Subject:{subject}\n\n{body}")
        email_type = cls["type"]

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
