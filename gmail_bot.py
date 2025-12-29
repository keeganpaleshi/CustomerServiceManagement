import argparse
import base64
import json
import time
from datetime import datetime, timedelta
from email.utils import parseaddr

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

# Label used to mark messages that already have a ticket to avoid duplicates
TICKET_LABEL_NAME = "Ticketed"

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
    has_existing_draft: bool,
    ticket_label_id: str | None,
    timeout: int = HTTP_TIMEOUT,
) -> str:
    """Route an email based on priority and information level.

    Returns an action string:
    - "ignored" if the email type is not handled
    - "ticketed" if a ticket was created (and labeled)
    - "followup_draft" if a simple follow-up draft was created
    - "no_action" if the message should proceed to full draft creation
    """

    email_type = cls.get("type")
    importance = cls.get("importance", 0)
    if email_type == "other":
        return "ignored"

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
        ticket = create_ticket(subject, sender, body, timeout=timeout)
        if ticket_label_id and ticket is not None:
            service.users().threads().modify(
                userId="me", id=thread_id, body={"addLabelIds": [ticket_label_id]}
            ).execute()
        return "ticketed"

    # Otherwise ask for more details if we haven't drafted already
    if not has_existing_draft:
        followup = (
            "Thank you for contacting us. Could you provide more details about "
            "your request so we can assist you?"
        )
        msg = create_base64_message("me", sender, f"Re: {subject}", followup)
        create_draft(service, "me", msg, thread_id=thread_id)
        return "followup_draft"

    return "no_action"


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

    params = None
    if since_iso:
        # Support multiple possible FreeScout filter names to avoid missing updates
        params = {"updated_since": since_iso, "updated_from": since_iso}

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

    data = resp.json() or []
    if isinstance(data, dict):
        return data.get("data") or data.get("conversations") or []
    if isinstance(data, list):
        return data
    return []


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


def send_update_email(service, summary: str, label_id: str | None = None):
    msg = create_base64_message("me", "me", "FreeScout Updates", summary)
    if label_id:
        msg["labelIds"] = [label_id]
    service.users().messages().send(userId="me", body=msg).execute()


def poll_freescout_updates(service, interval: int = 300, timeout: int = HTTP_TIMEOUT):
    """Continuously poll FreeScout and email a summary of new conversations."""
    label_id = ensure_label(service, "FreeScout Updates")
    since = datetime.utcnow() - timedelta(minutes=5)
    while True:
        convs = fetch_recent_conversations(since.isoformat(), timeout=timeout)
        if convs:
            lines = [f"#{c.get('id')} {c.get('subject', '')[:50]} [{c.get('status')}]" for c in convs]
            summary = "\n".join(lines)
            send_update_email(service, summary, label_id=label_id)
        since = datetime.utcnow()
        time.sleep(interval)


def main():
    args = parse_args()

    svc = get_gmail_service()
    ticket_label_id = None
    if TICKET_SYSTEM == "freescout":
        ticket_label_id = ensure_label(svc, TICKET_LABEL_NAME)
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
        raw_sender = next(
            (h["value"] for h in msg["payload"]["headers"] if h["name"] == "From"),
            "",
        )
        sender = parseaddr(raw_sender)[1] or raw_sender
        thread = msg["threadId"]

        if ticket_label_id and ticket_label_id in set(msg.get("labelIds", [])):
            print(f"{ref['id'][:8]}… skipped (ticket already created)")
            continue

        def decode_base64url(data: str) -> str:
            """Decode base64url strings that may be missing padding."""

            if not data:
                return ""

            padding = "=" * (-len(data) % 4)
            try:
                return base64.urlsafe_b64decode((data + padding).encode("utf-8")).decode(
                    "utf-8", "ignore"
                )
            except (base64.binascii.Error, ValueError) as exc:
                print(f"Failed to decode message body: {exc}")
                return ""

        def extract_plain_text(payload: dict | None) -> str:
            """Recursively search a payload tree for the first text/plain body."""

            if not payload:
                return ""

            mime_type = payload.get("mimeType", "")
            body_data = payload.get("body", {}).get("data")

            # Use the body if this part is plain text
            if mime_type == "text/plain" and body_data:
                return decode_base64url(body_data)

            # Multipart containers or parts with children
            for part in payload.get("parts", []) or []:
                text = extract_plain_text(part)
                if text:
                    return text

            # Fallback: single-part messages store data directly on payload
            if body_data:
                return decode_base64url(body_data)

            return ""

        payload = msg.get("payload", {})
        body = extract_plain_text(payload)
        snippet = msg.get("snippet", "")

        if is_promotional_or_spam(msg, body):
            print(f"{ref['id'][:8]}… skipped promotional/spam")
            continue

        # ---- classification ----
        cls = classify_email(f"Subject:{subject}\n\n{body}")
        email_type = cls["type"]

        has_draft = thread_has_draft(svc, thread)

        action = route_email(
            svc,
            subject,
            sender,
            body,
            thread,
            cls,
            has_existing_draft=has_draft,
            ticket_label_id=ticket_label_id,
            timeout=args.timeout,
        )

        if action in {"ignored", "ticketed", "followup_draft"}:
            continue

        if has_draft:
            continue

        # ---- draft creation with critic ----
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
