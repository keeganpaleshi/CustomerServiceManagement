import argparse
import json
import time
from datetime import datetime, timedelta
from email.utils import parseaddr
from typing import Dict, Optional

import requests
from openai import OpenAI

from Draft_Replies import generate_ai_reply
from utils import (
    classify_email,
    create_base64_message,
    create_draft,
    create_ticket,
    critic_email,
    ensure_label,
    fetch_all_unread_messages,
    get_gmail_service,
    get_settings,
    FreeScoutClient,
    is_promotional_or_spam,
    thread_has_draft,
    extract_plain_text,
    require_openai_api_key,
    require_ticket_settings,
)
from storage import TicketStore


def parse_args(settings: Optional[Dict] = None):
    settings = settings or get_settings()
    parser = argparse.ArgumentParser(description="Process Gmail messages")
    parser.add_argument(
        "--gmail-query", default=settings["GMAIL_QUERY"], help="Gmail search query"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=settings["HTTP_TIMEOUT"],
        help="HTTP request timeout",
    )
    parser.add_argument(
        "--poll-freescout",
        action="store_true",
        help="Continuously poll FreeScout for updates",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=settings.get("FREESCOUT_POLL_INTERVAL", 300),
        help="Seconds between FreeScout polls",
    )
    parser.add_argument(
        "--console-auth",
        action="store_true",
        default=settings["GMAIL_USE_CONSOLE"],
        help="Use console-based OAuth (paste auth code) instead of opening a browser",
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

def route_email(
    service,
    subject: str,
    sender: str,
    body: str,
    thread_id: str,
    message_id: str,
    cls: dict,
    has_existing_draft: bool,
    ticket_label_id: Optional[str],
    ticket_store: Optional[TicketStore] = None,
    timeout: Optional[int] = None,
) -> tuple[str, Optional[str]]:
    """Route an email based on priority and information level.

    Returns (action, conversation_id)
    - "ignored" if the email type is not handled
    - "ticketed" if a ticket exists/was created (and labeled)
    - "ticket_failed" if ticket creation failed
    - "followup_draft" if a simple follow-up draft was created
    - "no_action" if the message should proceed to full draft creation
    """

    settings = get_settings()
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]

    email_type = cls.get("type")
    importance = cls.get("importance", 0)
    if email_type == "other":
        return "ignored"

    high_priority = importance >= 8
    needs_info = False
    try:
        client = OpenAI(api_key=require_openai_api_key())
        resp = client.chat.completions.create(
            model=settings["CLASSIFY_MODEL"],
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

    existing_conv = None
    if ticket_store and thread_id:
        existing_conv = ticket_store.get_conv_id(thread_id)

    if high_priority or needs_info:
        if existing_conv:
            return "ticketed", existing_conv

        ticket = create_ticket(
            subject,
            sender,
            body,
            thread_id=thread_id,
            message_id=message_id,
            timeout=http_timeout,
        )
        conv_id = _extract_conversation_id(ticket)
        if ticket_store and conv_id and thread_id:
            ticket_store.upsert_thread_map(thread_id, conv_id)
        if ticket_label_id and ticket is not None:
            service.users().threads().modify(
                userId="me", id=thread_id, body={"addLabelIds": [ticket_label_id]}
            ).execute()
        return ("ticketed" if ticket is not None else "ticket_failed"), conv_id

    # Otherwise ask for more details if we haven't drafted already
    if not has_existing_draft:
        followup = (
            "Thank you for contacting us. Could you provide more details about "
            "your request so we can assist you?"
        )
        msg = create_base64_message("me", sender, f"Re: {subject}", followup)
        create_draft(service, "me", msg, thread_id=thread_id)
        return "followup_draft", None

    return "no_action", None


def poll_ticket_updates(limit: int = 10, timeout: Optional[int] = None):
    """Fetch recent ticket updates from FreeScout."""
    settings = get_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return []
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    url = f"{settings['FREESCOUT_URL'].rstrip('/')}/api/conversations"
    try:
        resp = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "X-FreeScout-API-Key": settings["FREESCOUT_KEY"],
            },
            params={"limit": limit},
            timeout=http_timeout,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.RequestException as e:
        print(f"Error polling FreeScout: {e}")
        return []


def _build_freescout_client(timeout: Optional[int] = None) -> Optional[FreeScoutClient]:
    settings = get_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return None
    url, key = require_ticket_settings()
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    return FreeScoutClient(url, key, timeout=http_timeout)


def _extract_latest_thread_text(conversation: dict) -> str:
    threads = conversation.get("threads") or conversation.get("data") or []
    if isinstance(threads, dict):
        threads = threads.get("threads", [])

    customer_threads = [
        t
        for t in threads
        if isinstance(t, dict) and t.get("type") in {"customer", "message", "reply"}
    ]
    if not customer_threads:
        return conversation.get("last_text", "") or conversation.get("text", "")

    latest = customer_threads[-1]
    return latest.get("text", "") or latest.get("body", "")


def _build_tags(cls: dict, high_priority: bool) -> list[str]:
    tags = [cls.get("type", "other")]
    if high_priority:
        tags.append("high-priority")
    return tags


def _prepare_custom_fields(cls: dict, settings: dict) -> dict:
    fields_cfg = settings.get("FREESCOUT_ACTIONS", {}).get("custom_fields", {})
    custom_fields: dict = {}
    type_field = fields_cfg.get("type_field_id")
    importance_field = fields_cfg.get("importance_field_id")
    if type_field:
        custom_fields[str(type_field)] = cls.get("type")
    if importance_field:
        custom_fields[str(importance_field)] = cls.get("importance")
    return custom_fields


def _extract_conversation_id(ticket: Optional[dict]) -> Optional[str]:
    if not ticket or not isinstance(ticket, dict):
        return None

    candidates = [
        ticket.get("id"),
        ticket.get("conversation_id"),
        (ticket.get("conversation") or {}).get("id"),
        (ticket.get("data") or {}).get("id"),
        ((ticket.get("data") or {}).get("conversation") or {}).get("id"),
    ]
    for conv_id in candidates:
        if conv_id:
            return str(conv_id)
    return None


def fetch_recent_conversations(
    since_iso: Optional[str] = None, timeout: Optional[int] = None
):
    """Return list of recent FreeScout conversations since a given ISO time."""
    settings = get_settings()
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    url = f"{settings['FREESCOUT_URL'].rstrip('/')}/api/conversations"

    params = None
    if since_iso:
        # Support multiple possible FreeScout filter names to avoid missing updates
        params = {"updated_since": since_iso, "updated_from": since_iso}

    try:
        resp = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "X-FreeScout-API-Key": settings["FREESCOUT_KEY"],
            },
            params=params,
            timeout=http_timeout,
        )
        resp.raise_for_status()

        data = resp.json() or []
        if isinstance(data, dict):
            return data.get("data") or data.get("conversations") or []
        if isinstance(data, list):
            return data
        return []
    except requests.RequestException as e:
        print(f"Error polling FreeScout: {e}")
        return []


def send_update_email(service, summary: str, label_id: Optional[str] = None):
    msg = create_base64_message("me", "me", "FreeScout Updates", summary)
    if label_id:
        msg["labelIds"] = [label_id]
    service.users().messages().send(userId="me", body=msg).execute()


def process_freescout_conversation(
    client: FreeScoutClient,
    conversation: dict,
    actions_cfg: dict,
    settings: dict,
):
    conv_id = conversation.get("id")
    if not conv_id:
        return

    try:
        details = client.get_conversation(conv_id)
    except requests.RequestException as exc:
        print(f"Failed to fetch conversation {conv_id}: {exc}")
        return

    subject = details.get("subject") or conversation.get("subject") or "(no subject)"
    latest_text = _extract_latest_thread_text(details) or conversation.get("last_text", "")

    cls = classify_email(f"Subject:{subject}\n\n{latest_text}")
    importance = cls.get("importance", 0)
    high_priority = importance >= actions_cfg.get("priority_high_threshold", 8)

    tags = None
    if actions_cfg.get("apply_tags", True):
        tags = _build_tags(cls, high_priority)

    custom_fields = _prepare_custom_fields(cls, settings)
    if not custom_fields:
        custom_fields = None

    priority_value = None
    if actions_cfg.get("update_priority", True):
        priority_value = "urgent" if high_priority else "normal"

    assignee = actions_cfg.get("assign_to_user_id")

    try:
        client.update_conversation(
            conv_id,
            priority=priority_value,
            assignee=assignee,
            tags=tags,
            custom_fields=custom_fields,
        )
    except requests.RequestException as exc:
        print(f"Failed to update conversation {conv_id}: {exc}")

    note_lines = [
        f"AI classification: {cls.get('type', 'unknown')}",
        f"Importance: {importance}",
    ]
    if high_priority:
        note_lines.append("Marked as high priority")

    if actions_cfg.get("post_internal_notes", True):
        try:
            client.add_internal_note(conv_id, "\n".join(note_lines))
        except requests.RequestException as exc:
            print(f"Failed to add internal note to {conv_id}: {exc}")

    if actions_cfg.get("post_suggested_reply", True):
        try:
            suggestion = generate_ai_reply(
                subject,
                "customer",
                latest_text,
                cls.get("type", "other"),
            )
            client.add_suggested_reply(conv_id, suggestion)
        except requests.RequestException as exc:
            print(f"Failed to add suggested reply to {conv_id}: {exc}")


def poll_freescout_updates(
    interval: int = 300, timeout: Optional[int] = None
):
    """Continuously poll FreeScout and classify new/updated conversations."""

    settings = get_settings()
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    client = _build_freescout_client(timeout=http_timeout)
    if not client:
        print("FreeScout polling skipped because ticket system is not set to freescout.")
        return

    actions_cfg = settings.get("FREESCOUT_ACTIONS", {})
    since = datetime.utcnow() - timedelta(minutes=5)
    while True:
        convs = fetch_recent_conversations(since.isoformat(), timeout=http_timeout)
        for conv in convs:
            process_freescout_conversation(client, conv, actions_cfg, settings)
        since = datetime.utcnow()
        time.sleep(interval)


def freescout_webhook_handler(payload: dict, headers: dict) -> tuple[str, int]:
    """Generic webhook handler usable by Flask or FastAPI routes."""

    settings = get_settings()
    secret = settings.get("FREESCOUT_WEBHOOK_SECRET", "")
    if secret and headers.get("X-Webhook-Secret") != secret:
        return "invalid signature", 401

    if not payload:
        return "missing payload", 400

    conv_id = payload.get("conversation_id") or payload.get("id")
    if not conv_id:
        return "missing conversation id", 400

    client = _build_freescout_client(timeout=settings["HTTP_TIMEOUT"])
    if not client:
        return "freescout disabled", 503

    actions_cfg = settings.get("FREESCOUT_ACTIONS", {})
    process_freescout_conversation(client, {"id": conv_id}, actions_cfg, settings)
    return "ok", 200


def main():
    settings = get_settings()
    args = parse_args(settings)

    if args.poll_freescout:
        poll_freescout_updates(
            interval=args.poll_interval,
            timeout=args.timeout,
        )
        return

    sqlite_path = settings.get("TICKET_SQLITE_PATH") or "./csm.sqlite"
    ticket_store = TicketStore(sqlite_path)
    svc = get_gmail_service(use_console=args.console_auth)
    ticket_label_id = None
    if settings["TICKET_SYSTEM"] == "freescout":
        ticket_label_id = ensure_label(svc, TICKET_LABEL_NAME)
    client = _build_freescout_client(timeout=args.timeout)

    skipped_already_processed = 0
    created_conversations = 0
    appended_threads = 0
    filtered_terminal = 0
    failed_freescout = 0

    for ref in fetch_all_unread_messages(svc, query=args.gmail_query)[
        : settings["MAX_DRAFTS"]
    ]:
        message_id = ref.get("id", "")
        thread = ref.get("threadId")

        if not message_id:
            print("Skipping message without id")
            continue

        if ticket_store.processed_terminal(message_id):
            skipped_already_processed += 1
            print(f"{message_id[:8]}… skipped (already processed)")
            continue

        try:
            msg = (
                svc.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

            thread = msg.get("threadId") or thread
            if not thread:
                ticket_store.mark_failed(message_id, thread, "missing thread id")
                failed_freescout += 1
                print(f"{message_id[:8]}… error: missing thread id")
                continue

            def get_header_value(payload: Optional[dict], name: str, default: str = "") -> str:
                headers = (payload or {}).get("headers") or []
                for header in headers:
                    if header.get("name") == name:
                        return header.get("value", default)
                return default

            payload = msg.get("payload", {})
            subject = get_header_value(payload, "Subject")
            raw_sender = get_header_value(payload, "From")
            sender = parseaddr(raw_sender)[1] or raw_sender
            body = extract_plain_text(payload)
            snippet = msg.get("snippet", "")
            body_text = body.strip() or snippet or "(no body)"

            if is_promotional_or_spam(msg, body_text):
                ticket_store.mark_filtered(
                    message_id,
                    thread,
                    reason="filtered: promotional/spam",
                )
                filtered_terminal += 1
                print(f"{ref['id'][:8]}… filtered promotional/spam")
                continue

            conv_id = ticket_store.get_conv_id(thread) if thread else None

            if conv_id:
                if not client:
                    ticket_store.mark_failed(message_id, thread, "freescout disabled", conv_id)
                    print(f"{message_id[:8]}… failed: freescout disabled")
                    continue
                try:
                    client.add_customer_thread(conv_id, body_text, imported=True)
                    ticket_store.mark_success(message_id, thread, conv_id)
                    appended_threads += 1
                    if ticket_label_id:
                        svc.users()
                        .threads()
                        .modify(userId="me", id=thread, body={"addLabelIds": [ticket_label_id]})
                        .execute()
                except requests.RequestException as exc:
                    ticket_store.mark_failed(message_id, thread, str(exc), conv_id)
                    failed_freescout += 1
                    print(f"{ref['id'][:8]}… error appending to {conv_id}: {exc}")
                continue

            ticket = create_ticket(
                subject,
                sender,
                body_text,
                thread_id=thread,
                message_id=message_id,
                timeout=args.timeout,
            )
            conv_id = _extract_conversation_id(ticket)
            if not ticket or not conv_id:
                ticket_store.mark_failed(message_id, thread, "ticket creation failed", conv_id)
                failed_freescout += 1
                print(f"{ref['id'][:8]}… error: ticket creation failed")
                continue

            ticket_store.upsert_thread_map(thread, conv_id)
            ticket_store.mark_success(message_id, thread, conv_id)
            created_conversations += 1
            if ticket_label_id:
                svc.users()
                .threads()
                .modify(userId="me", id=thread, body={"addLabelIds": [ticket_label_id]})
                .execute()
        except Exception as exc:
            ticket_store.mark_failed(message_id, thread, str(exc))
            failed_freescout += 1
            print(f"{ref['id'][:8]}… error: {exc}")
            continue

    updates = poll_ticket_updates()
    if updates:
        print(f"Fetched {len(updates)} ticket updates from FreeScout")

    print("Ingestion summary:")
    print(f"  skipped_already_processed: {skipped_already_processed}")
    print(f"  created_conversations: {created_conversations}")
    print(f"  appended_threads: {appended_threads}")
    print(f"  filtered_terminal: {filtered_terminal}")
    print(f"  failed_freescout: {failed_freescout}")


if __name__ == "__main__":
    main()
