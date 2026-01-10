import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parseaddr
from typing import Dict, Optional, Tuple

import requests

from Draft_Replies import generate_ai_reply
from utils import (
    classify_email,
    create_base64_message,
    apply_label_to_thread,
    ensure_label,
    fetch_all_unread_messages,
    get_gmail_service,
    get_settings,
    FreeScoutClient,
    is_promotional_or_spam,
    thread_has_draft,
    extract_plain_text,
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


# Label used to mark messages that already have a ticket to avoid duplicates
TICKET_LABEL_NAME = "Ticketed"
_TICKET_LABEL_ID: Optional[str] = None


@dataclass(frozen=True)
class ProcessResult:
    status: str
    reason: str
    freescout_conversation_id: Optional[str] = None


ProcessResult.SKIPPED_ALREADY_SUCCESS = ProcessResult(
    status="skipped_already_success",
    reason="already processed",
)
ProcessResult.FILTERED = ProcessResult(
    status="filtered",
    reason="already filtered",
)


def should_filter_message(message: dict) -> Tuple[bool, str]:
    body_text = message.get("body_text", "")
    if is_promotional_or_spam(message, body_text):
        return True, "filtered: promotional/spam"
    return False, ""


def process_gmail_message(
    message: dict,
    store: TicketStore,
    freescout: Optional[FreeScoutClient],
    gmail,
) -> ProcessResult:
    message_id = message.get("id", "")
    thread_id = message.get("threadId")

    if not message_id:
        reason = "missing message id"
        print("Skipping message without id")
        return ProcessResult(status="failed_retryable", reason=reason)

    if store.processed_success(message_id) is True:
        print(f"{message_id[:8]}… skipped (already processed)")
        return ProcessResult(
            status="skipped_already_success",
            reason="already processed",
        )
    try:
        full_message = message
        if "payload" not in message:
            full_message = (
                gmail.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

        thread_id = full_message.get("threadId") or thread_id
        if not thread_id:
            reason = "missing thread id"
            store.mark_failed(message_id, thread_id, reason)
            print(f"{message_id[:8]}… error: missing thread id")
            return ProcessResult(status="failed_permanent", reason=reason)

        def get_header_value(
            payload: Optional[dict], name: str, default: str = ""
        ) -> str:
            headers = (payload or {}).get("headers") or []
            for header in headers:
                if header.get("name") == name:
                    return header.get("value", default)
            return default

        payload = full_message.get("payload", {})
        subject = get_header_value(payload, "Subject")
        raw_sender = get_header_value(payload, "From")
        sender = parseaddr(raw_sender)[1] or raw_sender
        body = extract_plain_text(payload)
        snippet = full_message.get("snippet", "")
        body_text = body.strip() or snippet or "(no body)"

        message_with_body = dict(full_message)
        message_with_body["body_text"] = body_text
        filtered, reason = should_filter_message(message_with_body)
        if filtered:
            store.mark_filtered(message_id, thread_id, reason=reason)
            print(f"{message_id[:8]}… {reason}")
            return ProcessResult(status="filtered", reason=reason)

        conv_id = store.get_conversation_id_for_thread(thread_id)
        if conv_id:
            if not freescout:
                reason = "freescout disabled"
                store.mark_failed(message_id, thread_id, reason, conv_id)
                print(f"{message_id[:8]}… failed: freescout disabled")
                return ProcessResult(
                    status="failed_retryable",
                    reason=reason,
                    freescout_conversation_id=conv_id,
                )
            try:
                freescout.add_customer_thread(conv_id, body_text, imported=True)
            except requests.RequestException as exc:
                reason = f"append failed: {exc}"
                store.mark_failed(message_id, thread_id, str(exc), conv_id)
                print(f"{message_id[:8]}… error appending to {conv_id}: {exc}")
                return ProcessResult(
                    status="failed_retryable",
                    reason=reason,
                    freescout_conversation_id=conv_id,
                )

            store.mark_success(message_id, thread_id, conv_id, action="append")
            if _TICKET_LABEL_ID:
                apply_label_to_thread(gmail, thread_id, _TICKET_LABEL_ID)
            return ProcessResult(
                status="freescout_appended",
                reason="append success",
                freescout_conversation_id=conv_id,
            )

        if not freescout:
            reason = "freescout disabled"
            store.mark_failed(message_id, thread_id, reason)
            print(f"{message_id[:8]}… failed: freescout disabled")
            return ProcessResult(status="failed_retryable", reason=reason)

        settings = get_settings()
        mailbox_id = settings.get("FREESCOUT_MAILBOX_ID")
        if not mailbox_id:
            reason = "freescout mailbox missing"
            store.mark_failed(message_id, thread_id, reason)
            print(f"{message_id[:8]}… error: freescout mailbox missing")
            return ProcessResult(status="failed_permanent", reason=reason)

        gmail_thread_field = settings.get("FREESCOUT_GMAIL_THREAD_FIELD_ID")
        gmail_message_field = settings.get("FREESCOUT_GMAIL_MESSAGE_FIELD_ID")

        try:
            ticket = freescout.create_conversation(
                subject,
                sender,
                body_text,
                mailbox_id,
                thread_id=thread_id,
                message_id=message_id,
                gmail_thread_field=gmail_thread_field,
                gmail_message_field=gmail_message_field,
            )
        except requests.RequestException as exc:
            reason = f"ticket creation failed: {exc}"
            store.mark_failed(message_id, thread_id, str(exc))
            print(f"{message_id[:8]}… error: ticket creation failed: {exc}")
            return ProcessResult(status="failed_retryable", reason=reason)

        conv_id = _extract_conversation_id(ticket)
        if not ticket or not conv_id:
            reason = "ticket creation failed"
            store.mark_failed(message_id, thread_id, reason, conv_id)
            print(f"{message_id[:8]}… error: ticket creation failed")
            return ProcessResult(
                status="failed_retryable",
                reason=reason,
                freescout_conversation_id=conv_id,
            )

        store.upsert_thread_map(thread_id, conv_id)
        store.mark_success(message_id, thread_id, conv_id, action="create")
        if _TICKET_LABEL_ID:
            apply_label_to_thread(gmail, thread_id, _TICKET_LABEL_ID)
        return ProcessResult(
            status="freescout_created",
            reason="create success",
            freescout_conversation_id=conv_id,
        )
    except Exception as exc:
        reason = f"unexpected error: {exc}"
        store.mark_failed(message_id, thread_id, str(exc))
        print(f"{message_id[:8]}… error: {exc}")
        return ProcessResult(status="failed_retryable", reason=reason)


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


def process_gmail_message_freescout(
    ref: dict,
    ticket_store: TicketStore,
    client: Optional[FreeScoutClient],
    svc,
    ticket_label_id: Optional[str],
    timeout: int,
) -> Optional[ProcessResult]:
    """Deprecated: use process_gmail_message for single-message ingestion."""
    global _TICKET_LABEL_ID
    if ticket_label_id is not None:
        _TICKET_LABEL_ID = ticket_label_id
    _ = timeout
    return process_gmail_message(ref, ticket_store, client, svc)


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
    global _TICKET_LABEL_ID
    _TICKET_LABEL_ID = ticket_label_id
    client = _build_freescout_client(timeout=args.timeout)

    skipped_already_processed = 0
    created_conversations = 0
    appended_threads = 0
    filtered_terminal = 0
    failed_retryable = 0
    failed_permanent = 0

    for ref in fetch_all_unread_messages(svc, query=args.gmail_query)[
        : settings["MAX_DRAFTS"]
    ]:
        result = process_gmail_message(ref, ticket_store, client, svc)
        if result.status == "skipped_already_success":
            skipped_already_processed += 1
        elif result.status == "filtered":
            filtered_terminal += 1
        elif result.status == "freescout_appended":
            appended_threads += 1
        elif result.status == "freescout_created":
            created_conversations += 1
        elif result.status == "failed_retryable":
            failed_retryable += 1
        elif result.status == "failed_permanent":
            failed_permanent += 1

    updates = poll_ticket_updates()
    if updates:
        print(f"Fetched {len(updates)} ticket updates from FreeScout")

    print("Ingestion summary:")
    print(f"  skipped_already_processed: {skipped_already_processed}")
    print(f"  created_conversations: {created_conversations}")
    print(f"  appended_threads: {appended_threads}")
    print(f"  filtered_terminal: {filtered_terminal}")
    print(f"  failed_retryable: {failed_retryable}")
    print(f"  failed_permanent: {failed_permanent}")


if __name__ == "__main__":
    main()
