import argparse
import hashlib
import json
import logging
import os
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from typing import Dict, List, Optional, Tuple

import requests
from googleapiclient.errors import HttpError

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
    is_customer_thread,
    extract_plain_text,
    generate_ai_reply,
    log_event,
    normalize_id,
    require_ticket_settings,
    importance_to_bucket,
    reload_settings,
    retry_request,
    thread_timestamp,
    validate_conversation_id,
    get_freescout_rate_limiter,
)
from storage import TicketStore

# Thread-safe event for graceful shutdown in polling loops
_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    _shutdown_event.set()
    log_event(
        "shutdown",
        action="signal_received",
        signal=signal.Signals(signum).name,
    )


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested (thread-safe)."""
    return _shutdown_event.is_set()


def reset_shutdown_event() -> None:
    """Reset shutdown event (for testing purposes)."""
    _shutdown_event.clear()


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
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print processing status summary and recent failures without ingesting",
    )
    return parser.parse_args()


# Label used to mark messages that already have a ticket to avoid duplicates
TICKET_LABEL_NAME = "Ticketed"

# Default thresholds
DEFAULT_PRIORITY_HIGH_THRESHOLD = 8  # Default importance score that marks a message as high priority


@dataclass(frozen=True)
class ProcessResult:
    status: str
    reason: str
    freescout_conversation_id: Optional[str] = None
    drafted: bool = False


RESULT_SKIPPED_ALREADY_SUCCESS = ProcessResult(
    status="skipped_already_success",
    reason="already processed",
)
RESULT_FILTERED = ProcessResult(
    status="filtered",
    reason="already filtered",
)
RESULT_SKIPPED_ALREADY_CLAIMED = ProcessResult(
    status="skipped_already_claimed",
    reason="already claimed by another worker",
)


@dataclass(frozen=True)
class WebhookOutcome:
    action: str
    drafted: bool = False


def should_filter_message(message: dict) -> Tuple[bool, str]:
    body_text = message.get("body_text", "")
    if is_promotional_or_spam(message, body_text):
        return True, "filtered: promotional/spam"
    return False, ""


def _get_header_value(
    payload: Optional[dict], name: str, default: str = ""
) -> str:
    """Extract a header value from a Gmail message payload by name (case-insensitive)."""
    headers = (payload or {}).get("headers") or []
    normalized_name = name.lower()
    for header in headers:
        if header.get("name", "").lower() == normalized_name:
            return header.get("value", default)
    return default


@dataclass(frozen=True)
class _MessageContent:
    """Parsed content extracted from a Gmail message."""
    subject: str
    sender: str
    body_text: str


def _extract_message_content(full_message: dict) -> _MessageContent:
    """Extract subject, sender, and body text from a full Gmail message."""
    payload = full_message.get("payload", {})
    subject = _get_header_value(payload, "Subject")
    raw_sender = _get_header_value(payload, "From")
    sender = parseaddr(raw_sender)[1] or raw_sender
    body = extract_plain_text(payload)
    snippet = full_message.get("snippet", "")
    body_text = body.strip() or snippet or "(no body)"
    return _MessageContent(subject=subject, sender=sender, body_text=body_text)


def _finalize_ticket(
    freescout: FreeScoutClient,
    store: TicketStore,
    gmail,
    conv_id: str,
    content: _MessageContent,
    settings: dict,
    ticket_label_id: Optional[str],
    message_id: str,
    thread_id: str,
) -> bool:
    """Run post-ticket processing: classify, label, and draft a reply. Returns whether a draft was created."""
    email_type = "other"
    try:
        result = process_conversation(
            conv_id,
            {
                "subject": content.subject,
                "sender": content.sender,
                "latest_text": content.body_text,
            },
            settings,
            freescout,
        )
        if result and isinstance(result, dict):
            cls = result.get("classification") or {}
            email_type = cls.get("type", "other")
    except Exception as exc:
        log_event(
            "gmail_ingest",
            action="process_conversation",
            outcome="failed",
            reason=str(exc),
            message_id=message_id,
            thread_id=thread_id,
            conversation_id=conv_id,
        )
    if ticket_label_id:
        _apply_ticket_label(gmail, thread_id, ticket_label_id)
    return _maybe_write_draft_reply(
        freescout, store, conv_id,
        content.subject, content.sender, content.body_text, settings,
        email_type=email_type,
    )


def _append_to_conversation(
    freescout: FreeScoutClient,
    store: TicketStore,
    gmail,
    conv_id: str,
    content: _MessageContent,
    settings: dict,
    ticket_label_id: Optional[str],
    message_id: str,
    thread_id: str,
) -> ProcessResult:
    """Append a message to an existing FreeScout conversation."""
    try:
        freescout.add_customer_thread(conv_id, content.body_text, imported=True)
    except requests.RequestException as exc:
        reason = f"append failed: {exc}"
        store.mark_failed(message_id, thread_id, str(exc), conv_id)
        log_event(
            "gmail_ingest", action="append_thread", outcome="failed",
            reason=reason, message_id=message_id,
            thread_id=thread_id, conversation_id=conv_id,
        )
        return ProcessResult(
            status="failed_retryable", reason=reason,
            freescout_conversation_id=conv_id,
        )

    store.mark_success(message_id, thread_id, conv_id, action="append")
    log_event(
        "gmail_ingest", action="append_thread", outcome="success",
        message_id=message_id, thread_id=thread_id, conversation_id=conv_id,
    )
    drafted = _finalize_ticket(
        freescout, store, gmail, conv_id, content,
        settings, ticket_label_id, message_id, thread_id,
    )
    return ProcessResult(
        status="freescout_appended", reason="append success",
        freescout_conversation_id=conv_id, drafted=drafted,
    )


def _create_conversation_ticket(
    freescout: FreeScoutClient,
    store: TicketStore,
    gmail,
    content: _MessageContent,
    settings: dict,
    ticket_label_id: Optional[str],
    message_id: str,
    thread_id: str,
) -> ProcessResult:
    """Create a new FreeScout conversation for a Gmail message."""
    mailbox_id = normalize_id(settings.get("FREESCOUT_MAILBOX_ID"))
    if not mailbox_id:
        reason = "freescout mailbox missing"
        store.mark_failed(message_id, thread_id, reason)
        log_event(
            "gmail_ingest", action="create_ticket", outcome="failed",
            reason=reason, message_id=message_id, thread_id=thread_id,
        )
        return ProcessResult(status="failed_permanent", reason=reason)

    gmail_thread_field = normalize_id(settings.get("FREESCOUT_GMAIL_THREAD_FIELD_ID"))
    gmail_message_field = normalize_id(settings.get("FREESCOUT_GMAIL_MESSAGE_FIELD_ID"))

    try:
        ticket = freescout.create_conversation(
            content.subject, content.sender, content.body_text, mailbox_id,
            thread_id=thread_id, message_id=message_id,
            gmail_thread_field=gmail_thread_field,
            gmail_message_field=gmail_message_field,
        )
    except requests.RequestException as exc:
        reason = f"ticket creation failed: {exc}"
        store.mark_failed(message_id, thread_id, str(exc))
        log_event(
            "gmail_ingest", action="create_ticket", outcome="failed",
            reason=reason, message_id=message_id, thread_id=thread_id,
        )
        return ProcessResult(status="failed_retryable", reason=reason)

    if not ticket:
        reason = "ticket creation failed"
        store.mark_failed(message_id, thread_id, reason)
        log_event(
            "gmail_ingest", action="create_ticket", outcome="failed",
            reason=reason, message_id=message_id, thread_id=thread_id,
        )
        return ProcessResult(
            status="failed_retryable", reason=reason,
        )
    conv_id = _extract_conversation_id(ticket)
    if not conv_id:
        reason = "ticket creation failed"
        store.mark_failed(message_id, thread_id, reason, conv_id)
        log_event(
            "gmail_ingest", action="create_ticket", outcome="failed",
            reason=reason, message_id=message_id,
            thread_id=thread_id, conversation_id=conv_id,
        )
        return ProcessResult(
            status="failed_retryable", reason=reason,
            freescout_conversation_id=conv_id,
        )

    store.mark_success(message_id, thread_id, conv_id, action="create")
    store.upsert_thread_map(thread_id, conv_id)
    log_event(
        "gmail_ingest", action="create_ticket", outcome="success",
        message_id=message_id, thread_id=thread_id, conversation_id=conv_id,
    )
    drafted = _finalize_ticket(
        freescout, store, gmail, conv_id, content,
        settings, ticket_label_id, message_id, thread_id,
    )
    return ProcessResult(
        status="freescout_created", reason="create success",
        freescout_conversation_id=conv_id, drafted=drafted,
    )


def process_gmail_message(
    message: dict,
    store: TicketStore,
    freescout: Optional[FreeScoutClient],
    gmail,
    ticket_label_id: Optional[str] = None,
) -> ProcessResult:
    message_id = message.get("id", "")
    thread_id: Optional[str] = message.get("threadId")

    if not message_id:
        reason = "missing message id"
        log_event(
            "gmail_ingest", action="validate_message",
            outcome="failed", reason=reason,
        )
        return ProcessResult(status="failed_retryable", reason=reason)

    if not store.mark_processing_if_new(message_id, thread_id):
        if store.processed_success(message_id):
            log_event(
                "gmail_ingest", action="skip_message",
                outcome="already_processed",
                message_id=message_id, thread_id=thread_id,
            )
            return RESULT_SKIPPED_ALREADY_SUCCESS
        if store.processed_filtered(message_id):
            log_event(
                "gmail_ingest", action="skip_message",
                outcome="already_filtered",
                message_id=message_id, thread_id=thread_id,
            )
            return RESULT_FILTERED
        log_event(
            "gmail_ingest", action="skip_message",
            outcome="already_claimed",
            message_id=message_id, thread_id=thread_id,
        )
        return RESULT_SKIPPED_ALREADY_CLAIMED

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
            log_event(
                "gmail_ingest", action="validate_message",
                outcome="failed", reason=reason, message_id=message_id,
            )
            return ProcessResult(status="failed_permanent", reason=reason)

        content = _extract_message_content(full_message)

        message_with_body = dict(full_message)
        message_with_body["body_text"] = content.body_text
        filtered, reason = should_filter_message(message_with_body)
        if filtered:
            store.mark_filtered(message_id, thread_id, reason=reason)
            log_event(
                "gmail_ingest", action="filter_message", outcome="filtered",
                reason=reason, message_id=message_id, thread_id=thread_id,
            )
            return ProcessResult(status="filtered", reason=reason)

        settings = get_settings()
        conv_id = normalize_id(store.get_conversation_id_for_thread(thread_id))

        if conv_id:
            if not freescout:
                reason = "freescout disabled"
                store.mark_failed(message_id, thread_id, reason, conv_id)
                log_event(
                    "gmail_ingest", action="append_thread", outcome="failed",
                    reason=reason, message_id=message_id,
                    thread_id=thread_id, conversation_id=conv_id,
                )
                return ProcessResult(
                    status="failed_retryable", reason=reason,
                    freescout_conversation_id=conv_id,
                )
            return _append_to_conversation(
                freescout, store, gmail, conv_id, content,
                settings, ticket_label_id, message_id, thread_id,
            )

        if not freescout:
            reason = "freescout disabled"
            store.mark_failed(message_id, thread_id, reason)
            log_event(
                "gmail_ingest", action="create_ticket", outcome="failed",
                reason=reason, message_id=message_id, thread_id=thread_id,
            )
            return ProcessResult(status="failed_retryable", reason=reason)

        return _create_conversation_ticket(
            freescout, store, gmail, content,
            settings, ticket_label_id, message_id, thread_id,
        )
    except (
        requests.RequestException,  # Network/HTTP errors
        HttpError,                  # Gmail API errors
        json.JSONDecodeError,       # JSON parsing errors
        OSError,                    # File system / network socket errors
        TimeoutError,               # Timeout errors
    ) as exc:
        reason = f"transient error: {type(exc).__name__}: {exc}"
        store.mark_failed(message_id, thread_id, str(exc))
        log_event(
            "gmail_ingest", action="process_message", outcome="failed",
            reason=reason, error_type=type(exc).__name__,
            message_id=message_id, thread_id=thread_id,
        )
        return ProcessResult(status="failed_retryable", reason=reason)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        reason = f"data validation error: {type(exc).__name__}: {exc}"
        store.mark_failed(message_id, thread_id, str(exc))
        log_event(
            "gmail_ingest", action="process_message", outcome="failed",
            reason=reason, error_type=type(exc).__name__,
            message_id=message_id, thread_id=thread_id, level=logging.ERROR,
        )
        if os.getenv("DEBUG", "").lower() in ("1", "true", "yes"):
            raise
        return ProcessResult(status="failed_permanent", reason=reason)
    except (MemoryError, RecursionError):
        # Unrecoverable runtime errors — let them propagate
        raise
    except Exception as exc:
        reason = f"unexpected error: {type(exc).__name__}: {exc}"
        store.mark_failed(message_id, thread_id, str(exc))
        log_event(
            "gmail_ingest", action="process_message", outcome="failed",
            reason=reason, error_type=type(exc).__name__,
            message_id=message_id, thread_id=thread_id,
        )
        return ProcessResult(status="failed_retryable", reason=reason)


def poll_ticket_updates(
    limit: int = 10,
    timeout: Optional[int] = None,
    client: Optional[FreeScoutClient] = None,
):
    """Fetch recent ticket updates from FreeScout using FreeScoutClient."""
    settings = get_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return []

    owns_client = False
    if client is None:
        client = _build_freescout_client(timeout=timeout)
        owns_client = True
    if not client:
        return []

    try:
        return client.list_conversations({"limit": limit})
    except requests.RequestException as e:
        log_event(
            "freescout_poll",
            action="fetch_updates",
            outcome="failed",
            reason=str(e),
        )
        return []
    finally:
        if owns_client:
            client.close()


def _apply_ticket_label(gmail, thread_id: str, ticket_label_id: str) -> None:
    """Apply the ticket label to a Gmail thread with retry and error logging."""
    try:
        retry_request(
            lambda: apply_label_to_thread(gmail, thread_id, ticket_label_id),
            action_name="gmail.apply_label",
            exceptions=(HttpError,),
            log_context={"thread_id": thread_id, "label_id": ticket_label_id},
        )
    except HttpError as exc:
        log_event(
            "gmail_label",
            action="apply_label",
            outcome="failed",
            reason=str(exc),
            thread_id=thread_id,
            label_id=ticket_label_id,
        )


def _build_freescout_client(timeout: Optional[int] = None) -> Optional[FreeScoutClient]:
    settings = get_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return None
    url, key = require_ticket_settings()
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    rate_limiter = get_freescout_rate_limiter()
    return FreeScoutClient(url, key, timeout=http_timeout, rate_limiter=rate_limiter)


def _extract_latest_thread_text(conversation: dict) -> str:
    threads = conversation.get("threads") or conversation.get("data") or []
    if isinstance(threads, dict):
        threads = threads.get("threads", [])

    customer_threads = [
        t for t in threads if isinstance(t, dict) and is_customer_thread(t)
    ]
    if not customer_threads:
        return conversation.get("last_text", "") or conversation.get("text", "")

    latest: Optional[dict] = None
    latest_time: Optional[datetime] = None
    for thread in customer_threads:
        ts = thread_timestamp(thread)
        if ts and (latest_time is None or ts > latest_time):
            latest_time = ts
            latest = thread

    if latest is None:
        return conversation.get("last_text", "") or conversation.get("text", "")

    return latest.get("text", "") or latest.get("body", "")


def _build_tags(cls: dict, bucket: str, high_priority: bool) -> List[str]:
    tags = [cls.get("type", "other"), bucket]
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


def process_conversation(
    conversation_id: str,
    message_context: dict,
    settings: dict,
    freescout_client: FreeScoutClient,
) -> dict:
    actions_cfg = settings.get("FREESCOUT_ACTIONS", {})
    subject = message_context.get("subject") or "(no subject)"
    sender = message_context.get("sender") or "customer"
    latest_text = message_context.get("latest_text") or ""

    cls = classify_email(f"Subject:{subject}\n\n{latest_text}")
    importance = cls.get("importance", 0)
    high_priority = importance >= actions_cfg.get("priority_high_threshold", DEFAULT_PRIORITY_HIGH_THRESHOLD)
    bucket = importance_to_bucket(importance)

    tags = None
    if actions_cfg.get("apply_tags", True):
        tags = _build_tags(cls, bucket, high_priority)

    custom_fields = _prepare_custom_fields(cls, settings)
    if not custom_fields:
        custom_fields = None

    priority_value = None
    if actions_cfg.get("update_priority", True):
        priority_value = bucket

    assignee = actions_cfg.get("assign_to_user_id")

    # Track success/failure of each operation for reporting
    update_succeeded = True
    note_succeeded = True
    reply_succeeded = True

    try:
        freescout_client.update_conversation(
            conversation_id,
            priority=priority_value,
            assignee=assignee,
            tags=tags,
            custom_fields=custom_fields,
        )
    except requests.RequestException as exc:
        update_succeeded = False
        log_event(
            "freescout_update",
            action="update_conversation",
            outcome="failed",
            conversation_id=conversation_id,
            reason=str(exc),
        )

    note_lines = [
        f"AI classification: {cls.get('type', 'unknown')}",
        f"Importance: {importance}",
    ]
    if high_priority:
        note_lines.append("Marked as high priority")
    reasoning = cls.get("reasoning")
    if reasoning:
        note_lines.append(f"Reasoning: {reasoning}")
    facts = cls.get("facts") or []
    if isinstance(facts, list) and facts:
        note_lines.append("Extracted facts:")
        note_lines.extend([f"- {fact}" for fact in facts if fact])
    uncertainty = cls.get("uncertainty") or []
    if isinstance(uncertainty, list) and uncertainty:
        note_lines.append("Uncertainty:")
        note_lines.extend([f"- {item}" for item in uncertainty if item])

    if actions_cfg.get("post_internal_notes", True):
        try:
            freescout_client.add_internal_note(conversation_id, "\n".join(note_lines))
        except requests.RequestException as exc:
            note_succeeded = False
            log_event(
                "freescout_update",
                action="add_internal_note",
                outcome="failed",
                conversation_id=conversation_id,
                reason=str(exc),
            )

    if actions_cfg.get("post_suggested_reply", True):
        try:
            suggestion = generate_ai_reply(
                subject,
                sender,
                latest_text,
                cls.get("type", "other"),
            )
            freescout_client.add_suggested_reply(conversation_id, suggestion)
        except requests.RequestException as exc:
            reply_succeeded = False
            log_event(
                "freescout_update",
                action="add_suggested_reply",
                outcome="failed",
                conversation_id=conversation_id,
                reason=str(exc),
            )

    return {
        "classification": cls,
        "importance": importance,
        "high_priority": high_priority,
        "subject": subject,
        "latest_text": latest_text,
        "update_succeeded": update_succeeded,
        "note_succeeded": note_succeeded,
        "reply_succeeded": reply_succeeded,
    }


def _extract_id(response: Optional[dict], id_type: str = "conversation") -> Optional[str]:
    """
    Extract an ID from a FreeScout API response.

    Args:
        response: API response dict
        id_type: Type of ID to extract ('conversation', 'thread', etc.)

    Returns:
        Normalized ID string or None
    """
    if not response or not isinstance(response, dict):
        return None

    # Try direct ID keys
    candidates = [
        response.get("id"),
        response.get(f"{id_type}_id"),
        (response.get(id_type) or {}).get("id"),
        (response.get("data") or {}).get("id"),
        ((response.get("data") or {}).get(id_type) or {}).get("id"),
    ]

    for candidate in candidates:
        normalized = normalize_id(candidate)
        if normalized:
            return normalized
    return None


def _extract_conversation_id(ticket: Optional[dict]) -> Optional[str]:
    """Extract conversation ID from a FreeScout ticket response."""
    return _extract_id(ticket, "conversation")


def _create_and_record_draft(
    client: FreeScoutClient,
    store: TicketStore,
    conversation_id: str,
    reply_text: str,
    reply_hash: str,
    generated_at: str,
    max_drafts: Optional[int] = None,
    use_atomic_limit: bool = False,
) -> bool:
    """Create a new draft on FreeScout and record it locally.

    Args:
        use_atomic_limit: If True, use atomic upsert with draft limit check
            (for brand-new drafts). If False, use plain upsert (for replacements).
    """
    try:
        response = client.create_agent_draft_reply(conversation_id, reply_text)
    except requests.RequestException as exc:
        log_event(
            "freescout_draft",
            action="create_draft",
            outcome="failed",
            conversation_id=conversation_id,
            reason=str(exc),
        )
        return False

    thread_id = _extract_thread_id(response)

    if use_atomic_limit:
        if not store.atomic_upsert_bot_draft_if_under_limit(
            conversation_id, thread_id, reply_hash, generated_at, max_drafts
        ):
            log_event(
                "freescout_draft",
                action="create_draft",
                outcome="skipped",
                reason="draft_limit_reached_atomic",
                conversation_id=conversation_id,
            )
            return False
    else:
        store.upsert_bot_draft(conversation_id, thread_id, reply_hash, generated_at)

    log_event(
        "freescout_draft",
        action="create_draft",
        outcome="success",
        conversation_id=conversation_id,
        thread_id=thread_id,
    )
    return True


def _post_write_draft_reply(
    client: Optional[FreeScoutClient],
    store: TicketStore,
    conversation_id: Optional[str],
    subject: str,
    sender: str,
    body_text: str,
    max_drafts: Optional[int] = None,
    email_type: str = "other",
) -> bool:
    conversation_id = normalize_id(conversation_id)
    if not client or not conversation_id:
        return False

    # Pre-check if we can create a draft before calling the external API
    # This prevents orphaned drafts on FreeScout when the limit is reached
    if not store.can_create_new_draft(conversation_id, max_drafts):
        log_event(
            "freescout_draft",
            action="create_draft",
            outcome="skipped",
            reason="draft_limit_reached_precheck",
            conversation_id=conversation_id,
        )
        return False

    existing = store.get_bot_draft(conversation_id)

    # For existing drafts, check if the draft was modified externally before
    # generating an AI reply. This avoids wasting API budget on replies that
    # cannot be used.
    if existing:
        try:
            details = client.get_conversation(conversation_id)
        except requests.RequestException as exc:
            log_event(
                "freescout_draft",
                action="fetch_conversation",
                outcome="failed",
                conversation_id=conversation_id,
                reason=str(exc),
            )
            return False

        threads = _extract_conversation_threads(details)
        stored_thread_id = existing.get("freescout_thread_id")

        thread = _find_thread_by_id(threads, stored_thread_id) if stored_thread_id else None

        # If thread exists, check if it was modified externally before generating AI reply
        if thread:
            if not _thread_is_draft(thread) or not existing.get("last_hash"):
                _post_draft_skip_note(client, conversation_id, stored_thread_id)
                log_event(
                    "freescout_draft",
                    action="update_draft",
                    outcome="skipped",
                    conversation_id=conversation_id,
                    thread_id=stored_thread_id,
                    reason="draft updated outside bot",
                )
                return False

            current_text = _thread_text(thread)
            current_hash = _hash_draft_text(current_text)
            if current_hash != existing.get("last_hash"):
                _post_draft_skip_note(client, conversation_id, stored_thread_id)
                log_event(
                    "freescout_draft",
                    action="update_draft",
                    outcome="skipped",
                    conversation_id=conversation_id,
                    thread_id=stored_thread_id,
                    reason="draft edited",
                )
                return False

    # Generate AI reply only after confirming we can use it
    reply_text = generate_ai_reply(subject, sender, body_text, email_type)
    reply_hash = _hash_draft_text(reply_text)
    generated_at = datetime.now(timezone.utc).isoformat()

    # No existing draft: create new with atomic limit check
    if not existing:
        return _create_and_record_draft(
            client, store, conversation_id, reply_text, reply_hash,
            generated_at, max_drafts=max_drafts, use_atomic_limit=True,
        )

    # Re-use the stored_thread_id and thread from the validation above
    stored_thread_id = existing.get("freescout_thread_id")
    thread = _find_thread_by_id(threads, stored_thread_id) if stored_thread_id else None

    # Existing draft record but no thread ID or thread not found: create replacement
    if not thread:
        return _create_and_record_draft(
            client, store, conversation_id, reply_text, reply_hash, generated_at,
        )

    try:
        client.update_thread(conversation_id, stored_thread_id, reply_text, draft=True)
    except requests.RequestException as exc:
        log_event(
            "freescout_draft",
            action="update_draft",
            outcome="failed",
            conversation_id=conversation_id,
            thread_id=stored_thread_id,
            reason=str(exc),
        )
        return False

    store.upsert_bot_draft(conversation_id, stored_thread_id, reply_hash, generated_at)
    log_event(
        "freescout_draft",
        action="update_draft",
        outcome="success",
        conversation_id=conversation_id,
        thread_id=stored_thread_id,
    )
    return True


def _maybe_write_draft_reply(
    client: Optional[FreeScoutClient],
    store: TicketStore,
    conversation_id: Optional[str],
    subject: str,
    sender: str,
    body_text: str,
    settings: dict,
    email_type: str = "other",
) -> bool:
    # Draft limit is now enforced atomically in _post_write_draft_reply
    # via atomic_upsert_bot_draft_if_under_limit
    max_drafts = settings.get("MAX_DRAFTS")
    return _post_write_draft_reply(
        client,
        store,
        conversation_id,
        subject,
        sender,
        body_text,
        max_drafts=max_drafts,
        email_type=email_type,
    )


def _hash_draft_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_conversation_threads(conversation: dict) -> List[dict]:
    threads = conversation.get("threads") or conversation.get("data") or []
    if isinstance(threads, dict):
        threads = threads.get("threads", [])
    if not isinstance(threads, list):
        return []
    return [thread for thread in threads if isinstance(thread, dict)]


def _find_thread_by_id(threads: List[dict], thread_id: Optional[object]) -> Optional[dict]:
    if not thread_id:
        return None
    thread_id_str = str(thread_id)
    for thread in threads:
        if str(thread.get("id")) == thread_id_str:
            return thread
    return None


def _thread_is_draft(thread: dict) -> bool:
    if "draft" in thread:
        return bool(thread.get("draft"))
    state = thread.get("state") or thread.get("status")
    if state:
        return str(state).lower() == "draft"
    return False


def _thread_text(thread: dict) -> str:
    return thread.get("text") or thread.get("body") or ""


def _extract_thread_id(response: Optional[dict]) -> Optional[str]:
    """Extract thread ID from a FreeScout thread response."""
    return _extract_id(response, "thread")


def _post_draft_skip_note(
    client: FreeScoutClient,
    conversation_id: str,
    thread_id: Optional[str],
) -> None:
    message = "Skipped updating the AI draft because it appears to have been edited."
    if thread_id:
        message = f"{message} (draft thread {thread_id})"
    try:
        client.add_internal_note(conversation_id, message)
    except requests.RequestException as exc:
        log_event(
            "freescout_draft",
            action="add_skip_note",
            outcome="failed",
            conversation_id=conversation_id,
            thread_id=thread_id,
            reason=str(exc),
        )


def fetch_recent_conversations(
    since_iso: Optional[str] = None,
    timeout: Optional[int] = None,
    client: Optional[FreeScoutClient] = None,
):
    """Return list of recent FreeScout conversations since a given ISO time.

    Args:
        since_iso: ISO timestamp to filter conversations updated after this time
        timeout: HTTP timeout override
        client: Optional FreeScoutClient to reuse (avoids creating a new session)
    """
    params: Dict[str, str] = {}
    if since_iso:
        params["updated_since"] = since_iso

    owns_client = False
    if client is None:
        client = _build_freescout_client(timeout=timeout)
        owns_client = True
    if not client:
        return []

    try:
        return client.list_conversations(params)
    except requests.RequestException as e:
        log_event(
            "freescout_poll",
            action="fetch_recent",
            outcome="failed",
            reason=str(e),
        )
        return []
    finally:
        if owns_client:
            client.close()


def send_update_email(service, summary: str, label_id: Optional[str] = None):
    msg = create_base64_message("me", "me", "FreeScout Updates", summary)
    if label_id:
        msg["labelIds"] = [label_id]
    try:
        service.users().messages().send(userId="me", body=msg).execute()
    except HttpError as exc:
        log_event(
            "gmail_send",
            action="send_update_email",
            outcome="failed",
            reason=str(exc),
        )


def process_freescout_conversation(
    client: FreeScoutClient,
    conversation: dict,
    settings: dict,
):
    conv_id = normalize_id(conversation.get("id"))
    if not conv_id:
        return

    try:
        details = client.get_conversation(conv_id)
    except requests.RequestException as exc:
        log_event(
            "freescout_poll",
            action="fetch_conversation",
            outcome="failed",
            conversation_id=conv_id,
            reason=str(exc),
        )
        return

    subject = details.get("subject") or conversation.get("subject") or "(no subject)"
    latest_text = _extract_latest_thread_text(details) or conversation.get("last_text", "")
    # Extract sender from FreeScout conversation customer info
    customer = details.get("customer") or conversation.get("customer") or {}
    sender = (
        details.get("customerEmail")
        or customer.get("email")
        or customer.get("name")
        or "customer"
    )

    process_conversation(
        conv_id,
        {
            "subject": subject,
            "sender": sender,
            "latest_text": latest_text,
        },
        settings,
        client,
    )


def poll_freescout_updates(
    interval: int = 300, timeout: Optional[int] = None
):
    """Continuously poll FreeScout and classify new/updated conversations.

    Supports graceful shutdown via SIGINT/SIGTERM signals.
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    settings = get_settings()
    if settings.get("FREESCOUT_WEBHOOK_ENABLED"):
        log_event(
            "freescout_poll",
            action="poll_updates",
            outcome="skipped",
            reason="webhook ingestion enabled",
        )
        return
    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    client = _build_freescout_client(timeout=http_timeout)
    if not client:
        log_event(
            "freescout_poll",
            action="poll_updates",
            outcome="skipped",
            reason="ticket system not freescout",
        )
        return

    since = datetime.now(timezone.utc) - timedelta(minutes=5)
    # Track config fingerprint to only rebuild client when settings change
    _last_client_config: Optional[tuple] = None
    while not is_shutdown_requested():
        try:
            reload_settings()
            settings = get_settings()
            if settings.get("FREESCOUT_WEBHOOK_ENABLED"):
                log_event(
                    "freescout_poll",
                    action="poll_updates",
                    outcome="skipped",
                    reason="webhook ingestion enabled",
                )
                return
            http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
            # Only rebuild client when relevant settings change
            current_config = (
                settings.get("FREESCOUT_URL"),
                settings.get("FREESCOUT_KEY"),
                http_timeout,
                settings.get("TICKET_SYSTEM"),
            )
            if current_config != _last_client_config:
                if client:
                    client.close()
                client = _build_freescout_client(timeout=http_timeout)
                _last_client_config = current_config
            if not client:
                log_event(
                    "freescout_poll",
                    action="poll_updates",
                    outcome="skipped",
                    reason="ticket system not freescout",
                )
                return

            # Record poll start time BEFORE fetching to avoid missing updates
            poll_start = datetime.now(timezone.utc)
            convs = fetch_recent_conversations(
                since.isoformat(), timeout=http_timeout, client=client
            )
            for conv in convs:
                if is_shutdown_requested():
                    break
                process_freescout_conversation(client, conv, settings)
            since = poll_start  # Use poll start time to avoid gaps
            _shutdown_event.wait(interval)
        except Exception as exc:
            log_event(
                "freescout_poll",
                action="poll_updates",
                outcome="failed",
                error=str(exc),
            )
            # Sleep before retry to avoid busy-wait loop on persistent errors
            _shutdown_event.wait(min(interval, 60))

    if client:
        client.close()
    log_event(
        "freescout_poll",
        action="poll_updates",
        outcome="shutdown",
        reason="graceful shutdown requested",
    )


def _infer_webhook_outcome(payload: dict, headers: Optional[Dict[str, str]] = None) -> WebhookOutcome:
    # FreeScout sends the event type in the X-FreeScout-Event header (e.g. "convo.created")
    # Also check payload fields for compatibility with other webhook sources
    headers = headers or {}
    event = str(
        headers.get("X-FreeScout-Event", "")
        or payload.get("event") or payload.get("event_type") or payload.get("type") or ""
    ).lower()
    thread_type = str(payload.get("thread_type") or payload.get("thread", "")).lower()
    is_draft = bool(payload.get("draft")) or "draft" in event or "draft" in thread_type
    if "filter" in event or payload.get("filtered") is True:
        return WebhookOutcome(action="filtered", drafted=is_draft)
    # FreeScout uses "convo.created"; also support "conversation.created" for other sources
    if ("convo.created" == event) or ("conversation" in event and "created" in event):
        return WebhookOutcome(action="created", drafted=is_draft)
    # FreeScout: "convo.customer.reply.created", "convo.agent.reply.created", "convo.note.created"
    if "reply" in event and "created" in event:
        return WebhookOutcome(action="appended", drafted=is_draft)
    if "note" in event and "created" in event:
        return WebhookOutcome(action="appended", drafted=is_draft)
    if "thread" in event and "created" in event:
        return WebhookOutcome(action="appended", drafted=is_draft)
    if ("message" in event) and "created" in event:
        return WebhookOutcome(action="appended", drafted=is_draft)
    # FreeScout status/assignment events
    if "status" in event or "assigned" in event or "moved" in event:
        return WebhookOutcome(action="processed", drafted=is_draft)
    return WebhookOutcome(action="processed", drafted=is_draft)


def freescout_webhook_handler(
    payload: dict, headers: dict
) -> Tuple[str, int, Optional[WebhookOutcome]]:
    """Generic webhook handler usable by Flask or FastAPI routes.

    Note: Secret validation is expected to be performed by the caller
    (e.g., the FastAPI route handler in webhook_server.py) before
    invoking this function.
    """

    reload_settings()
    settings = get_settings()

    if not payload:
        return "missing payload", 400, None

    raw_id = payload.get("conversation_id") or payload.get("id")
    is_valid, conv_id, validation_error = validate_conversation_id(raw_id)
    if not is_valid:
        return validation_error, 400, None

    client = _build_freescout_client(timeout=settings["HTTP_TIMEOUT"])
    if not client:
        return "freescout disabled", 503, None

    try:
        process_freescout_conversation(client, {"id": conv_id}, settings)
        return "ok", 200, _infer_webhook_outcome(payload, headers)
    finally:
        client.close()


def main():
    reload_settings()
    settings = get_settings()
    args = parse_args(settings)

    if args.status:
        sqlite_path = settings.get("TICKET_SQLITE_PATH") or "./csm.sqlite"
        with TicketStore(sqlite_path) as ticket_store:
            status_counts = ticket_store.get_processed_status_counts()
            draft_count = ticket_store.get_bot_draft_count()
            recent_failures = ticket_store.get_recent_failures(limit=10)
            log_event(
                "status_summary",
                success=status_counts.get("success", 0),
                filtered=status_counts.get("filtered", 0),
                failed=status_counts.get("failed", 0),
                bot_drafts=draft_count,
                recent_failures=recent_failures or [],
            )
        return

    if args.poll_freescout:
        if settings.get("FREESCOUT_WEBHOOK_ENABLED"):
            log_event(
                "freescout_poll",
                action="poll_updates",
                outcome="skipped",
                reason="webhook ingestion enabled",
            )
            return
        poll_freescout_updates(
            interval=args.poll_interval,
            timeout=args.timeout,
        )
        return

    sqlite_path = settings.get("TICKET_SQLITE_PATH") or "./csm.sqlite"
    with TicketStore(sqlite_path) as ticket_store:
        svc = get_gmail_service(use_console=args.console_auth)
        ticket_label_id = None
        if settings["TICKET_SYSTEM"] == "freescout":
            try:
                ticket_label_id = retry_request(
                    lambda: ensure_label(svc, TICKET_LABEL_NAME),
                    action_name="gmail.ensure_label",
                    exceptions=(HttpError,),
                    log_context={"label_name": TICKET_LABEL_NAME},
                )
            except HttpError as exc:
                log_event(
                    "gmail_label",
                    action="ensure_label",
                    outcome="failed",
                    reason=str(exc),
                    label_name=TICKET_LABEL_NAME,
                )
        client = _build_freescout_client(timeout=args.timeout)

        try:
            processed = 0
            created_conversations = 0
            appended_threads = 0
            drafted = 0
            skipped = 0
            filtered_terminal = 0
            failed = 0

            for ref in fetch_all_unread_messages(
                svc, query=args.gmail_query, limit=settings["MAX_MESSAGES_PER_RUN"]
            ):
                processed += 1
                result = process_gmail_message(ref, ticket_store, client, svc, ticket_label_id)
                if result.status in {"skipped_already_success", "skipped_already_claimed"}:
                    skipped += 1
                elif result.status == "filtered":
                    filtered_terminal += 1
                elif result.status == "freescout_appended":
                    appended_threads += 1
                elif result.status == "freescout_created":
                    created_conversations += 1
                elif result.status == "failed_retryable":
                    failed += 1
                elif result.status == "failed_permanent":
                    failed += 1
                if result.drafted:
                    drafted += 1

            updates = poll_ticket_updates(client=client)
            if updates:
                log_event(
                    "freescout_poll",
                    action="fetch_updates",
                    outcome="success",
                    count=len(updates),
                )
            log_event(
                "gmail_ingest_summary",
                processed=processed,
                created=created_conversations,
                appended=appended_threads,
                drafted=drafted,
                skipped=skipped,
                filtered=filtered_terminal,
                failed=failed,
            )
        finally:
            if client:
                client.close()


if __name__ == "__main__":
    main()
