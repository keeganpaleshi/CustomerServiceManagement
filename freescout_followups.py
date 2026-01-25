from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import smtplib
from typing import Dict, Iterable, List, Optional, Sequence

import requests

from utils import (
    FreeScoutClient,
    generate_ai_reply,
    get_settings,
    is_customer_thread,
    log_event,
    normalize_id,
    parse_datetime,
    reload_settings,
    require_ticket_settings,
    thread_timestamp,
)


def parse_args(settings: Optional[Dict] = None):
    settings = settings or get_settings()
    followup_cfg = settings.get("FREESCOUT_FOLLOWUP", {})
    parser = argparse.ArgumentParser(
        description="Generate follow-up drafts for stale FreeScout conversations"
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=followup_cfg.get("hours_without_reply", 24),
        help="Minimum hours since last customer message",
    )
    parser.add_argument(
        "--required-tags",
        nargs="*",
        default=followup_cfg.get("required_tags", []),
        help="Only include conversations containing these tags",
    )
    parser.add_argument(
        "--excluded-tags",
        nargs="*",
        default=followup_cfg.get("excluded_tags", []),
        help="Exclude conversations containing these tags",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=followup_cfg.get("required_states", []),
        help="Only include conversations in these states/statuses",
    )
    parser.add_argument(
        "--followup-tag",
        default=followup_cfg.get("followup_tag", "followup-ready"),
        help="Tag applied to conversations with a drafted follow-up",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=followup_cfg.get("limit", 100),
        help="Maximum number of conversations to inspect",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=followup_cfg.get("dry_run", False),
        help="Log qualifying conversations without updating FreeScout",
    )
    return parser.parse_args()


def _extract_tags(conversation: dict) -> list[str]:
    tags = conversation.get("tags") or []
    if isinstance(tags, dict):
        tags = tags.get("data") or tags.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    normalized = []
    for tag in tags or []:
        if isinstance(tag, dict):
            name = tag.get("name") or tag.get("slug") or ""
        else:
            name = str(tag)
        if name:
            normalized.append(name)
    return normalized


def _status_values(conversation: dict) -> set[str]:
    status = conversation.get("status") or conversation.get("state")
    status_id = conversation.get("status_id")
    values = set()
    if isinstance(status, dict):
        for key in ("id", "name", "slug", "value"):
            if status.get(key) is not None:
                values.add(str(status.get(key)))
    elif status is not None:
        values.add(str(status))
    if status_id is not None:
        values.add(str(status_id))
    return {value.lower() for value in values}


def _matches_required(values: Iterable[str], required: Sequence[str]) -> bool:
    if not required:
        return True
    available = {value.lower() for value in values}
    return all(req.lower() in available for req in required)


def _matches_any(values: Iterable[str], required: Sequence[str]) -> bool:
    if not required:
        return True
    available = {value.lower() for value in values}
    return any(req.lower() in available for req in required)


def _has_excluded(values: Iterable[str], excluded: Sequence[str]) -> bool:
    if not excluded:
        return False
    available = {value.lower() for value in values}
    return any(excluded_tag.lower() in available for excluded_tag in excluded)


def _contains_tag(tags: Sequence[str], tag: str) -> bool:
    """Case-insensitive check if a tag is present in a list of tags."""
    tag_lower = tag.lower()
    return any(t.lower() == tag_lower for t in tags)


def _is_agent_thread(thread: dict) -> bool:
    if thread.get("type") in {"reply", "message"}:
        return True
    return bool(thread.get("user_id"))


def _extract_latest_customer_thread(threads: Sequence[dict]) -> Optional[dict]:
    latest: Optional[dict] = None
    latest_time: Optional[datetime] = None
    for thread in threads:
        if not is_customer_thread(thread):
            continue
        ts = thread_timestamp(thread)
        if ts and (latest_time is None or ts > latest_time):
            latest_time = ts
            latest = thread
    return latest


def _latest_timestamp(threads: Sequence[dict], predicate) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for thread in threads:
        if not predicate(thread):
            continue
        ts = thread_timestamp(thread)
        if ts and (latest is None or ts > latest):
            latest = ts
    return latest


def _conversation_threads(details: dict) -> list[dict]:
    threads = details.get("threads") or details.get("data") or []
    if isinstance(threads, dict):
        return threads.get("threads", [])
    if isinstance(threads, list):
        return threads
    return []


def _conversation_subject(details: dict) -> str:
    return details.get("subject") or "(no subject)"


def _conversation_sender(details: dict) -> str:
    customer = details.get("customer") or {}
    return (
        details.get("customer_email")
        or customer.get("email")
        or customer.get("name")
        or "customer"
    )


def _conversation_link(base_url: str, conv_id: str) -> str:
    return f"{base_url.rstrip('/')}/conversation/{conv_id}"


def _is_p0(conversation: dict, tags: Sequence[str], p0_tags: Sequence[str]) -> bool:
    tag_set = {tag.lower() for tag in tags}
    if any(p0_tag.lower() in tag_set for p0_tag in p0_tags):
        return True
    priority = conversation.get("priority")
    if isinstance(priority, dict):
        value = priority.get("name") or priority.get("value")
    else:
        value = priority
    return bool(value and str(value).lower() in {"urgent", "high", "p0"})


def _send_slack_notification(
    webhook_url: str, text: str, conversation_id: str
) -> None:
    try:
        requests.post(webhook_url, json={"text": text}, timeout=10).raise_for_status()
    except requests.RequestException as exc:
        log_event(
            "freescout_followup",
            action="notify_slack",
            outcome="failed",
            conversation_id=conversation_id,
            reason=str(exc),
        )


def _send_email_notification(email_cfg: dict, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = email_cfg["from"]
    message["To"] = email_cfg["to"]
    message["Subject"] = subject
    message.set_content(body)

    host = email_cfg["smtp_host"]
    # Default to port 587 for TLS, 465 for SSL, 25 for plain
    use_tls = email_cfg.get("use_tls", True)
    use_ssl = email_cfg.get("use_ssl", False)
    default_port = 465 if use_ssl else (587 if use_tls else 25)
    port = int(email_cfg.get("smtp_port", default_port))

    # Validate mutually exclusive TLS/SSL settings
    if use_tls and use_ssl:
        log_event(
            "freescout_followup",
            action="email_notification",
            outcome="config_warning",
            reason="Both use_tls and use_ssl are enabled; SSL takes precedence",
        )

    server = None
    # Default SMTP timeout of 30 seconds to prevent indefinite hangs
    smtp_timeout = int(email_cfg.get("timeout", 30))
    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=smtp_timeout)
        else:
            server = smtplib.SMTP(host, port, timeout=smtp_timeout)

        if use_tls and not use_ssl:
            server.starttls()
        username = email_cfg.get("smtp_username")
        password = email_cfg.get("smtp_password")
        if username and password:
            server.login(username, password)
        server.send_message(message)
    except smtplib.SMTPException as exc:
        log_event(
            "freescout_followup",
            action="email_notification",
            outcome="failed",
            reason=str(exc),
        )
        raise
    finally:
        if server is not None:
            try:
                server.quit()
            except smtplib.SMTPException:
                pass


def _notify_if_configured(
    settings: dict,
    conversation: dict,
    tags: Sequence[str],
    p0_tags: Sequence[str],
    followup_tag: str,
) -> None:
    followup_cfg = settings.get("FREESCOUT_FOLLOWUP", {})
    notify_cfg = followup_cfg.get("notify", {})
    slack_url = notify_cfg.get("slack_webhook_url")
    email_cfg = notify_cfg.get("email") or {}

    if not slack_url and not email_cfg:
        return

    if not _is_p0(conversation, tags, p0_tags):
        return

    conv_id = normalize_id(conversation.get("id"))
    if not conv_id:
        return
    base_url = settings.get("FREESCOUT_URL", "")
    link = _conversation_link(base_url, conv_id) if base_url else ""

    subject = f"[P0] Follow-up draft ready for conversation {conv_id}"
    body_lines = [
        f"Conversation ID: {conv_id}",
        f"Tags: {', '.join(tags) if tags else 'none'}",
        f"Follow-up tag: {followup_tag}",
    ]
    if link:
        body_lines.append(f"Link: {link}")
    body = "\n".join(body_lines)

    if slack_url:
        _send_slack_notification(slack_url, body, conv_id)
    if email_cfg.get("smtp_host") and email_cfg.get("from") and email_cfg.get("to"):
        _send_email_notification(email_cfg, subject, body)


def _iter_conversations(
    client: FreeScoutClient, params: dict, limit: int, max_pages: int = 100
) -> list[dict]:
    """
    Iterate through FreeScout conversations with pagination.

    Args:
        client: FreeScoutClient instance
        params: Query parameters
        limit: Maximum number of conversations to return
        max_pages: Maximum number of pages to fetch (prevents infinite loops)

    Returns:
        List of conversations up to limit
    """
    conversations: list[dict] = []
    page = 1
    page_size = min(50, limit)
    while len(conversations) < limit and page <= max_pages:
        page_params = dict(params)
        page_params["page"] = page
        page_params.setdefault("limit", page_size)
        data = client.list_conversations(page_params)
        if not data:
            break
        conversations.extend(data)
        if len(data) < page_params["limit"]:
            break
        page += 1

    if page > max_pages:
        log_event(
            "freescout_followup",
            action="pagination",
            outcome="max_pages_reached",
            max_pages=max_pages,
            conversations_fetched=len(conversations),
        )

    return conversations[:limit]


def main() -> None:
    reload_settings()
    settings = get_settings()
    args = parse_args(settings)
    if settings["TICKET_SYSTEM"] != "freescout":
        log_event(
            "freescout_followup",
            action="start",
            outcome="skipped",
            reason="ticket system not freescout",
        )
        return

    url, key = require_ticket_settings()
    client = FreeScoutClient(url, key, timeout=settings.get("HTTP_TIMEOUT", 15))
    followup_cfg = settings.get("FREESCOUT_FOLLOWUP", {})
    params = followup_cfg.get("list_params", {})
    p0_tags = followup_cfg.get("p0_tags", ["p0"])

    now = datetime.now(timezone.utc)
    min_age = timedelta(hours=args.hours)

    processed = 0
    qualified = 0
    drafted = 0
    filtered = 0
    failed = 0

    conversations = _iter_conversations(client, params, args.limit)
    for conversation in conversations:
        processed += 1
        conv_id = normalize_id(conversation.get("id"))
        if not conv_id:
            filtered += 1
            continue

        try:
            details = client.get_conversation(conv_id)
        except requests.RequestException as exc:
            failed += 1
            log_event(
                "freescout_followup",
                action="fetch_conversation",
                outcome="failed",
                conversation_id=conv_id,
                reason=str(exc),
            )
            continue

        tags = _extract_tags(details or conversation)
        if _has_excluded(tags, args.excluded_tags):
            filtered += 1
            continue
        if not _matches_required(tags, args.required_tags):
            filtered += 1
            continue
        if not _matches_any(_status_values(details or conversation), args.states):
            filtered += 1
            continue
        if _contains_tag(tags, args.followup_tag):
            filtered += 1
            continue

        threads = _conversation_threads(details)
        latest_customer = _extract_latest_customer_thread(threads)
        if not latest_customer:
            filtered += 1
            continue

        last_customer_time = thread_timestamp(latest_customer)
        if not last_customer_time or now - last_customer_time < min_age:
            filtered += 1
            continue

        last_agent_time = _latest_timestamp(threads, _is_agent_thread)
        if last_agent_time and last_agent_time >= last_customer_time:
            filtered += 1
            continue

        qualified += 1
        subject = _conversation_subject(details)
        sender = _conversation_sender(details)
        snippet = latest_customer.get("text") or latest_customer.get("body") or ""
        if not snippet:
            snippet = details.get("last_text", "") or ""

        draft = generate_ai_reply(subject, sender, snippet, "customer")
        if args.dry_run:
            log_event(
                "freescout_followup",
                action="draft_followup",
                outcome="dry_run",
                conversation_id=conv_id,
                followup_tag=args.followup_tag,
            )
            continue

        try:
            client.create_agent_draft_reply(conv_id, draft)
        except requests.RequestException as exc:
            failed += 1
            log_event(
                "freescout_followup",
                action="draft_followup",
                outcome="failed",
                conversation_id=conv_id,
                reason=str(exc),
            )
            continue

        # Avoid duplicate tags if followup_tag is somehow already present
        updated_tags = tags if _contains_tag(tags, args.followup_tag) else tags + [args.followup_tag]
        try:
            client.update_conversation(conv_id, tags=updated_tags)
        except requests.RequestException as exc:
            failed += 1
            log_event(
                "freescout_followup",
                action="tag_followup",
                outcome="failed",
                conversation_id=conv_id,
                reason=str(exc),
            )
            continue

        drafted += 1
        log_event(
            "freescout_followup",
            action="draft_followup",
            outcome="success",
            conversation_id=conv_id,
            followup_tag=args.followup_tag,
        )
        try:
            _notify_if_configured(
                settings,
                details or conversation,
                updated_tags,
                p0_tags,
                args.followup_tag,
            )
        except Exception as exc:
            # Don't increment failed counter here - the draft was created successfully,
            # only the optional notification failed. Log the error but don't affect stats.
            log_event(
                "freescout_followup",
                action="notify_followup",
                outcome="failed",
                conversation_id=conv_id,
                reason=str(exc),
            )

    log_event(
        "freescout_followup_summary",
        processed=processed,
        qualified=qualified,
        created=0,
        appended=0,
        drafted=drafted,
        filtered=filtered,
        failed=failed,
    )


if __name__ == "__main__":
    main()
