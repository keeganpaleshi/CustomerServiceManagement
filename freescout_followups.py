import argparse
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import smtplib
from typing import Dict, Iterable, Optional, Sequence

import requests

from Draft_Replies import generate_ai_reply
from utils import FreeScoutClient, get_settings, require_ticket_settings


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


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


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


def _thread_timestamp(thread: dict) -> Optional[datetime]:
    return _parse_datetime(thread.get("created_at") or thread.get("updated_at"))


def _is_customer_thread(thread: dict) -> bool:
    if thread.get("type") == "customer":
        return True
    return bool(thread.get("customer_id") and not thread.get("user_id"))


def _is_agent_thread(thread: dict) -> bool:
    if thread.get("type") in {"reply", "message"}:
        return True
    return bool(thread.get("user_id"))


def _extract_latest_customer_thread(threads: Sequence[dict]) -> Optional[dict]:
    latest: Optional[dict] = None
    latest_time: Optional[datetime] = None
    for thread in threads:
        if not _is_customer_thread(thread):
            continue
        ts = _thread_timestamp(thread)
        if ts and (latest_time is None or ts > latest_time):
            latest_time = ts
            latest = thread
    return latest


def _latest_timestamp(threads: Sequence[dict], predicate) -> Optional[datetime]:
    latest: Optional[datetime] = None
    for thread in threads:
        if not predicate(thread):
            continue
        ts = _thread_timestamp(thread)
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


def _send_slack_notification(webhook_url: str, text: str) -> None:
    requests.post(webhook_url, json={"text": text}, timeout=10).raise_for_status()


def _send_email_notification(email_cfg: dict, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = email_cfg["from"]
    message["To"] = email_cfg["to"]
    message["Subject"] = subject
    message.set_content(body)

    host = email_cfg["smtp_host"]
    port = int(email_cfg.get("smtp_port", 25))
    use_tls = email_cfg.get("use_tls", True)
    use_ssl = email_cfg.get("use_ssl", False)

    if use_ssl:
        server = smtplib.SMTP_SSL(host, port)
    else:
        server = smtplib.SMTP(host, port)

    try:
        if use_tls and not use_ssl:
            server.starttls()
        username = email_cfg.get("smtp_username")
        password = email_cfg.get("smtp_password")
        if username and password:
            server.login(username, password)
        server.send_message(message)
    finally:
        server.quit()


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

    conv_id = str(conversation.get("id"))
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
        _send_slack_notification(slack_url, body)
    if email_cfg.get("smtp_host") and email_cfg.get("from") and email_cfg.get("to"):
        _send_email_notification(email_cfg, subject, body)


def _iter_conversations(
    client: FreeScoutClient, params: dict, limit: int
) -> list[dict]:
    conversations: list[dict] = []
    page = 1
    page_size = min(50, limit)
    while len(conversations) < limit:
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
    return conversations[:limit]


def main() -> None:
    settings = get_settings()
    args = parse_args(settings)
    if settings["TICKET_SYSTEM"] != "freescout":
        print("FreeScout follow-up skipped because ticket system is not freescout.")
        return

    url, key = require_ticket_settings()
    client = FreeScoutClient(url, key, timeout=settings.get("HTTP_TIMEOUT", 15))
    followup_cfg = settings.get("FREESCOUT_FOLLOWUP", {})
    params = followup_cfg.get("list_params", {})
    p0_tags = followup_cfg.get("p0_tags", ["p0"])

    now = datetime.now(timezone.utc)
    min_age = timedelta(hours=args.hours)

    inspected = 0
    qualified = 0
    drafted = 0

    conversations = _iter_conversations(client, params, args.limit)
    for conversation in conversations:
        inspected += 1
        conv_id = conversation.get("id")
        if not conv_id:
            continue

        try:
            details = client.get_conversation(conv_id)
        except requests.RequestException as exc:
            print(f"Failed to fetch conversation {conv_id}: {exc}")
            continue

        tags = _extract_tags(details or conversation)
        if _has_excluded(tags, args.excluded_tags):
            continue
        if not _matches_required(tags, args.required_tags):
            continue
        if not _matches_any(_status_values(details or conversation), args.states):
            continue
        if args.followup_tag in tags:
            continue

        threads = _conversation_threads(details)
        latest_customer = _extract_latest_customer_thread(threads)
        if not latest_customer:
            continue

        last_customer_time = _thread_timestamp(latest_customer)
        if not last_customer_time or now - last_customer_time < min_age:
            continue

        last_agent_time = _latest_timestamp(threads, _is_agent_thread)
        if last_agent_time and last_agent_time >= last_customer_time:
            continue

        qualified += 1
        subject = _conversation_subject(details)
        sender = _conversation_sender(details)
        snippet = latest_customer.get("text") or latest_customer.get("body") or ""
        if not snippet:
            snippet = details.get("last_text", "") or ""

        draft = generate_ai_reply(subject, sender, snippet, "customer")
        if args.dry_run:
            print(f"{conv_id}: would add follow-up draft and tag '{args.followup_tag}'")
            continue

        try:
            client.add_draft_reply(conv_id, draft)
        except requests.RequestException as exc:
            print(f"Failed to add draft to {conv_id}: {exc}")
            continue

        updated_tags = tags + [args.followup_tag]
        try:
            client.update_conversation(conv_id, tags=updated_tags)
        except requests.RequestException as exc:
            print(f"Failed to tag conversation {conv_id}: {exc}")
            continue

        drafted += 1
        print(f"{conv_id}: follow-up draft created and tagged {args.followup_tag}")
        try:
            _notify_if_configured(
                settings,
                details or conversation,
                updated_tags,
                p0_tags,
                args.followup_tag,
            )
        except Exception as exc:
            print(f"Failed to send notification for {conv_id}: {exc}")

    print("Follow-up summary:")
    print(f"  inspected: {inspected}")
    print(f"  qualified: {qualified}")
    print(f"  drafted: {drafted}")


if __name__ == "__main__":
    main()
