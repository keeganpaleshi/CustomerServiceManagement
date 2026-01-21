import base64
import binascii
import copy
import json
import logging
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import yaml
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from openai import OpenAI


CONFIG_PATH = Path(__file__).with_name("config.yaml")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
LOGGER = logging.getLogger("csm")


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured log line."""

    payload = {
        "event": event,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(fields)
    LOGGER.log(level, json.dumps(payload, ensure_ascii=False, default=str))


@lru_cache(maxsize=1)
def _load_settings() -> Dict[str, Any]:
    """Load configuration and environment overrides lazily."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    return {
        "SCOPES": cfg["gmail"]["scopes"],
        "GMAIL_CLIENT_SECRET": cfg["gmail"]["client_secret_file"],
        "GMAIL_QUERY": cfg["gmail"].get("query", "is:unread"),
        "MAX_DRAFTS": cfg.get("limits", {}).get("max_drafts", 100),
        "MAX_MESSAGES_PER_RUN": cfg.get("limits", {}).get(
            "max_messages_per_run", 100
        ),
        "CRITIC_THRESHOLD": cfg["thresholds"]["critic_threshold"],
        "MAX_RETRIES": cfg["thresholds"]["max_retries"],
        "GMAIL_TOKEN_FILE": cfg["gmail"]["token_file"],
        "HTTP_TIMEOUT": cfg.get("http", {}).get("timeout", 15),
        "OPENAI_API_KEY": os.getenv(cfg["openai"]["api_key_env"]),
        "OPENAI_API_KEY_ENV": cfg["openai"]["api_key_env"],
        "DRAFT_MODEL": cfg["openai"]["draft_model"],
        "DRAFT_MAX_TOKENS": cfg["openai"].get("draft_max_tokens", 16384),
        "DRAFT_SYSTEM_MSG": cfg["openai"].get("draft_system_message", ""),
        "CLASSIFY_MODEL": cfg["openai"]["classify_model"],
        "CLASSIFY_MAX_TOKENS": cfg["openai"].get("classify_max_tokens", 50),
        "OPENAI_TIMEOUT": cfg["openai"].get("timeout", 30),
        "OPENAI_RATE_LIMIT": cfg["openai"].get("rate_limit", {}),
        "FALLBACK_SIGNATURE": cfg["openai"].get("fallback_signature", "Best,\nSupport Team"),
        "PROMO_LABELS": {
            "SPAM",
            "CATEGORY_PROMOTIONS",
            "CATEGORY_SOCIAL",
            "CATEGORY_UPDATES",
            "CATEGORY_FORUMS",
        },
        "GMAIL_USE_CONSOLE": cfg["gmail"].get("use_console_oauth", False),
        "TICKET_SYSTEM": cfg["ticket"]["system"],
        "FREESCOUT_URL": os.getenv("FREESCOUT_URL")
        or cfg["ticket"].get("freescout_url", ""),
        "FREESCOUT_KEY": os.getenv("FREESCOUT_KEY")
        or cfg["ticket"].get("freescout_key", ""),
        "FREESCOUT_MAILBOX_ID": cfg["ticket"].get("mailbox_id"),
        "FREESCOUT_GMAIL_THREAD_FIELD_ID": cfg["ticket"].get(
            "gmail_thread_field_id"
        ),
        "FREESCOUT_GMAIL_MESSAGE_FIELD_ID": cfg["ticket"].get(
            "gmail_message_field_id"
        ),
        "FREESCOUT_WEBHOOK_SECRET": cfg["ticket"].get("webhook_secret", ""),
        "FREESCOUT_WEBHOOK_ENABLED": cfg["ticket"].get("webhook_enabled", False),
        "FREESCOUT_POLL_INTERVAL": cfg["ticket"].get("poll_interval", 300),
        "FREESCOUT_ACTIONS": cfg["ticket"].get("actions", {}),
        "FREESCOUT_FOLLOWUP": cfg["ticket"].get("followup", {}),
        "TICKET_SQLITE_PATH": cfg["ticket"].get("sqlite_path", "./csm.sqlite"),
        "WEBHOOK_LOG_DIR": cfg.get("webhook", {}).get("log_dir", ""),
    }


def get_settings() -> Dict[str, Any]:
    """Public accessor for cached settings.

    Returns a deep copy to prevent accidental mutation of cached values,
    especially nested dictionaries like FREESCOUT_ACTIONS.
    """

    return copy.deepcopy(_load_settings())


def validate_settings() -> List[str]:
    """Validate required configuration values.

    Returns a list of validation error messages. Empty list means valid.
    """
    errors = []
    settings = _load_settings()

    # Check FreeScout settings if it's the configured ticket system
    if settings.get("TICKET_SYSTEM") == "freescout":
        if not settings.get("FREESCOUT_URL"):
            errors.append(
                "FREESCOUT_URL is required when ticket.system is 'freescout'. "
                "Set via environment variable or config.yaml ticket.freescout_url"
            )
        if not settings.get("FREESCOUT_KEY"):
            errors.append(
                "FREESCOUT_KEY is required when ticket.system is 'freescout'. "
                "Set via environment variable or config.yaml ticket.freescout_key"
            )
        if not settings.get("FREESCOUT_MAILBOX_ID"):
            errors.append(
                "FREESCOUT_MAILBOX_ID is required when ticket.system is 'freescout'. "
                "Set in config.yaml ticket.mailbox_id"
            )

    return errors


def reload_settings() -> None:
    """Clear cached settings after config or environment changes.

    Call this in long-running processes (or tests) when config.yaml or
    environment variables are updated and fresh settings are required.
    """

    _load_settings.cache_clear()


def require_openai_api_key() -> str:
    """Return the OpenAI API key or raise a clear error when missing."""

    api_key = _load_settings()["OPENAI_API_KEY"]
    if not api_key:
        env_name = _load_settings()["OPENAI_API_KEY_ENV"]
        raise RuntimeError(
            f"Please set your {env_name} environment variable before calling OpenAI helpers."
        )
    return api_key


def require_ticket_settings() -> tuple[str, str]:
    """Return ticket URL/key or raise when configuration is incomplete."""

    settings = _load_settings()
    url, key = settings["FREESCOUT_URL"], settings["FREESCOUT_KEY"]
    if not url or not key:
        raise RuntimeError(
            "Please set FREESCOUT_URL and FREESCOUT_KEY via environment variables or config.yaml."
        )
    return url, key


def serialize_custom_fields(field_map: Dict[Any, Any]) -> List[Dict[str, Any]]:
    """Serialize FreeScout custom fields to the expected list-of-dicts format."""

    serialized = []
    for key, value in (field_map or {}).items():
        if value is None or value == "":
            continue
        try:
            field_id = int(key)
        except (TypeError, ValueError):
            continue
        serialized.append({"id": field_id, "value": str(value)})

    return serialized


def retry_request(
    action: Callable[[], Any],
    *,
    action_name: str,
    max_attempts: Optional[int] = None,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
    log_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """Retry an action with exponential backoff and structured logging."""

    settings = get_settings()
    if max_attempts is None:
        max_attempts = int(settings.get("MAX_RETRIES", 3))
    max_attempts = max(1, max_attempts)
    context = log_context or {}

    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except exceptions as exc:
            is_last_attempt = attempt >= max_attempts
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            event = "retry_failed" if is_last_attempt else "retrying"
            log_payload = {
                "event": event,
                "action": action_name,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "error": str(exc),
                "delay_seconds": 0 if is_last_attempt else delay,
            }
            if context:
                log_payload["context"] = context
            log_event(
                log_payload.pop("event", "retrying"),
                level=logging.WARNING,
                **log_payload,
            )
            if is_last_attempt:
                raise
            time.sleep(delay)
    raise RuntimeError(f"Retry loop failed for {action_name}")


class SimpleRateLimiter:
    """Thread-safe rolling-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: deque = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        if self.max_requests <= 0 or self.window_seconds <= 0:
            return True
        with self._lock:
            now = time.monotonic()
            window_start = now - self.window_seconds
            # Remove timestamps that fall outside the sliding window.
            # Using < (not <=) keeps timestamps at exactly window_start,
            # making the window [window_start, now] inclusive on both ends.
            while self.timestamps and self.timestamps[0] < window_start:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.max_requests:
                return False
            self.timestamps.append(now)
            return True


_cached_rate_limiter: Optional[SimpleRateLimiter] = None
_rate_limiter_config: Optional[tuple] = None


def _get_openai_rate_limiter() -> Optional[SimpleRateLimiter]:
    """Get or create rate limiter, recreating if config changed."""
    global _cached_rate_limiter, _rate_limiter_config
    settings = _load_settings()
    rate_limit = settings.get("OPENAI_RATE_LIMIT", {}) or {}
    max_requests = rate_limit.get("max_requests", 0)
    window_seconds = rate_limit.get("window_seconds", 0)
    current_config = (max_requests, window_seconds)

    if not max_requests or not window_seconds:
        _cached_rate_limiter = None
        _rate_limiter_config = current_config
        return None

    if _rate_limiter_config != current_config:
        _cached_rate_limiter = SimpleRateLimiter(int(max_requests), int(window_seconds))
        _rate_limiter_config = current_config

    if _cached_rate_limiter is None:
        _cached_rate_limiter = SimpleRateLimiter(int(max_requests), int(window_seconds))
        _rate_limiter_config = current_config

    return _cached_rate_limiter


class FreeScoutClient:
    """Minimal FreeScout API helper for conversations."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def _normalize_id(value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _maybe_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-FreeScout-API-Key": self.api_key,
        }

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        conversation_id = self._normalize_id(conversation_id)
        resp = requests.get(
            f"{self.base_url}/api/conversations/{conversation_id}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        if isinstance(data, dict):
            return data.get("conversation") or data.get("data") or data
        return {}

    def list_conversations(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        resp = requests.get(
            f"{self.base_url}/api/conversations",
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        if isinstance(payload, dict):
            data = payload.get("data") or payload.get("conversations") or []
            return data if isinstance(data, list) else []
        if isinstance(payload, list):
            return payload
        return []

    def update_conversation(
        self,
        conversation_id: str,
        priority: Optional[object] = None,
        assignee: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        conversation_id = self._normalize_id(conversation_id)
        payload: Dict[str, Any] = {}
        if priority is not None:
            bucket_priority = {"P0": 4, "P1": 3, "P2": 2, "P3": 1}
            normalized = (
                priority.strip().upper() if isinstance(priority, str) else priority
            )
            priority_value: Optional[int] = None
            if isinstance(normalized, str) and normalized in bucket_priority:
                priority_value = bucket_priority[normalized]
            else:
                try:
                    numeric_priority = int(normalized)
                except (TypeError, ValueError):
                    numeric_priority = None
                if numeric_priority in {1, 2, 3, 4}:
                    priority_value = numeric_priority
            if priority_value is not None:
                payload["priority"] = priority_value
            else:
                log_event(
                    "freescout.invalid_priority",
                    level=logging.WARNING,
                    conversation_id=conversation_id,
                    priority=priority,
                )
        if assignee is not None:
            payload["user_id"] = self._maybe_int(assignee) or assignee
        if tags is not None:
            payload["tags"] = tags
        serialized = serialize_custom_fields(custom_fields or {})
        if serialized:
            payload["customFields"] = serialized

        if not payload:
            return {}

        resp = requests.put(
            f"{self.base_url}/api/conversations/{conversation_id}",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def add_customer_thread(
        self, conversation_id: str, text: str, imported: bool = True
    ) -> Dict[str, Any]:
        conversation_id = self._normalize_id(conversation_id)
        payload: Dict[str, Any] = {"type": "customer", "text": text, "imported": imported}

        def _request() -> Dict[str, Any]:
            resp = requests.post(
                f"{self.base_url}/api/conversations/{conversation_id}/threads",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        return retry_request(
            _request,
            action_name="freescout.add_customer_thread",
            exceptions=(requests.RequestException,),
            log_context={"conversation_id": conversation_id},
        )

    def create_conversation(
        self,
        subject: str,
        sender: str,
        body: str,
        mailbox_id: str,
        *,
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        gmail_thread_field: Optional[str] = None,
        gmail_message_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        mailbox_id = self._normalize_id(mailbox_id)
        custom_fields: Dict[str, Any] = {}
        tags: List[str] = []

        if thread_id:
            if gmail_thread_field:
                custom_fields[str(gmail_thread_field)] = thread_id
            else:
                tags.append(f"gmail_thread:{thread_id}")

        if message_id:
            if gmail_message_field:
                custom_fields[str(gmail_message_field)] = message_id
            else:
                tags.append(f"gmail_message:{message_id}")

        thread_payload = {
            "type": "customer",
            "text": body or "(no body)",
            "imported": True,
        }

        payload = {
            "type": "email",
            "mailboxId": self._maybe_int(mailbox_id) or mailbox_id,
            "subject": subject or "(no subject)",
            "customerEmail": sender,
            "customerName": sender,
            "imported": True,
            "threads": [thread_payload],
        }

        if tags:
            payload["tags"] = tags

        serialized = serialize_custom_fields(custom_fields)
        if serialized:
            payload["customFields"] = serialized

        def _request() -> Dict[str, Any]:
            resp = requests.post(
                f"{self.base_url}/api/conversations",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        return retry_request(
            _request,
            action_name="freescout.create_conversation",
            exceptions=(requests.RequestException,),
            log_context={"mailbox_id": mailbox_id},
        )

    def add_internal_note(
        self, conversation_id: str, text: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        conversation_id = self._normalize_id(conversation_id)
        payload: Dict[str, Any] = {"type": "note", "text": text}
        if user_id:
            payload["user_id"] = self._maybe_int(user_id) or user_id
        resp = requests.post(
            f"{self.base_url}/api/conversations/{conversation_id}/threads",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def add_draft_reply(
        self, conversation_id: str, text: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        draft_text = f"Follow-up draft (not sent):\n\n{text}"
        return self.add_internal_note(conversation_id, draft_text, user_id=user_id)

    def add_suggested_reply(
        self, conversation_id: str, text: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.add_internal_note(conversation_id, text, user_id=user_id)

    def create_agent_draft_reply(
        self,
        conversation_id: str,
        text: str,
        user_id: Optional[str] = None,
        draft: bool = True,
    ) -> Dict[str, Any]:
        conversation_id = self._normalize_id(conversation_id)
        payload: Dict[str, Any] = {"type": "reply", "text": text}
        if user_id:
            payload["user_id"] = self._maybe_int(user_id) or user_id
        if draft:
            payload["draft"] = True

        def _request() -> Dict[str, Any]:
            resp = requests.post(
                f"{self.base_url}/api/conversations/{conversation_id}/threads",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        return retry_request(
            _request,
            action_name="freescout.create_agent_draft_reply",
            exceptions=(requests.RequestException,),
            log_context={"conversation_id": conversation_id},
        )

    def update_thread(
        self,
        conversation_id: str,
        thread_id: str,
        text: str,
        draft: bool = True,
    ) -> Dict[str, Any]:
        conversation_id = self._normalize_id(conversation_id)
        thread_id = self._normalize_id(thread_id)
        payload: Dict[str, Any] = {"text": text, "draft": draft}

        def _request() -> Dict[str, Any]:
            resp = requests.put(
                f"{self.base_url}/api/conversations/{conversation_id}/threads/{thread_id}",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        return retry_request(
            _request,
            action_name="freescout.update_thread",
            exceptions=(requests.RequestException,),
            log_context={
                "conversation_id": conversation_id,
                "thread_id": thread_id,
            },
        )


# ----- Gmail helpers -----


def get_gmail_service(
    creds_filename: Optional[str] = None,
    token_filename: Optional[str] = None,
    use_console: Optional[bool] = None,
):
    """Return an authenticated Gmail service instance."""

    settings = _load_settings()
    creds_filename = creds_filename or settings["GMAIL_CLIENT_SECRET"]
    token_filename = token_filename or settings["GMAIL_TOKEN_FILE"]
    creds = None
    use_console = (
        use_console
        if use_console is not None
        else settings.get("GMAIL_USE_CONSOLE", False)
    )
    if os.path.exists(token_filename):
        try:
            with open(token_filename, "r", encoding="utf-8") as t:
                creds = Credentials.from_authorized_user_info(
                    json.load(t), settings["SCOPES"]
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log_event(
                "gmail_auth",
                level=logging.WARNING,
                action="load_credentials",
                outcome="failed",
                reason=f"Invalid token file, will re-authenticate: {e}",
            )
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                log_event(
                    "gmail_auth",
                    level=logging.WARNING,
                    action="refresh_credentials",
                    outcome="failed",
                    reason=f"Token refresh failed, will re-authenticate: {e}",
                )
                creds = None
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_filename, settings["SCOPES"]
            )
            if use_console:
                creds = flow.run_console()
            else:
                creds = flow.run_local_server(port=0)
        # Save credentials as JSON instead of pickle for security
        with open(token_filename, "w", encoding="utf-8") as t:
            # Convert credentials to JSON-serializable format
            creds_data = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }
            json.dump(creds_data, t, indent=2)
    return build("gmail", "v1", credentials=creds)


def fetch_all_unread_messages(service, query, limit=None):
    unread, token = [], None
    while True:
        # Apply limit to maxResults if specified to avoid over-fetching
        list_params = {"userId": "me", "q": query}
        if token:
            list_params["pageToken"] = token
        if limit is not None:
            # Request only what we still need (up to 500 per Gmail API limits)
            remaining = limit - len(unread)
            if remaining <= 0:
                break
            list_params["maxResults"] = min(remaining, 500)

        resp = service.users().messages().list(**list_params).execute()
        messages = resp.get("messages", [])
        unread.extend(messages)

        # Stop if we've reached the limit
        if limit is not None and len(unread) >= limit:
            break

        token = resp.get("nextPageToken")
        if not token:
            break
    return unread


def decode_base64url(data: str) -> str:
    """Decode base64url strings that may be missing padding."""

    if not data:
        return ""

    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode((data + padding).encode("utf-8")).decode(
            "utf-8", "ignore"
        )
    except (binascii.Error, ValueError):
        return ""


def extract_plain_text(payload: Optional[dict], max_depth: int = 50) -> str:
    """Recursively search a payload tree for the first text/plain body.

    Args:
        payload: The email payload dictionary
        max_depth: Maximum recursion depth to prevent stack overflow (default: 50)

    Returns:
        The extracted plain text or empty string
    """
    if not payload or max_depth <= 0:
        return ""

    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return decode_base64url(body_data)

    for part in payload.get("parts", []):
        text = extract_plain_text(part, max_depth - 1)
        if text:
            return text

    if body_data:
        return decode_base64url(body_data)

    return ""


def create_base64_message(sender, to, subject, body):
    msg = MIMEText(body)
    msg["to"], msg["from"], msg["subject"] = to, sender, subject
    return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}


def ensure_label(service, label_name: str) -> Optional[str]:
    """Return an existing label's ID, creating it when missing."""

    labels = (
        service.users().labels().list(userId="me", fields="labels/id,labels/name").execute()
    ).get("labels", [])
    for label in labels:
        if label.get("name") == label_name:
            return label.get("id")

    body = {
        "name": label_name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
    }
    created = service.users().labels().create(userId="me", body=body).execute()
    return created.get("id")


def apply_label_to_thread(service, thread_id: str, label_id: str) -> bool:
    """Add a label to a thread; return True on success."""

    service.users().threads().modify(
        userId="me", id=thread_id, body={"addLabelIds": [label_id]}
    ).execute()
    return True


def is_promotional_or_spam(message, body_text):
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in message.get("payload", {}).get("headers", [])
    }
    body_text = body_text or ""
    body_lower = body_text.lower()
    subject = headers.get("subject", "").lower()
    from_header = headers.get("from", "").lower()

    list_header_names = {
        "list-unsubscribe",
        "list-id",
        "list-help",
        "list-subscribe",
        "list-post",
        "list-archive",
        "list-owner",
    }
    list_headers_present = list_header_names.intersection(headers)
    list_unsubscribe_present = "list-unsubscribe" in list_headers_present

    spam_flag = headers.get("x-spam-flag", "").strip().lower() == "yes"
    spam_status = headers.get("x-spam-status", "").lower().startswith("yes")
    if spam_flag or spam_status:
        return True

    scl_header = headers.get("x-ms-exchange-organization-scl", "").strip()
    try:
        if scl_header and int(scl_header) >= 5:
            return True
    except ValueError:
        pass

    precedence = headers.get("precedence", "").strip().lower()
    bulk_header = precedence in {"bulk", "list", "junk"}
    auto_submitted = headers.get("auto-submitted", "").strip().lower()
    auto_generated = auto_submitted in {"auto-generated", "auto-replied"}

    footer_patterns = [
        r"(?im)^\s*unsubscribe\b",
        r"(?im)\bclick here to unsubscribe\b",
        r"(?im)\bto unsubscribe\b",
        r"(?im)\bmanage (your )?(email )?preferences\b",
        r"(?im)\bupdate (your )?(email )?preferences\b",
        r"(?im)\bmanage subscriptions?\b",
        r"(?im)\bopt[- ]?out\b",
    ]
    footer_match = any(re.search(pattern, body_text) for pattern in footer_patterns)
    negated_unsubscribe = re.search(
        r"\b(do\s*not|don't|dont|no)\s+(want\s+to\s+)?unsubscribe\b",
        body_lower,
    )
    if footer_match and negated_unsubscribe:
        footer_match = False

    promo_subject = re.search(
        r"\b(newsletter|promo|promotion|sale|deal|discount|offer|coupon|limited time|free trial)\b",
        subject,
    )

    header_signals = 0
    if list_unsubscribe_present:
        header_signals += 2
    if list_headers_present:
        header_signals += 1
    if bulk_header:
        header_signals += 2
    if auto_generated:
        header_signals += 1

    if footer_match and header_signals >= 1:
        return True

    noreply_sender = "no-reply" in from_header or "noreply@" in from_header
    if list_unsubscribe_present and promo_subject and noreply_sender:
        return True

    score = header_signals
    if footer_match:
        score += 2
    if promo_subject:
        score += 1
    if noreply_sender:
        score += 1

    return score >= 5


def _get_fallback_reply(settings: Dict[str, Any]) -> str:
    """Return a fallback reply when AI generation fails or is rate-limited."""
    fallback_body = (
        "Hello,\n\n"
        "I'm sorry, but I couldn't generate a response at this time. "
        "Please review this email manually.\n\n"
    )
    fallback_signature = settings.get("FALLBACK_SIGNATURE", "Best,\nSupport Team")
    return fallback_body + fallback_signature


def generate_ai_reply(subject, sender, snippet_or_body, email_type):
    """
    Generate a draft reply using OpenAI's new library (>=1.0.0).
    """
    settings = get_settings()
    limiter = _get_openai_rate_limiter()
    if limiter and not limiter.allow():
        log_event(
            "openai_throttled",
            level=logging.WARNING,
            action="generate_ai_reply",
            reason="rate_limited",
        )
        return _get_fallback_reply(settings)
    try:
        client = OpenAI(
            api_key=require_openai_api_key(), timeout=settings["OPENAI_TIMEOUT"]
        )
        instructions = (
            f"[Email type: {email_type}]\n\n"
            "You are an AI email assistant. The user received an email.\n"
            f"Subject: {subject}\n"
            f"From: {sender}\n"
            f"Email content/snippet: {snippet_or_body}\n\n"
            "Please write a friendly and professional draft reply addressing the sender's query. "
            "Return only the draft reply text without analysis, reasoning, or extra labels."
        )
        response = client.chat.completions.create(
            model=settings["DRAFT_MODEL"],
            messages=[
                {"role": "system", "content": settings["DRAFT_SYSTEM_MSG"]},
                {"role": "user", "content": instructions},
            ],
            max_tokens=settings["DRAFT_MAX_TOKENS"],
            temperature=0.7,
        )
        # Check if response has choices before accessing
        if not response.choices:
            raise ValueError("OpenAI response contains no choices")
        raw_reply = response.choices[0].message.content
        if raw_reply is None:
            raise ValueError("OpenAI response content is None")
        return sanitize_draft_reply(raw_reply.strip())
    except Exception as exc:
        log_event(
            "openai_error",
            level=logging.ERROR,
            action="generate_ai_reply",
            reason=str(exc),
            error_type=type(exc).__name__,
        )
        return _get_fallback_reply(settings)


def sanitize_draft_reply(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    for label in ("Final:", "Reply:", "Response:", "Draft reply:"):
        if label in cleaned:
            cleaned = cleaned.split(label)[-1].strip()

    cleaned_lines = []
    skip_reasoning = False
    for line in cleaned.splitlines():
        marker = line.strip().lower()
        if marker.startswith(("reasoning:", "analysis:", "thoughts:", "notes:")):
            skip_reasoning = True
            continue
        if skip_reasoning and marker == "":
            skip_reasoning = False
            continue
        if not skip_reasoning:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text.strip()


def classify_email(text):
    """Classify an email and return a dict with type, importance, and reasoning."""
    settings = get_settings()
    default_response = {
        "type": "other",
        "importance": 0,
        "reasoning": "",
        "facts": [],
        "uncertainty": [],
    }
    def is_valid_response(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        if not isinstance(payload.get("type"), str):
            return False
        if not isinstance(payload.get("importance"), (int, float)):
            return False
        if not isinstance(payload.get("reasoning"), str):
            return False
        if not isinstance(payload.get("facts"), list):
            return False
        if not isinstance(payload.get("uncertainty"), list):
            return False
        return True
    limiter = _get_openai_rate_limiter()
    if limiter and not limiter.allow():
        log_event(
            "openai_throttled",
            level=logging.WARNING,
            action="classify_email",
            reason="rate_limited",
        )
        return default_response
    try:
        client = OpenAI(
            api_key=require_openai_api_key(), timeout=settings["OPENAI_TIMEOUT"]
        )
        resp = client.chat.completions.create(
            model=settings["CLASSIFY_MODEL"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Categorize the email as lead, customer, or other. "
                        "Return ONLY JSON with keys "
                        "{\"type\":\"lead|customer|other\",\"importance\":1-10,"
                        "\"reasoning\":\"...\",\"facts\":[\"...\"],\"uncertainty\":[\"...\"]}. "
                        "Keep reasoning and lists concise. NO other text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=settings["CLASSIFY_MAX_TOKENS"],
        )
        # Check if response has choices before accessing
        if not resp.choices:
            raise ValueError("OpenAI response contains no choices")
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response content is None")
        parsed = json.loads(content)
        if not is_valid_response(parsed):
            return default_response
        # Clamp importance to valid 1-10 range
        importance = parsed.get("importance", 0)
        if isinstance(importance, (int, float)):
            parsed["importance"] = max(1, min(10, int(importance)))
        return parsed
    except Exception as exc:
        log_event(
            "openai_error",
            level=logging.ERROR,
            action="classify_email",
            reason=str(exc),
            error_type=type(exc).__name__,
        )
        return default_response


def importance_to_bucket(importance_score: Optional[float]) -> str:
    """Map an importance score (1-10) to a P0-P3 bucket."""
    if importance_score is None:
        return "P3"
    try:
        score = float(importance_score)
    except (TypeError, ValueError):
        return "P3"
    if score >= 9:
        return "P0"
    if score >= 7:
        return "P1"
    if score >= 4:
        return "P2"
    return "P3"


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string to datetime object with timezone support."""
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


def normalize_id(value: object) -> Optional[str]:
    """Normalize an ID value to a string, returning None for empty/null values."""
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def is_customer_thread(thread: dict) -> bool:
    """Check if a FreeScout thread is from a customer."""
    if thread.get("type") == "customer":
        return True
    return bool(thread.get("customer_id") and not thread.get("user_id"))


def thread_timestamp(thread: dict) -> Optional[datetime]:
    """Extract timestamp from a FreeScout thread."""
    return parse_datetime(thread.get("created_at") or thread.get("updated_at"))


