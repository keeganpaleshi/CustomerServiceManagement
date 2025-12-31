import base64
import json
import os
import pickle
import sqlite3
import time
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
import yaml
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI


CONFIG_PATH = Path(__file__).with_name("config.yaml")


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
        "FREESCOUT_POLL_INTERVAL": cfg["ticket"].get("poll_interval", 300),
        "FREESCOUT_ACTIONS": cfg["ticket"].get("actions", {}),
        "STATE_DB_PATH": (
            (cfg.get("state", {}) or {}).get("db_path")
            or str(Path(__file__).with_name("ingestion_state.sqlite"))
        ),
    }


def get_settings() -> Dict[str, Any]:
    """Public accessor for cached settings."""

    return _load_settings().copy()


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


class FreeScoutClient:
    """Minimal FreeScout API helper for conversations."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-FreeScout-API-Key": self.api_key,
        }

    def get_conversation(self, conversation_id: int) -> Dict[str, Any]:
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

    def update_conversation(
        self,
        conversation_id: int,
        priority: Optional[str] = None,
        assignee: Optional[int] = None,
        tags: Optional[list[str]] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if priority:
            payload["priority"] = priority
        if assignee is not None:
            payload["user_id"] = assignee
        if tags is not None:
            payload["tags"] = tags
        if custom_fields:
            payload["custom_fields"] = custom_fields

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

    def add_internal_note(
        self, conversation_id: int, text: str, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": "note", "text": text}
        if user_id:
            payload["user_id"] = user_id
        resp = requests.post(
            f"{self.base_url}/api/conversations/{conversation_id}/threads",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def add_suggested_reply(
        self, conversation_id: int, text: str, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        return self.add_internal_note(
            conversation_id,
            f"Suggested reply:\n\n{text}",
            user_id=user_id,
        )

    def add_customer_thread(
        self,
        conversation_id: int,
        body: str,
        *,
        customer_email: Optional[str] = None,
        imported: bool = True,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": "customer",
            "text": body or "(no body)",
            "imported": imported,
        }

        if customer_email:
            payload["customer"] = {"email": customer_email}

        if custom_fields:
            payload["customFields"] = custom_fields

        resp = requests.post(
            f"{self.base_url}/api/conversations/{conversation_id}/threads",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


class GmailIngestionDB:
    """Lightweight SQLite store for Gmail ingestion idempotency."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    conv_id INTEGER,
                    status TEXT DEFAULT 'success',
                    error TEXT,
                    processed_at REAL DEFAULT (strftime('%s','now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_conversations (
                    thread_id TEXT PRIMARY KEY,
                    conv_id INTEGER,
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
                """
            )
            cur = conn.execute("PRAGMA table_info(processed_messages)")
            columns = {row[1] for row in cur.fetchall()}
            if "status" not in columns:
                conn.execute(
                    "ALTER TABLE processed_messages ADD COLUMN status TEXT DEFAULT 'success'"
                )
            if "error" not in columns:
                conn.execute("ALTER TABLE processed_messages ADD COLUMN error TEXT")
            conn.commit()

    def processed_success(self, message_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT 1 FROM processed_messages
                WHERE message_id = ? AND status = 'success'
                """,
                (message_id,)
            )
            return cur.fetchone() is not None

    def get_conv_id(self, thread_id: str) -> Optional[int]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT conv_id FROM thread_conversations WHERE thread_id = ?", (thread_id,)
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None

    def set_thread_conv(self, thread_id: str, conv_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO thread_conversations(thread_id, conv_id)
                VALUES(?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET conv_id = excluded.conv_id,
                    updated_at = strftime('%s','now')
                """,
                (thread_id, conv_id),
            )
            conn.commit()

    def mark_success(self, message_id: str, thread_id: str, conv_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages(message_id, thread_id, conv_id, status, error, processed_at)
                VALUES(?, ?, ?, 'success', NULL, strftime('%s','now'))
                ON CONFLICT(message_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    conv_id = excluded.conv_id,
                    status = 'success',
                    error = NULL,
                    processed_at = excluded.processed_at
                """,
                (message_id, thread_id, conv_id),
            )
            conn.commit()

    def mark_failure(
        self, message_id: str, thread_id: str, conv_id: Optional[int], error: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages(message_id, thread_id, conv_id, status, error, processed_at)
                VALUES(?, ?, ?, 'failed', ?, strftime('%s','now'))
                ON CONFLICT(message_id) DO UPDATE SET
                    thread_id = COALESCE(excluded.thread_id, processed_messages.thread_id),
                    conv_id = COALESCE(excluded.conv_id, processed_messages.conv_id),
                    status = 'failed',
                    error = excluded.error,
                    processed_at = excluded.processed_at
                """,
                (message_id, thread_id, conv_id, error[:500] if error else None),
            )
            conn.commit()


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
        with open(token_filename, "rb") as t:
            creds = pickle.load(t)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_filename, settings["SCOPES"]
            )
            if use_console:
                creds = flow.run_console()
            else:
                creds = flow.run_local_server(port=0)
        with open(token_filename, "wb") as t:
            pickle.dump(creds, t)
    return build("gmail", "v1", credentials=creds)


def fetch_all_unread_messages(service, query):
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


def decode_base64url(data: str) -> str:
    """Decode base64url strings that may be missing padding."""

    if not data:
        return ""

    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode((data + padding).encode("utf-8")).decode(
            "utf-8", "ignore"
        )
    except (base64.binascii.Error, ValueError):
        return ""


def extract_plain_text(payload: Optional[dict]) -> str:
    """Recursively search a payload tree for the first text/plain body."""

    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return decode_base64url(body_data)

    for part in payload.get("parts", []) or []:
        text = extract_plain_text(part)
        if text:
            return text

    if body_data:
        return decode_base64url(body_data)

    return ""


def create_base64_message(sender, to, subject, body):
    msg = MIMEText(body)
    msg["to"], msg["from"], msg["subject"] = to, sender, subject
    return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}


def create_draft(service, user_id, msg_body, thread_id=None):
    data = {"message": msg_body}
    if thread_id:
        data["message"]["threadId"] = thread_id
    return service.users().drafts().create(userId=user_id, body=data).execute()


def thread_has_draft(service, thread_id):
    data = service.users().threads().get(userId="me", id=thread_id).execute()
    return any("DRAFT" in (m.get("labelIds") or []) for m in data.get("messages", []))


def ensure_label(service, label_name: str) -> Optional[str]:
    """Return an existing label's ID, creating it when missing."""

    try:
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
    except HttpError as exc:
        print(f"Error ensuring label '{label_name}': {exc}")
        return None


def apply_label_to_thread(service, thread_id: str, label_id: str) -> bool:
    """Add a label to a thread; return True on success."""

    try:
        service.users().threads().modify(
            userId="me", id=thread_id, body={"addLabelIds": [label_id]}
        ).execute()
        return True
    except HttpError as exc:
        print(f"Failed to apply label to thread {thread_id}: {exc}")
        return False


def is_promotional_or_spam(message, body_text):
    settings = _load_settings()
    labels = set(message.get("labelIds", []))
    if labels & settings["PROMO_LABELS"]:
        return True
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in message.get("payload", {}).get("headers", [])
    }
    if "list-unsubscribe" in headers or "list-id" in headers:
        return True
    if "unsubscribe" in body_text.lower():
        return True
    return False


def critic_email(draft, original):
    """Self-grade a draft reply using GPT-4.1."""
    client = OpenAI(api_key=require_openai_api_key())
    default_rating = {
        "score": 0,
        "feedback": "Critic check failed; please review manually.",
    }

    try:
        resp = client.chat.completions.create(
            model=_load_settings()["CLASSIFY_MODEL"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY JSON {\"score\":1-10,\"feedback\":\"...\"} rating on correctness, tone, length."
                    ),
                },
                {"role": "assistant", "content": draft},
                {"role": "user", "content": f"Original email:\n\n{original}"},
            ],
        )
        parsed = json.loads(resp.choices[0].message.content)
        if not isinstance(parsed, dict):
            return default_rating

        return {
            "score": parsed.get("score", 0),
            "feedback": parsed.get("feedback", ""),
        }
    except Exception as exc:
        print(f"Error critiquing email draft: {exc}")
        return default_rating


def classify_email(text):
    """Classify an email and return a dict with type and importance."""
    settings = _load_settings()
    client = OpenAI(api_key=require_openai_api_key())
    try:
        resp = client.chat.completions.create(
            model=settings["CLASSIFY_MODEL"],
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
            max_tokens=settings["CLASSIFY_MAX_TOKENS"],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


def create_ticket(
    subject: str,
    sender: str,
    body: str,
    *,
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
    timeout: Optional[int] = None,
    retries: int = 3,
):
    """Create a FreeScout conversation using a body/text thread payload.

    FreeScout examples show both `text` and `customFields`. We populate both
    `text` and `body` for the initial customer thread to keep payloads
    compatible with prior integrations while storing Gmail metadata in
    `customFields` when configured.
    """
    settings = _load_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return None

    url, key = require_ticket_settings()
    mailbox_id = settings.get("FREESCOUT_MAILBOX_ID")
    if not mailbox_id:
        raise RuntimeError("ticket.mailbox_id must be configured for FreeScout")

    custom_fields: Dict[str, Any] = {}
    gmail_thread_field = settings.get("FREESCOUT_GMAIL_THREAD_FIELD_ID")
    gmail_message_field = settings.get("FREESCOUT_GMAIL_MESSAGE_FIELD_ID")

    if gmail_thread_field and thread_id:
        custom_fields[str(gmail_thread_field)] = thread_id
    if gmail_message_field and message_id:
        custom_fields[str(gmail_message_field)] = message_id

    thread_payload = {
        "type": "customer",
        "text": body or "(no body)",
        "body": body or "(no body)",
        "imported": True,
    }

    payload = {
        "type": "email",
        "mailboxId": mailbox_id,
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "imported": True,
        "threads": [thread_payload],
    }

    if custom_fields:
        payload["customFields"] = custom_fields

    http_timeout = timeout if timeout is not None else settings["HTTP_TIMEOUT"]
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{url.rstrip('/')}/api/conversations",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-FreeScout-API-Key": key,
                },
                json=payload,
                timeout=http_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"Failed to create ticket after {retries} attempts: {exc}")
                return None
            delay = 2 ** (attempt - 1)
            print(f"Ticket creation error: {exc}. Retrying in {delay}s…")
            time.sleep(delay)
