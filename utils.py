import base64
import json
import os
import pickle
import time
from email.mime.text import MIMEText
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

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
        "FREESCOUT_FOLLOWUP": cfg["ticket"].get("followup", {}),
        "TICKET_SQLITE_PATH": cfg["ticket"].get("sqlite_path", "./csm.sqlite"),
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


def serialize_custom_fields(field_map: Dict[Any, Any]) -> list[dict]:
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

    def list_conversations(self, params: Optional[Dict[str, Any]] = None) -> list[dict]:
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
        conversation_id: int,
        priority: Optional[object] = None,
        assignee: Optional[int] = None,
        tags: Optional[list[str]] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if priority:
            bucket_priority = {"P0": 4, "P1": 3, "P2": 2, "P3": 1}
            if isinstance(priority, str) and priority in bucket_priority:
                payload["priority"] = bucket_priority[priority]
            else:
                payload["priority"] = priority
        if assignee is not None:
            payload["user_id"] = assignee
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
        self, conversation_id: int, text: str, imported: bool = True
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": "customer", "text": text, "imported": imported}
        resp = requests.post(
            f"{self.base_url}/api/conversations/{conversation_id}/threads",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def create_conversation(
        self,
        subject: str,
        sender: str,
        body: str,
        mailbox_id: int,
        *,
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        gmail_thread_field: Optional[int] = None,
        gmail_message_field: Optional[int] = None,
    ) -> Dict[str, Any]:
        custom_fields: Dict[str, Any] = {}
        tags: list[str] = []

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
            "mailboxId": mailbox_id,
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

        resp = requests.post(
            f"{self.base_url}/api/conversations",
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

    def add_draft_reply(
        self, conversation_id: int, text: str, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        draft_text = f"Follow-up draft (not sent):\n\n{text}"
        return self.add_internal_note(conversation_id, draft_text, user_id=user_id)

    def add_suggested_reply(
        self, conversation_id: int, text: str, user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        return self.add_internal_note(conversation_id, text, user_id=user_id)

    def create_agent_draft_reply(
        self,
        conversation_id: int,
        text: str,
        user_id: Optional[int] = None,
        draft: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": "reply", "text": text}
        if user_id:
            payload["user_id"] = user_id
        if draft:
            payload["draft"] = True
        resp = requests.post(
            f"{self.base_url}/api/conversations/{conversation_id}/threads",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def update_thread(
        self,
        conversation_id: int,
        thread_id: int,
        text: str,
        draft: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text, "draft": draft}
        resp = requests.put(
            f"{self.base_url}/api/conversations/{conversation_id}/threads/{thread_id}",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


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
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in message.get("payload", {}).get("headers", [])
    }
    if "list-unsubscribe" in headers or "list-id" in headers:
        return True
    if "unsubscribe" in body_text.lower():
        return True
    return False


def generate_ai_reply(subject, sender, snippet_or_body, email_type):
    """
    Generate a draft reply using OpenAI's new library (>=1.0.0).
    """
    settings = get_settings()
    client = OpenAI(api_key=require_openai_api_key())

    instructions = (
        f"[Email type: {email_type}]\n\n"
        "You are an AI email assistant. The user received an email.\n"
        f"Subject: {subject}\n"
        f"From: {sender}\n"
        f"Email content/snippet: {snippet_or_body}\n\n"
        "Please write a friendly and professional draft reply addressing the sender's query."
    )
    try:
        response = client.chat.completions.create(
            model=settings["DRAFT_MODEL"],
            messages=[
                {"role": "system", "content": settings["DRAFT_SYSTEM_MSG"]},
                {"role": "user", "content": instructions},
            ],
            max_tokens=settings["DRAFT_MAX_TOKENS"],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"Error calling OpenAI API: {exc}")
        fallback_lines = [
            "Hello,",
            "",
            "I'm sorry, but I couldn't generate a response at this time. Please review this email manually.",
            "",
            "Best,",
            "Automated Script",
        ]
        return "\n".join(fallback_lines)


def classify_email(text):
    """Classify an email and return a dict with type, importance, and reasoning."""
    settings = _load_settings()
    client = OpenAI(api_key=require_openai_api_key())
    try:
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
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


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
    """Create a FreeScout conversation using a text-only thread payload.

    Threads are sent using text-only payloads (text), not body.
    """
    settings = _load_settings()
    if settings["TICKET_SYSTEM"] != "freescout":
        return None

    url, key = require_ticket_settings()
    mailbox_id = settings.get("FREESCOUT_MAILBOX_ID")
    if not mailbox_id:
        raise RuntimeError("ticket.mailbox_id must be configured for FreeScout")

    custom_fields: Dict[str, Any] = {}
    tags: list[str] = []
    gmail_thread_field = settings.get("FREESCOUT_GMAIL_THREAD_FIELD_ID")
    gmail_message_field = settings.get("FREESCOUT_GMAIL_MESSAGE_FIELD_ID")

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
        "mailboxId": mailbox_id,
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
