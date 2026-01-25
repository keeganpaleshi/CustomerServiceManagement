from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse

from gmail_bot import freescout_webhook_handler
from storage import TicketStore
from utils import get_settings, log_event, reload_settings

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs" / "webhooks"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
COUNTER_KEYS = ("processed", "created", "appended", "drafted", "filtered", "failed")
# Fields that may contain sensitive data and should be redacted in logs
SENSITIVE_FIELDS = {"email", "customer_email", "customerEmail", "body", "text", "content", "password", "secret", "token", "api_key"}
# Pre-computed lowercase version for efficient case-insensitive comparison
SENSITIVE_FIELDS_LOWER = {f.lower() for f in SENSITIVE_FIELDS}

# Maximum allowed webhook payload size (1MB)
MAX_PAYLOAD_SIZE = 1 * 1024 * 1024  # 1MB
# Maximum size for logged payloads (100KB)
MAX_LOG_SIZE = 100 * 1024  # 100KB


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    counter_store = _get_counter_store()
    try:
        counters = counter_store.get_webhook_counters()
    finally:
        counter_store.close()
    log_event(
        "webhook_ingest_summary",
        processed=counters.get("processed", 0),
        created=counters.get("created", 0),
        appended=counters.get("appended", 0),
        drafted=counters.get("drafted", 0),
        filtered=counters.get("filtered", 0),
        failed=counters.get("failed", 0),
    )


app = FastAPI(lifespan=lifespan)


def _get_log_dir() -> Path:
    """Get log directory from settings or use default."""
    settings = get_settings()
    log_dir = settings.get("WEBHOOK_LOG_DIR")
    if log_dir:
        return Path(log_dir)
    return _DEFAULT_LOG_DIR


def _safe_filename(value: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("-", value.strip())
    return cleaned or "unknown"


def _sanitize_payload(payload: Any) -> Any:
    """Redact sensitive fields from payload for logging."""
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            if key.lower() in SENSITIVE_FIELDS_LOWER:
                if isinstance(value, str) and len(value) > 0:
                    sanitized[key] = f"[REDACTED:{len(value)} chars]"
                else:
                    sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _sanitize_payload(value)
        return sanitized
    elif isinstance(payload, list):
        return [_sanitize_payload(item) for item in payload]
    return payload


def _select_event_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("event_id", "conversation_id", "id"):
            if payload.get(key):
                return str(payload[key])
    return "unknown"


def log_webhook_payload(payload: Any) -> Path:
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    event_id = _safe_filename(_select_event_id(payload))
    logfile = log_dir / f"{timestamp}-{event_id}.json"

    # Sanitize sensitive data before logging
    sanitized_payload = _sanitize_payload(payload)

    # Serialize and check size before writing
    serialized = json.dumps(sanitized_payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(serialized) > MAX_LOG_SIZE:
        # Truncate large payloads - use the already-serialized JSON for partial data
        # to preserve valid JSON format in the truncated portion
        partial_json = serialized[:MAX_LOG_SIZE // 2]
        truncated_payload = {
            "_truncated": True,
            "_original_size": len(serialized),
            "_max_size": MAX_LOG_SIZE,
            "partial_data": partial_json
        }
        serialized = json.dumps(truncated_payload, ensure_ascii=False, indent=2)

    with logfile.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
    return logfile


def _get_counter_store() -> TicketStore:
    reload_settings()
    settings = get_settings()
    sqlite_path = settings.get("TICKET_SQLITE_PATH") or "./csm.sqlite"
    return TicketStore(sqlite_path)


@app.get("/health")
async def health():
    """Health check endpoint for container orchestration."""
    return {"status": "ok"}


@app.post("/freescout")
async def freescout(request: Request, x_webhook_secret: Optional[str] = Header(None)):
    # Check content-length header first to reject oversized payloads quickly
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            content_length_int = int(content_length)
        except (ValueError, TypeError):
            log_event(
                "webhook_ingest",
                action="reject_payload",
                outcome="failed",
                reason="invalid_content_length",
                content_length=content_length,
            )
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")

        if content_length_int > MAX_PAYLOAD_SIZE:
            log_event(
                "webhook_ingest",
                action="reject_payload",
                outcome="failed",
                reason="payload_too_large",
                size=content_length_int,
                max_size=MAX_PAYLOAD_SIZE,
            )
            raise HTTPException(status_code=413, detail="Payload too large")

    raw_body = await request.body()

    # Validate actual body size
    if len(raw_body) > MAX_PAYLOAD_SIZE:
        log_event(
            "webhook_ingest",
            action="reject_payload",
            outcome="failed",
            reason="payload_too_large",
            size=len(raw_body),
            max_size=MAX_PAYLOAD_SIZE,
        )
        raise HTTPException(status_code=413, detail="Payload too large")

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        # Decode raw body but limit length to prevent logging excessive data
        # and redact common sensitive patterns
        decoded = raw_body.decode("utf-8", errors="replace")
        if len(decoded) > MAX_LOG_SIZE:
            decoded = decoded[:MAX_LOG_SIZE] + "...[truncated]"
        body = {"raw": "[NON-JSON PAYLOAD - content not logged for security]", "raw_length": len(raw_body)}

    logfile = log_webhook_payload(body)
    if isinstance(body, dict):
        conversation_id = body.get("conversation_id") or body.get("id")
    else:
        conversation_id = None
    log_event(
        "webhook_ingest",
        action="log_payload",
        outcome="success",
        conversation_id=conversation_id,
        logfile=str(logfile),
    )

    # Create a single TicketStore instance and reuse it for all counter operations
    counter_store = _get_counter_store()
    try:
        if not isinstance(body, dict):
            message = "Expected JSON object payload for FreeScout webhook."
            counter_store.increment_webhook_counter("failed")
            log_event(
                "webhook_ingest",
                action="handle_payload",
                outcome="failed",
                conversation_id=conversation_id,
                reason=message,
            )
            return JSONResponse({"message": message}, status_code=400)

        message, status, outcome = freescout_webhook_handler(
            body, {"X-Webhook-Secret": x_webhook_secret}
        )
        if status >= 400:
            counter_store.increment_webhook_counter("failed")
            log_event(
                "webhook_ingest",
                action="handle_payload",
                outcome="failed",
                conversation_id=conversation_id,
                reason=message,
            )
        else:
            counter_store.increment_webhook_counter("processed")
            if outcome:
                action = outcome.action
                if action in COUNTER_KEYS:
                    counter_store.increment_webhook_counter(action)
                if outcome.drafted:
                    counter_store.increment_webhook_counter("drafted")
            log_event(
                "webhook_ingest",
                action="handle_payload",
                outcome="success",
                conversation_id=conversation_id,
            )
        return JSONResponse({"message": message}, status_code=status)
    finally:
        counter_store.close()
