from __future__ import annotations

import json
import random
import re
import hmac
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from gmail_bot import freescout_webhook_handler
from storage import TicketStore
from utils import get_settings, log_event, reload_settings, validate_conversation_id


import os

# HSTS configuration from environment variable
# Set ENABLE_HSTS=1 or ENABLE_HSTS=true to enable Strict-Transport-Security header
ENABLE_HSTS = os.getenv("ENABLE_HSTS", "").lower() in ("1", "true", "yes")
HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))  # Default: 1 year


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Enforce HTTPS (configurable via ENABLE_HSTS environment variable)
        # Enable when running behind HTTPS proxy/load balancer
        if ENABLE_HSTS:
            response.headers["Strict-Transport-Security"] = f"max-age={HSTS_MAX_AGE}; includeSubDomains"

        # Content Security Policy - very restrictive for API
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'none'; "
            "form-action 'none'"
        )

        # Referrer policy
        response.headers["Referrer-Policy"] = "no-referrer"

        # Permissions policy (disable all features)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )

        return response

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs" / "webhooks"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
SAFE_FILENAME_MAX_LENGTH = 128
COUNTER_KEYS = ("processed", "created", "appended", "drafted", "filtered", "failed")
# Fields that may contain sensitive data and should be redacted in logs
SENSITIVE_FIELDS = {"email", "customer_email", "customerEmail", "body", "text", "content", "password", "secret", "token", "api_key"}
# Pre-computed lowercase version for efficient case-insensitive comparison
SENSITIVE_FIELDS_LOWER = {f.lower() for f in SENSITIVE_FIELDS}


# Maximum allowed webhook payload size (1MB)
MAX_PAYLOAD_SIZE = 1 * 1024 * 1024  # 1MB
# Maximum size for logged payloads (100KB)
MAX_LOG_SIZE = 100 * 1024  # 100KB
# Maximum size for error log truncation (10KB) - smaller for error messages
MAX_ERROR_LOG_SIZE = 10 * 1024  # 10KB

# Webhook replay protection settings - defaults used when config is unavailable
# These can be overridden in config.yaml under webhook.security
_DEFAULT_MAX_TIMESTAMP_SKEW_SECONDS = 300  # 5 minutes
_DEFAULT_NONCE_CACHE_SIZE = 10000  # Maximum nonces to track
_DEFAULT_NONCE_CACHE_TTL_SECONDS = 600  # 10 minutes


def _get_webhook_security_settings() -> tuple[int, int, int]:
    """Get webhook security settings from config with defaults.

    Returns:
        Tuple of (max_timestamp_skew_seconds, nonce_cache_size, nonce_cache_ttl_seconds)
    """
    try:
        settings = get_settings()
        return (
            settings.get("WEBHOOK_MAX_TIMESTAMP_SKEW_SECONDS", _DEFAULT_MAX_TIMESTAMP_SKEW_SECONDS),
            settings.get("WEBHOOK_NONCE_CACHE_SIZE", _DEFAULT_NONCE_CACHE_SIZE),
            settings.get("WEBHOOK_NONCE_CACHE_TTL_SECONDS", _DEFAULT_NONCE_CACHE_TTL_SECONDS),
        )
    except Exception:
        # Fall back to defaults if config loading fails
        return (
            _DEFAULT_MAX_TIMESTAMP_SKEW_SECONDS,
            _DEFAULT_NONCE_CACHE_SIZE,
            _DEFAULT_NONCE_CACHE_TTL_SECONDS,
        )


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

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)


def _get_log_dir() -> Path:
    """Get log directory from settings or use default."""
    settings = get_settings()
    log_dir = settings.get("WEBHOOK_LOG_DIR")
    if log_dir:
        return Path(log_dir)
    return _DEFAULT_LOG_DIR


def _safe_filename(value: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("-", value.strip())
    cleaned = cleaned[:SAFE_FILENAME_MAX_LENGTH]
    return cleaned or "unknown"


def _sanitize_payload(payload: Any, max_depth: int = 10) -> Any:
    """Redact sensitive fields from payload for logging."""
    if not isinstance(payload, (dict, list)):
        return payload

    if max_depth < 0:
        return "[TRUNCATED]"

    if isinstance(payload, dict):
        sanitized_root: Any = {}
    else:
        sanitized_root = []

    stack: list[tuple[Any, int, Any]] = [(payload, 0, sanitized_root)]

    while stack:
        current, depth, sanitized = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key.lower() in SENSITIVE_FIELDS_LOWER:
                    if isinstance(value, str) and len(value) > 0:
                        sanitized[key] = f"[REDACTED:{len(value)} chars]"
                    else:
                        sanitized[key] = "[REDACTED]"
                    continue

                if isinstance(value, (dict, list)):
                    if depth + 1 >= max_depth:
                        sanitized[key] = "[TRUNCATED]"
                        continue
                    next_container: Any = {} if isinstance(value, dict) else []
                    sanitized[key] = next_container
                    stack.append((value, depth + 1, next_container))
                else:
                    sanitized[key] = value
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    if depth + 1 >= max_depth:
                        sanitized.append("[TRUNCATED]")
                        continue
                    next_container = {} if isinstance(item, dict) else []
                    sanitized.append(next_container)
                    stack.append((item, depth + 1, next_container))
                else:
                    sanitized.append(item)

    return sanitized_root


def _select_event_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("event_id", "conversation_id", "id"):
            if payload.get(key):
                return str(payload[key])
    return "unknown"


def _validate_webhook_timestamp(timestamp: Any) -> tuple[bool, str]:
    """Validate webhook timestamp to prevent replay attacks.

    Args:
        timestamp: Timestamp value from webhook payload (ISO format or Unix timestamp)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if timestamp is None:
        return True, ""  # Timestamp is optional

    max_timestamp_skew, _, _ = _get_webhook_security_settings()

    try:
        # Try parsing as ISO format
        if isinstance(timestamp, str):
            if timestamp.endswith("Z"):
                timestamp = timestamp.replace("Z", "+00:00")
            webhook_time = datetime.fromisoformat(timestamp)
            if webhook_time.tzinfo is None:
                webhook_time = webhook_time.replace(tzinfo=timezone.utc)
        # Try parsing as Unix timestamp
        elif isinstance(timestamp, (int, float)):
            webhook_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            return False, "timestamp must be ISO string or Unix timestamp"

        # Check if timestamp is within acceptable range
        now = datetime.now(timezone.utc)
        time_diff = abs((now - webhook_time).total_seconds())

        if time_diff > max_timestamp_skew:
            return False, f"timestamp too old or too far in future (max skew: {max_timestamp_skew}s)"

        return True, ""
    except (ValueError, TypeError, OverflowError) as e:
        return False, f"invalid timestamp format: {e}"


def _check_and_record_nonce(nonce: Any) -> tuple[bool, str]:
    """Check if nonce has been seen before and record it.

    This function implements replay protection by tracking nonces. It enforces
    memory limits to prevent unbounded growth under sustained attack conditions.

    Args:
        nonce: Nonce value from webhook payload

    Returns:
        Tuple of (is_new, error_message)
    """
    if nonce is None:
        return True, ""  # Nonce is optional

    try:
        nonce_str = str(nonce)
    except (TypeError, ValueError):
        return False, "nonce must be convertible to string"

    if len(nonce_str) > 256:
        return False, "nonce exceeds maximum length (256)"

    _, nonce_cache_size, nonce_cache_ttl = _get_webhook_security_settings()

    nonce_store = _get_counter_store()
    try:
        is_new = nonce_store.check_and_record_webhook_nonce(
            nonce_str,
            nonce_cache_ttl,
            nonce_cache_size,
        )
    finally:
        nonce_store.close()

    if not is_new:
        return False, "nonce has been used before (replay attack detected)"

    return True, ""


def _cleanup_old_webhook_logs(log_dir: Path, max_age_days: int = 30, max_files: int = 10000) -> None:
    """Remove old webhook logs to prevent disk space exhaustion.

    Args:
        log_dir: Directory containing webhook logs
        max_age_days: Remove files older than this many days
        max_files: Maximum number of log files to keep (oldest removed first)
    """
    try:
        if not log_dir.exists():
            return

        # Get all JSON log files
        log_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove files older than max_age_days
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        removed_count = 0

        for log_file in log_files:
            try:
                # Check age
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff_time:
                    log_file.unlink()
                    removed_count += 1
            except (OSError, ValueError) as e:
                log_event(
                    "webhook_cleanup",
                    action="remove_old_log",
                    outcome="failed",
                    file=str(log_file),
                    reason=str(e),
                )

        # If still too many files, remove oldest
        log_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(log_files) > max_files:
            for log_file in log_files[max_files:]:
                try:
                    log_file.unlink()
                    removed_count += 1
                except OSError as e:
                    log_event(
                        "webhook_cleanup",
                        action="remove_excess_log",
                        outcome="failed",
                        file=str(log_file),
                        reason=str(e),
                    )

        if removed_count > 0:
            log_event(
                "webhook_cleanup",
                action="cleanup_logs",
                outcome="success",
                removed_count=removed_count,
            )
    except Exception as e:
        log_event(
            "webhook_cleanup",
            action="cleanup_logs",
            outcome="failed",
            reason=str(e),
        )


def log_webhook_payload(payload: Any) -> Path:
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Periodically clean up old logs (1% chance per request to avoid overhead)
    if random.random() < 0.01:
        settings = get_settings()
        max_age_days = settings.get("WEBHOOK_LOG_MAX_AGE_DAYS", 30)
        max_files = settings.get("WEBHOOK_LOG_MAX_FILES", 10000)
        _cleanup_old_webhook_logs(log_dir, max_age_days, max_files)

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

    reload_settings()
    settings = get_settings()
    secret = settings.get("FREESCOUT_WEBHOOK_SECRET", "")
    if secret:
        provided_secret = x_webhook_secret or ""
        if not hmac.compare_digest(provided_secret, secret):
            client_host = request.client.host if request.client else "unknown"
            log_event(
                "webhook_ingest",
                action="validate_secret",
                outcome="failed",
                reason="invalid_signature",
                client_ip=client_host,
            )
            raise HTTPException(status_code=401, detail="invalid signature")

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
        # Use smaller error log size for non-JSON payloads (likely malformed)
        decoded = raw_body.decode("utf-8", errors="replace")
        if len(decoded) > MAX_ERROR_LOG_SIZE:
            decoded = decoded[:MAX_ERROR_LOG_SIZE] + "...[truncated]"
        body = {"raw": "[NON-JSON PAYLOAD - content not logged for security]", "raw_length": len(raw_body)}

    logfile = log_webhook_payload(body)

    # Extract and validate conversation ID
    conversation_id = None
    if isinstance(body, dict):
        # Validate timestamp for replay protection
        timestamp = body.get("timestamp") or body.get("created_at") or body.get("updated_at")
        is_valid_ts, ts_error = _validate_webhook_timestamp(timestamp)
        if not is_valid_ts:
            log_event(
                "webhook_ingest",
                action="validate_timestamp",
                outcome="failed",
                reason=ts_error,
                logfile=str(logfile),
            )
            raise HTTPException(status_code=400, detail=f"Invalid timestamp: {ts_error}")

        # Check nonce for replay protection
        nonce = body.get("nonce") or body.get("event_id") or body.get("id")
        is_new_nonce, nonce_error = _check_and_record_nonce(nonce)
        if not is_new_nonce:
            log_event(
                "webhook_ingest",
                action="check_nonce",
                outcome="failed",
                reason=nonce_error,
                logfile=str(logfile),
            )
            raise HTTPException(status_code=400, detail=f"Replay attack detected: {nonce_error}")

        raw_id = body.get("conversation_id") or body.get("id")
        is_valid, validated_id, validation_error = validate_conversation_id(raw_id)
        if is_valid:
            conversation_id = validated_id
        elif raw_id is not None:
            # Log validation failure but continue processing
            log_event(
                "webhook_ingest",
                action="validate_conversation_id",
                outcome="failed",
                reason=validation_error,
                logfile=str(logfile),
            )

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
            raise HTTPException(status_code=400, detail=message)

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
