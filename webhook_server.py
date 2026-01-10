from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from gmail_bot import freescout_webhook_handler
from storage import TicketStore
from utils import get_settings, log_event

LOG_DIR = Path(__file__).resolve().parent / "logs" / "webhooks"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
COUNTER_KEYS = ("processed", "created", "appended", "drafted", "filtered", "failed")


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


def _safe_filename(value: str) -> str:
    cleaned = SAFE_FILENAME_RE.sub("-", value.strip())
    return cleaned or "unknown"


def _select_event_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("event_id", "conversation_id", "id"):
            if payload.get(key):
                return str(payload[key])
    return "unknown"


def log_webhook_payload(payload: Any) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    event_id = _safe_filename(_select_event_id(payload))
    logfile = LOG_DIR / f"{timestamp}-{event_id}.json"
    with logfile.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return logfile


def _get_counter_store() -> TicketStore:
    settings = get_settings()
    sqlite_path = settings.get("TICKET_SQLITE_PATH") or "./csm.sqlite"
    return TicketStore(sqlite_path)


@app.post("/freescout")
async def freescout(payload: Request, x_webhook_secret: str | None = Header(None)):
    raw_body = await payload.body()
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        body = {"raw": raw_body.decode("utf-8", errors="replace")}

    logfile = log_webhook_payload(body)
    conversation_id = (
        body.get("conversation_id") or body.get("id") if isinstance(body, dict) else None
    )
    log_event(
        "webhook_ingest",
        action="log_payload",
        outcome="success",
        conversation_id=conversation_id,
        logfile=str(logfile),
    )

    message, status, outcome = freescout_webhook_handler(
        body, {"X-Webhook-Secret": x_webhook_secret}
    )
    counter_store = _get_counter_store()
    try:
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
    finally:
        counter_store.close()
    return JSONResponse({"message": message}, status_code=status)

