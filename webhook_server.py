from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from gmail_bot import freescout_webhook_handler
from utils import log_event

app = FastAPI()
LOG_DIR = Path(__file__).resolve().parent / "logs" / "webhooks"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
COUNTERS = {
    "processed": 0,
    "created": 0,
    "appended": 0,
    "drafted": 0,
    "filtered": 0,
    "failed": 0,
}


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

    message, status = freescout_webhook_handler(body, {"X-Webhook-Secret": x_webhook_secret})
    if status >= 400:
        COUNTERS["failed"] += 1
        log_event(
            "webhook_ingest",
            action="handle_payload",
            outcome="failed",
            conversation_id=conversation_id,
            reason=message,
        )
    else:
        COUNTERS["processed"] += 1
        log_event(
            "webhook_ingest",
            action="handle_payload",
            outcome="success",
            conversation_id=conversation_id,
        )
    return JSONResponse({"message": message}, status_code=status)


@app.on_event("shutdown")
def log_summary() -> None:
    log_event(
        "webhook_ingest_summary",
        processed=COUNTERS["processed"],
        created=COUNTERS["created"],
        appended=COUNTERS["appended"],
        drafted=COUNTERS["drafted"],
        filtered=COUNTERS["filtered"],
        failed=COUNTERS["failed"],
    )
