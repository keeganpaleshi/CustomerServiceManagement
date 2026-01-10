from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from gmail_bot import freescout_webhook_handler

app = FastAPI()
LOG_DIR = Path(__file__).resolve().parent / "logs" / "webhooks"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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

    log_webhook_payload(body)

    message, status = freescout_webhook_handler(body, {"X-Webhook-Secret": x_webhook_secret})
    return JSONResponse({"message": message}, status_code=status)
