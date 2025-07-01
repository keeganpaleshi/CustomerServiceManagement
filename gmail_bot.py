import os

# additional imports
import json
import requests

# — model choices —
CLASSIFY_MODEL  = "gpt-4.1"
DRAFT_MODEL     = "o3"

# — routing & thresholds —
TICKET_SYSTEM   = "freescout"
FREESCOUT_URL   = os.getenv("FREESCOUT_URL", "")
FREESCOUT_KEY   = os.getenv("FREESCOUT_KEY", "")
CRITIC_THRESHOLD = 8.0
MAX_RETRIES      = 2


def create_ticket(subject: str, sender: str, body: str):
    if TICKET_SYSTEM != "freescout":
        return
    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }
    r = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-FreeScout-API-Key": FREESCOUT_KEY,
        },
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
