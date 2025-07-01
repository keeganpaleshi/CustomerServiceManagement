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
