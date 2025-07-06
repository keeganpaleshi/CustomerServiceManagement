# CustomerServiceManagement

**What it does**

1. Reads all **unread** Gmail messages.
2. Uses **GPT-4.1** to classify each as `lead`, `customer`, or `other` and assign an `importance` score (1-10).
3. Ignores `other` and automatically skips promotional or newsletter emails
   based on Gmail labels or unsubscribe headers.
4. For leads & customers:
   * Drafts a reply with **o3** (never sends).
   * Runs a self-critique loop until the draft scores ≥ 8.
   * Saves the draft to Gmail.
   * Opens a ticket in FreeScout.
5. Prints a one-line log per processed email.

---

## Quick start

```bash
pip install google-api-python-client google-auth-oauthlib openai requests
export OPENAI_API_KEY="sk-…"
export FREESCOUT_URL="https://desk.example.com"
export FREESCOUT_KEY="your-freescout-api-key"
python gmail_bot.py

OAuth – the first run opens a browser window; token is cached in token.pickle.
```

### Tweaking behavior

Edit `config.yaml` to adjust limits and model settings. Key options include:

- `openai.draft_system_message` – system prompt used for draft replies.
- `openai.draft_max_tokens` – token limit for reply generation.
- `openai.classify_max_tokens` – token limit for the classification step.
- `limits.max_drafts` – maximum number of drafts created per run.

## Syncing FreeScout updates to Gmail

The script can keep Gmail informed of new activity in FreeScout. By default it
polls the FreeScout API every 5 minutes (see `ticket.freescout_poll_interval` in
`config.yaml`). Each poll sends a summary email listing conversations updated
since the last check.

If you prefer real-time updates, set up a webhook receiver using Flask:

```python
from flask import Flask, request
from gmail_bot import get_gmail_service, send_summary_email

app = Flask(__name__)

@app.route('/freescout', methods=['POST'])
def freescout_webhook():
    service = get_gmail_service()
    payload = request.json
    send_summary_email(service, [payload])
    return '', 204

if __name__ == '__main__':
    app.run(port=5000)
```

Configure FreeScout to POST conversation events to your `/freescout` endpoint.
