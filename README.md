# CustomerServiceManagement

**What it does**

1. Reads all **unread** Gmail messages.
2. Uses **GPT-4.1** to classify each as `lead`, `customer`, or `other` and assign an `importance` score (1-10).
3. Ignores `other` and automatically skips promotional or newsletter emails
   based on Gmail labels or unsubscribe headers.
4. Routes each message:
   * High-importance emails open a ticket in **FreeScout** with retry logic.
   * Lower-importance emails get a draft asking for more details.
   * Drafts are self-critiqued until scoring ≥ 8 and then saved to Gmail.
5. Prints a one-line log per processed email.
6. Polls FreeScout for recent ticket updates after processing.

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
- `gmail.query` – default Gmail search query.
- `http.timeout` – HTTP request timeout in seconds.


## Development

Run `flake8` before committing to check for style issues:

```bash
flake8
```

## FreeScout status updates

Run the bot with `--poll-freescout` to periodically check FreeScout for recent
conversation updates. A short summary email is sent to your own inbox after each
poll. Use `--poll-interval` to configure how often (in seconds) the endpoint is
queried.

### Webhook alternative

Instead of polling, you can point FreeScout's webhook integration at a small
Flask app which calls `send_update_email` from `gmail_bot.py`.

```python
from flask import Flask, request
from gmail_bot import get_gmail_service, send_update_email

app = Flask(__name__)
svc = get_gmail_service()

@app.post("/freescout")
def freescout_hook():
    payload = request.get_json()
    send_update_email(svc, str(payload))
    return "", 204
```

Deploy the webhook on a server with HTTPS and set the URL in FreeScout's
settings to have Gmail updated whenever a conversation changes.

