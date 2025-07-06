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
