# CustomerServiceManagement

**What it does**

1. Reads all **unread** Gmail messages.
2. Uses **GPT-4.1** to classify each as `lead`, `customer`, or `other` and assign an `importance` score (1-10).
3. Ignores `other`.
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
