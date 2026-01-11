# CustomerServiceManagement

**What it does**

1. Reads all **unread** Gmail messages.
2. Uses **GPT-4o** to classify each as `lead`, `customer`, or `other` and assign an `importance` score (1-10).
3. Ignores `other` and automatically skips promotional or newsletter emails
   based on message headers/body content (e.g., List-Unsubscribe/List-ID or
   unsubscribe text).
4. Routes each message:
   * High-importance emails open a ticket in **FreeScout** with retry logic.
   * Lower-importance emails can receive a draft reply suggestion as an internal
     note in FreeScout (no Gmail drafts).
5. Prints a one-line log per processed email.
6. Polls FreeScout for recent ticket updates after processing.

---

## Processing gates and terminal states

The database is the **only** skip gate for Gmail message processing and is the
source of truth. If a message is not recorded in the DB, it will be evaluated
regardless of Gmail labels or other metadata.

Gmail labels are **cosmetic only** and are never used for control flow. Label
changes do not prevent reprocessing or bypass any step.

Filtered messages are recorded in the DB as a **terminal state** (`filtered`)
and are not reprocessed. Successfully ticketed/appended messages are stored as
`success`. If FreeScout is unavailable or ticket creation fails, the run is
marked `failed` (retryable) and reruns will continue processing the same
message until it reaches a terminal DB state.

---

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-…"
export FREESCOUT_URL="https://desk.example.com"   # or set in config.yaml
export FREESCOUT_KEY="your-freescout-api-key"      # or set in config.yaml
python gmail_bot.py

OAuth – the first run opens a browser window; token is cached in token.json.
Pass `--console-auth` or set `gmail.use_console_oauth: true` in `config.yaml`
to use the copy/paste console flow when no browser is available.
```
### Development

Run `flake8` before committing:
```bash
flake8 Draft_Replies.py gmail_bot.py utils.py
```


### Tweaking behavior

Edit `config.yaml` to adjust limits and model settings. Key options include:

#### Core settings (required)

- `gmail.query` – default Gmail search query.
- `ticket.mailbox_id` – FreeScout mailbox ID used when creating conversations.
- `http.timeout` – HTTP request timeout in seconds.
- `limits.max_drafts` – maximum number of drafts created per run.
- `openai.classify_max_tokens` – token limit for the classification step.

#### AI drafting settings (optional)

- `openai.draft_system_message` – system prompt used for draft replies.
- `openai.draft_max_tokens` – token limit for reply generation.
- `actions.post_suggested_reply` – add an internal note containing a suggested AI reply.

#### Follow-up drafting settings (optional)

- `ticket.followup.*` – controls follow-up draft behavior and scheduling.

## Working fully inside FreeScout

Reps can now triage and respond without Gmail drafts. FreeScout conversations
can be ingested either by polling from `gmail_bot.py` or via a lightweight
webhook handler. The bot uses the FreeScout API to fetch the conversation's
latest customer message, classifies it with OpenAI, and then updates the ticket
in place: priority, assignment, tags, custom fields, and internal notes with an
optional suggested reply.

### Draft reply guidance and deprecation notice

Draft replies now live inside FreeScout as internal notes. The legacy
`Draft_Replies.py` script is deprecated and retained only for historical
reference; new reply suggestions should be generated through the FreeScout
ticket flow in `gmail_bot.py` or the follow-up tooling.

### Configure actions

Set the following keys in `config.yaml` under `ticket:`:

- `webhook_enabled` – set to true to use webhook ingestion. When enabled,
  polling is disabled and webhooks are the primary ingestion path.
- `webhook_secret` – shared secret to validate webhook calls (header
  `X-Webhook-Secret`).
- `poll_interval` – how often to poll FreeScout when running with
  `--poll-freescout`.
- `actions.update_priority` – set FreeScout priority using the AI importance
  score; `actions.priority_high_threshold` controls the urgency cutoff.
- `actions.assign_to_user_id` – FreeScout user ID to auto-assign (optional).
- `actions.apply_tags` – tag conversations with the detected type and a
  `high-priority` tag when applicable.
- `mailbox_id` – FreeScout mailbox ID used when creating conversations (required
  for ticket creation).
- `gmail_thread_field_id` / `gmail_message_field_id` – custom field IDs to store
  the Gmail `threadId`/`messageId` on the ticket for traceability (optional).
- `actions.custom_fields.type_field_id` / `importance_field_id` – custom field
  IDs to store the classification and importance (optional).
- `actions.post_internal_notes` – add an internal note summarizing the
  classification.
- `actions.post_suggested_reply` – add an internal note containing a suggested
  AI reply so agents can reply directly from FreeScout.

### Scheduled polling

```bash
python gmail_bot.py --poll-freescout --poll-interval 300
```

The bot fetches recent conversations from FreeScout, classifies the newest
customer thread, updates ticket metadata, and posts notes/suggestions based on
the configured toggles.

Polling is intended as a fallback when webhook ingestion is disabled (set
`ticket.webhook_enabled` to `false`).

### Follow-up drafts for stale conversations

Use `freescout_followups.py` to find conversations with no agent reply after a
minimum number of hours, draft a follow-up reply (as an internal note), and tag
them with `followup-ready`.

```bash
python freescout_followups.py --hours 24 --required-tags needs-followup
```

Configuration lives under `ticket.followup` in `config.yaml`:

- `hours_without_reply` – minimum hours since the last customer message.
- `required_tags` / `excluded_tags` – tag filters for qualifying conversations.
- `required_states` – optional FreeScout status/state filters.
- `followup_tag` – tag applied after drafting the follow-up.
- `list_params` – raw query params sent to `/api/conversations` (optional).
- `p0_tags` / `notify` – optional Slack/email notifications for P0 follow-ups.

### Webhook handler

Add this minimal FastAPI or Flask endpoint to receive FreeScout webhooks and
let `gmail_bot` do the rest:

```python
from fastapi import FastAPI, Header, Request
from gmail_bot import freescout_webhook_handler

app = FastAPI()

@app.post("/freescout")
async def freescout(payload: Request, x_webhook_secret: str | None = Header(None)):
    body = await payload.json()
    message, status, _ = freescout_webhook_handler(
        body, {"X-Webhook-Secret": x_webhook_secret}
    )
    return message, status
```

Deploy the webhook with HTTPS and set the URL inside FreeScout's webhook
settings. The handler validates `X-Webhook-Secret` (when configured) and then
fetches the conversation from FreeScout before applying the same classification
and update flow used by the poller.

Webhook ingestion takes precedence over polling, so disable
`ticket.webhook_enabled` if you need to run the poller instead.
