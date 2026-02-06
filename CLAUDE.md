# CLAUDE.md — AI Assistant Guide

## Project Overview

Customer Service Management (CSM) is a Python 3.11+ application that routes emails between Gmail and FreeScout (help desk) using OpenAI for classification and reply drafting. It runs as Docker containers with four service modes: Gmail ingestion, FreeScout polling, webhook server, and follow-up drafting.

## Repository Structure

```
├── gmail_bot.py          # Gmail ingestion pipeline (fetch, classify, create/append tickets)
├── utils.py              # Shared utilities: FreeScoutClient, rate limiter, cost tracker, logging
├── storage.py            # SQLite-backed TicketStore (processed messages, thread maps, drafts)
├── webhook_server.py     # FastAPI webhook receiver with HMAC/nonce security
├── freescout_followups.py # Generate follow-up drafts for stale conversations
├── config.yaml           # All runtime configuration (models, prompts, limits, credentials)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Python 3.11-slim, non-root user, health check on :8000
├── docker-compose.yml    # Four service profiles: gmail-bot, freescout-poller, webhook-server, followups
├── tests/                # unittest-based test suite
│   ├── test_gmail_ingestion.py
│   ├── test_utils.py
│   ├── test_webhook_validation.py
│   └── test_storage.py
├── .env.example          # Required/optional environment variables
├── .flake8               # max-line-length = 180
└── README.md             # User-facing documentation
```

## Development Commands

### Install dependencies
```bash
pip install -r requirements.txt
pip install pytest   # for testing (not in requirements.txt)
```

### Linting (run before committing)
```bash
flake8 gmail_bot.py utils.py storage.py webhook_server.py freescout_followups.py
```

### Tests
```bash
python -m pytest tests/ -v
```

**Note:** 2 tests are currently failing (pre-existing): `test_estimate_cost_known_model` and `test_truncates_deeply_nested_payload`. All other 71 tests pass.

### Docker
```bash
docker compose up gmail-bot                          # one-shot Gmail ingestion
docker compose --profile webhook up webhook-server   # webhook on port 8000
docker compose --profile polling up freescout-poller # continuous polling
docker compose --profile followups up followups      # follow-up draft generation
```

## Architecture & Key Patterns

### Data flow
Gmail messages → `gmail_bot.py` (fetch & extract) → OpenAI (classify) → `FreeScoutClient` (create/append ticket) → OpenAI (draft reply) → FreeScout (internal note/draft)

### State management
SQLite is the single source of truth. `TicketStore` tracks processed messages, Gmail-to-FreeScout thread mappings, bot draft hashes (deduplication), webhook counters, and nonces. Uses WAL mode for concurrent access.

### Key classes and entry points
- `FreeScoutClient` (`utils.py:891`) — REST wrapper for all FreeScout API calls with retry logic
- `TicketStore` (`storage.py:10`) — SQLite persistence layer
- `OpenAICostTracker` (`utils.py:599`) — Budget enforcement (daily/monthly limits)
- `SimpleRateLimiter` (`utils.py:573`) — Token bucket rate limiting
- `process_gmail_message()` (`gmail_bot.py`) — Core per-message processing pipeline
- `app` (`webhook_server.py`) — FastAPI application with security middleware

### Configuration
All runtime config lives in `config.yaml`. Secrets come from environment variables (`OPENAI_API_KEY`, `FREESCOUT_URL`, `FREESCOUT_KEY`). Settings are loaded via `get_settings()` / `reload_settings()` in `utils.py`.

### Security conventions
- Input validation: `validate_conversation_id()`, `normalize_id()` for all external IDs
- Log sanitization: `_sanitize_log_value()` redacts sensitive fields, `_escape_control_chars()` prevents log injection
- Webhook security: HMAC signature verification, nonce-based replay prevention, timestamp skew validation
- Token encryption: Optional Fernet encryption for OAuth tokens via `GMAIL_TOKEN_ENCRYPTION_KEY`
- Security headers middleware: HSTS, CSP, X-Frame-Options on webhook server

## Code Conventions

- **Type hints** on all function signatures (use `Optional`, `Dict`, `List`, `Tuple`)
- **Frozen dataclasses** for result types (`ProcessResult`, `WebhookOutcome`, `_MessageContent`)
- **Private functions** prefixed with `_` (e.g., `_get_openai_client()`)
- **Naming**: `CamelCase` classes, `snake_case` functions/variables, `UPPER_CASE` constants
- **Structured logging** via `log_event()` and `log_error()` — no bare `print()` calls
- **Max line length**: 180 characters (per `.flake8`)
- **Import order**: stdlib → third-party → local
- **Error handling**: retry with exponential backoff for external APIs; graceful degradation with fallback values

## Important Constraints

- AI drafts are **never auto-sent** — always stored as internal notes or agent drafts for human review
- Database-first design: a message is only processed once (deduplicated by `gmail_message_id`)
- Budget limits are enforced — check `openai.budget` in `config.yaml` before changing models
- The `.gitignore` excludes `token.json`, `client_secret.json`, `.env`, and `*.sqlite` — never commit secrets or databases
