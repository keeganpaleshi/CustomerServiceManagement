# Codex Instructions: Polish & Production Readiness

## Overview

All core functionality is complete! These tasks focus on **documentation, deployment, and operational readiness**. Each task is a standalone PR.

---

## PR #1: Update README.md to Reflect Current Implementation

**Priority:** HIGH
**Estimated scope:** Small (documentation only)

### Problem
README.md contains outdated information about Gmail draft workflows that were removed in Phase 3.

### Tasks

1. **Update "What it does" section (lines 3-16)**
   - Remove mention of "self-critiqued drafts saved to Gmail" (lines 12-14)
   - Update to reflect FreeScout draft reply workflow
   - Describe agent draft creation with hash-based edit detection
   - Mention webhook/polling ingestion options

2. **Remove/update Draft_Replies.py section (lines 77-82)**
   - Either remove this section entirely OR
   - Update to clarify it's deprecated and raises DeprecationWarning
   - Remove statement about "draft-only script"

3. **Add deployment section**
   - Add "Deployment" section after "Quick start"
   - Include instructions for running `webhook_server.py`:
     ```bash
     uvicorn webhook_server:app --host 0.0.0.0 --port 8000
     ```
   - Mention that webhook URL should be `https://your-domain.com/freescout`
   - Reference configuration: `ticket.webhook_enabled` and `ticket.webhook_secret`

4. **Update feature descriptions**
   - Add section explaining bot draft tracking
   - Explain hash-based edit detection (prevents overwriting human edits)
   - Clarify that draft replies are created in FreeScout, not Gmail
   - Update follow-up section to mention draft creation (not just internal notes)

### Acceptance Criteria
- [ ] README accurately describes current FreeScout-only workflow
- [ ] No mention of Gmail draft creation (except in historical context)
- [ ] Deployment instructions present for webhook server
- [ ] Bot draft tracking feature documented

### Example PR Title
```
docs: Update README to reflect FreeScout draft workflow
```

---

## PR #2: Add Production Deployment Files

**Priority:** HIGH
**Estimated scope:** Medium (multiple files)

### Problem
No standardized way to deploy the system in production.

### Tasks

1. **Create systemd service file: `systemd/csm-gmail-bot.service`**
   ```ini
   [Unit]
   Description=CSM Gmail Bot (ingestion)
   After=network.target

   [Service]
   Type=simple
   User=csm
   WorkingDirectory=/opt/customerservicemanagement
   Environment="OPENAI_API_KEY=your_key_here"
   Environment="FREESCOUT_URL=https://desk.example.com"
   Environment="FREESCOUT_KEY=your_api_key"
   ExecStart=/opt/customerservicemanagement/venv/bin/python gmail_bot.py
   Restart=on-failure
   RestartSec=10
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

2. **Create systemd service file: `systemd/csm-webhook-server.service`**
   ```ini
   [Unit]
   Description=CSM Webhook Server
   After=network.target

   [Service]
   Type=simple
   User=csm
   WorkingDirectory=/opt/customerservicemanagement
   Environment="OPENAI_API_KEY=your_key_here"
   Environment="FREESCOUT_URL=https://desk.example.com"
   Environment="FREESCOUT_KEY=your_api_key"
   ExecStart=/opt/customerservicemanagement/venv/bin/uvicorn webhook_server:app --host 0.0.0.0 --port 8000 --workers 2
   Restart=on-failure
   RestartSec=10
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

3. **Create systemd timer for follow-ups: `systemd/csm-followups.timer` and `systemd/csm-followups.service`**
   ```ini
   # csm-followups.timer
   [Unit]
   Description=Run CSM follow-ups every 6 hours

   [Timer]
   OnBootSec=15min
   OnUnitActiveSec=6h

   [Install]
   WantedBy=timers.target

   # csm-followups.service
   [Unit]
   Description=CSM FreeScout Follow-ups

   [Service]
   Type=oneshot
   User=csm
   WorkingDirectory=/opt/customerservicemanagement
   Environment="OPENAI_API_KEY=your_key_here"
   Environment="FREESCOUT_URL=https://desk.example.com"
   Environment="FREESCOUT_KEY=your_api_key"
   ExecStart=/opt/customerservicemanagement/venv/bin/python freescout_followups.py
   StandardOutput=journal
   StandardError=journal
   ```

4. **Create Docker Compose file: `docker-compose.yml`**
   ```yaml
   version: '3.8'

   services:
     webhook-server:
       build: .
       ports:
         - "8000:8000"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - FREESCOUT_URL=${FREESCOUT_URL}
         - FREESCOUT_KEY=${FREESCOUT_KEY}
       volumes:
         - ./config.yaml:/app/config.yaml:ro
         - ./csm.sqlite:/app/csm.sqlite
         - ./logs:/app/logs
       command: uvicorn webhook_server:app --host 0.0.0.0 --port 8000
       restart: unless-stopped

     gmail-ingestion:
       build: .
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - FREESCOUT_URL=${FREESCOUT_URL}
         - FREESCOUT_KEY=${FREESCOUT_KEY}
       volumes:
         - ./config.yaml:/app/config.yaml:ro
         - ./csm.sqlite:/app/csm.sqlite
         - ./token.pickle:/app/token.pickle
         - ./client_secret.json:/app/client_secret.json:ro
       command: python gmail_bot.py --poll-freescout
       restart: unless-stopped
       depends_on:
         - webhook-server
   ```

5. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY *.py ./
   COPY config.yaml ./

   CMD ["python", "gmail_bot.py"]
   ```

6. **Create deployment guide: `DEPLOYMENT.md`**
   - Document systemd setup (copy files, enable services)
   - Document Docker setup (build, run)
   - Document environment variable requirements
   - Add troubleshooting section

### Acceptance Criteria
- [ ] Systemd service files for all components
- [ ] Docker Compose setup for containerized deployment
- [ ] Deployment documentation complete
- [ ] All files in appropriate directories

### Example PR Title
```
ops: Add systemd and Docker deployment configurations
```

---

## PR #3: Add Health Check Endpoint and Startup Validation

**Priority:** MEDIUM
**Estimated scope:** Small

### Problem
- No health check endpoint for load balancers/monitoring
- No validation that required configuration is present at startup

### Tasks

1. **Add health check endpoint to `webhook_server.py`**
   ```python
   @app.get("/health")
   async def health():
       """Health check endpoint for load balancers."""
       return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

   @app.get("/ready")
   async def ready():
       """Readiness check - validates configuration."""
       from utils import get_settings
       settings = get_settings()

       issues = []
       if not settings.get("FREESCOUT_URL"):
           issues.append("FREESCOUT_URL not configured")
       if not settings.get("FREESCOUT_KEY"):
           issues.append("FREESCOUT_KEY not configured")
       if not settings.get("OPENAI_API_KEY"):
           issues.append("OPENAI_API_KEY not configured")

       if issues:
           return JSONResponse(
               {"status": "not_ready", "issues": issues},
               status_code=503
           )
       return {"status": "ready"}
   ```

2. **Add startup validation to `gmail_bot.py`**
   - Create function `validate_configuration(settings)` that checks:
     - FreeScout URL and key present (if system is "freescout")
     - OpenAI API key present
     - mailbox_id set (if creating conversations)
     - SQLite path writable
   - Call at start of `main()` before processing
   - Raise clear error if validation fails

3. **Add startup validation to `freescout_followups.py`**
   - Similar validation at start of `main()`

### Acceptance Criteria
- [ ] `/health` endpoint returns 200 with timestamp
- [ ] `/ready` endpoint returns 503 if config invalid, 200 if valid
- [ ] `gmail_bot.py` validates config at startup
- [ ] `freescout_followups.py` validates config at startup
- [ ] Clear error messages for missing configuration

### Example PR Title
```
feat: Add health checks and startup configuration validation
```

---

## PR #4: Add Structured Logging

**Priority:** MEDIUM
**Estimated scope:** Medium (touches many files)

### Problem
System uses `print()` statements instead of proper logging. This makes it hard to:
- Filter by severity (info vs error)
- Parse logs programmatically
- Integrate with log aggregation tools

### Tasks

1. **Add logging configuration module: `logging_config.py`**
   ```python
   import logging
   import sys
   from typing import Optional

   def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
       """Configure logging for the application."""
       log_level = getattr(logging, level.upper(), logging.INFO)

       if json_format:
           # JSON formatter for structured logging
           formatter = logging.Formatter(
               '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
           )
       else:
           formatter = logging.Formatter(
               '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
           )

       handler = logging.StreamHandler(sys.stdout)
       handler.setFormatter(formatter)

       root_logger = logging.getLogger()
       root_logger.setLevel(log_level)
       root_logger.addHandler(handler)

   def get_logger(name: str) -> logging.Logger:
       """Get a logger instance."""
       return logging.getLogger(name)
   ```

2. **Replace print() with logging in `gmail_bot.py`**
   - Add at top: `from logging_config import get_logger`
   - Create logger: `logger = get_logger(__name__)`
   - Replace all `print()` calls:
     - Errors → `logger.error()`
     - Warnings → `logger.warning()`
     - Info → `logger.info()`
     - Debug → `logger.debug()`
   - Example:
     ```python
     # Before
     print(f"Failed to add draft reply to {conversation_id}: {exc}")

     # After
     logger.error(f"Failed to add draft reply to {conversation_id}: {exc}")
     ```

3. **Replace print() in other files**
   - `utils.py`
   - `freescout_followups.py`
   - `webhook_server.py`

4. **Add logging configuration to config.yaml**
   ```yaml
   logging:
     level: "INFO"  # DEBUG, INFO, WARNING, ERROR
     json_format: false  # true for structured logging
   ```

5. **Call setup in main() functions**
   ```python
   from logging_config import setup_logging

   def main():
       settings = get_settings()
       setup_logging(
           level=settings.get("LOG_LEVEL", "INFO"),
           json_format=settings.get("LOG_JSON", False)
       )
       # ... rest of main
   ```

### Acceptance Criteria
- [ ] All print() statements replaced with logger calls
- [ ] Logging configured at application startup
- [ ] Log levels used appropriately (debug/info/warning/error)
- [ ] JSON logging optional via config
- [ ] Existing functionality unchanged

### Example PR Title
```
refactor: Replace print statements with structured logging
```

---

## PR #5: Add Operational Runbook

**Priority:** LOW
**Estimated scope:** Small (documentation only)

### Problem
No troubleshooting guide for operators.

### Tasks

1. **Create `RUNBOOK.md`**
   - Include sections:
     - **Architecture Overview** (diagram or description)
     - **Common Issues and Solutions**
       - Gmail API quota exceeded
       - FreeScout API down
       - Webhook secret mismatch
       - SQLite database locked
       - OpenAI API errors
     - **Monitoring Queries**
       - SQLite queries to check system health
       - Example: Count messages by status
       - Example: Find failed messages
       - Example: Check bot draft state
     - **Recovery Procedures**
       - How to replay failed messages
       - How to reset stuck drafts
       - How to clear terminal filtered state
     - **Log Analysis**
       - How to find errors in journalctl
       - How to trace message processing flow
     - **Configuration Changes**
       - How to enable/disable webhooks
       - How to adjust polling interval
       - How to change AI models

2. **Add example queries section**
   ```sql
   -- Check message processing status
   SELECT status, COUNT(*) FROM processed_messages GROUP BY status;

   -- Find recent failures
   SELECT * FROM processed_messages
   WHERE status = 'failed'
   ORDER BY processed_at DESC LIMIT 10;

   -- Check thread mappings
   SELECT COUNT(*) FROM thread_map;

   -- Find conversations with bot drafts
   SELECT * FROM bot_drafts
   ORDER BY last_generated_at DESC LIMIT 10;
   ```

3. **Add troubleshooting flowcharts**
   - "Message not creating ticket" → check flowchart
   - "Draft not appearing" → check flowchart
   - "Webhook not firing" → check flowchart

### Acceptance Criteria
- [ ] RUNBOOK.md created with all sections
- [ ] Common failure modes documented
- [ ] Recovery procedures clear and actionable
- [ ] Example queries provided

### Example PR Title
```
docs: Add operational runbook for troubleshooting
```

---

## PR #6: Expand Test Coverage (Optional)

**Priority:** LOW
**Estimated scope:** Large

### Problem
Test coverage is limited. Missing tests for:
- Webhook handler
- Follow-up automation
- Draft hash tracking
- Integration tests

### Tasks

1. **Create `tests/test_webhook_handler.py`**
   - Test valid webhook with correct secret
   - Test webhook with invalid secret (401)
   - Test webhook with missing payload (400)
   - Test webhook with missing conversation_id (400)
   - Test webhook event logging

2. **Create `tests/test_followups.py`**
   - Test conversation qualification logic
   - Test required/excluded tags filtering
   - Test age calculation
   - Test draft creation
   - Test P0 notification logic

3. **Create `tests/test_draft_tracking.py`**
   - Test draft creation (no existing draft)
   - Test draft update (unchanged hash)
   - Test draft skip (hash mismatch - human edited)
   - Test thread not found (creates new draft)

4. **Add integration test: `tests/test_integration.py`**
   - Test full flow: Gmail message → FreeScout conversation → draft
   - Mock external APIs (Gmail, FreeScout, OpenAI)
   - Verify database state at each step

5. **Add test runner script: `run_tests.sh`**
   ```bash
   #!/bin/bash
   python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
   ```

### Acceptance Criteria
- [ ] All new test files created
- [ ] Tests pass
- [ ] Coverage increased (aim for >80% on core logic)
- [ ] Integration test covers end-to-end flow

### Example PR Title
```
test: Expand coverage for webhooks, follow-ups, and draft tracking
```

---

## PR #7: Add Basic Metrics (Optional)

**Priority:** LOW
**Estimated scope:** Medium

### Problem
No visibility into system behavior in production.

### Tasks

1. **Add simple metrics endpoint to `webhook_server.py`**
   ```python
   from collections import defaultdict
   from datetime import datetime, timezone

   # Simple in-memory metrics
   METRICS = {
       "webhooks_received": 0,
       "webhooks_failed": 0,
       "last_webhook": None,
       "uptime_start": datetime.now(timezone.utc).isoformat()
   }

   @app.get("/metrics")
   async def metrics():
       """Basic metrics endpoint."""
       return METRICS
   ```

2. **Increment metrics in webhook handler**
   ```python
   METRICS["webhooks_received"] += 1
   METRICS["last_webhook"] = datetime.now(timezone.utc).isoformat()

   # On error:
   METRICS["webhooks_failed"] += 1
   ```

3. **Add database metrics to `/metrics` endpoint**
   ```python
   from storage import TicketStore

   @app.get("/metrics")
   async def metrics():
       """Metrics with database stats."""
       store = TicketStore("./csm.sqlite")

       # Count messages by status
       conn = store.conn
       cursor = conn.execute(
           "SELECT status, COUNT(*) FROM processed_messages GROUP BY status"
       )
       status_counts = dict(cursor.fetchall())

       # Count bot drafts
       cursor = conn.execute("SELECT COUNT(*) FROM bot_drafts")
       bot_draft_count = cursor.fetchone()[0]

       return {
           **METRICS,
           "db_messages_by_status": status_counts,
           "db_bot_drafts": bot_draft_count,
       }
   ```

4. **Add metrics to gmail_bot.py main()**
   - After processing, log summary metrics
   - Consider writing to metrics file or StatsD

### Acceptance Criteria
- [ ] `/metrics` endpoint returns basic stats
- [ ] Metrics include webhook counts
- [ ] Metrics include database stats
- [ ] Metrics endpoint documented in README

### Example PR Title
```
feat: Add basic metrics endpoint
```

---

## Priority Recommendation

**Start with these in order:**

1. **PR #1** (Update README) - Quick win, high value
2. **PR #2** (Deployment files) - Enables production deployment
3. **PR #3** (Health checks) - Essential for production monitoring
4. **PR #4** (Structured logging) - Helps with operations
5. **PR #5** (Runbook) - Helps with incident response

**Optional (if time permits):**
- PR #6 (Test coverage) - Nice to have, prevents regressions
- PR #7 (Metrics) - Helpful for monitoring

---

## PR Best Practices for Codex

1. **One PR per task** - Don't combine multiple tasks
2. **Clear commit messages** - Follow conventional commits format
3. **Update relevant docs** - If you change code, update README/RUNBOOK
4. **Test before committing** - Run existing tests to ensure no regressions
5. **Small, focused changes** - Easier to review and merge

---

## Questions to Ask Before Starting

- Which PR should I start with? (Recommend #1)
- Do you want Docker or systemd deployment (or both)?
- What log aggregation tools do you use? (affects PR #4)
- Do you need Prometheus metrics or simple JSON? (affects PR #7)
