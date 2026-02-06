import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("csm.storage")


class TicketStore:
    """Durable SQLite store for message processing and thread mapping."""

    # Default timeout for SQLite connection (seconds)
    DEFAULT_CONNECTION_TIMEOUT = 30.0

    def __init__(self, sqlite_path: str, timeout: float = DEFAULT_CONNECTION_TIMEOUT) -> None:
        self.sqlite_path = sqlite_path
        try:
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.sqlite_path, timeout=timeout, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout = 10000")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._init_schema()
        except sqlite3.Error as e:
            LOGGER.error("Failed to initialize SQLite database at %s: %s", sqlite_path, e)
            raise RuntimeError(f"Database initialization failed: {e}") from e
        except OSError as e:
            LOGGER.error("Failed to create database directory for %s: %s", sqlite_path, e)
            raise RuntimeError(f"Database directory creation failed: {e}") from e

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_messages (
                gmail_message_id TEXT PRIMARY KEY,
                gmail_thread_id TEXT,
                freescout_conversation_id TEXT,
                status TEXT NOT NULL CHECK(status IN ('success', 'filtered', 'failed', 'processing')),
                action TEXT CHECK(action IN ('create','append')),
                error TEXT,
                processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._migrate_processed_messages()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_map (
                gmail_thread_id TEXT PRIMARY KEY,
                freescout_conversation_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_drafts (
                freescout_conversation_id TEXT PRIMARY KEY,
                freescout_thread_id TEXT NULL,
                last_hash TEXT,
                last_generated_at TEXT
            )
            """
        )
        self._migrate_bot_drafts()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS webhook_counters (
                counter TEXT PRIMARY KEY,
                value INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS webhook_nonces (
                nonce TEXT PRIMARY KEY,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_webhook_nonces_created_at
            ON webhook_nonces(created_at)
            """
        )
        # Add index on status column for faster queries by status
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_processed_messages_status
            ON processed_messages(status)
            """
        )
        # Add index on gmail_thread_id for faster thread lookups
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_processed_messages_thread_id
            ON processed_messages(gmail_thread_id)
            """
        )
        self._conn.commit()

    def _migrate_processed_messages(self) -> None:
        # Use PRAGMA table_info for reliable column checking instead of parsing SQL schema
        cur = self._conn.execute("PRAGMA table_info(processed_messages)")
        table_info = cur.fetchall()
        if not table_info:
            return  # Table doesn't exist yet, will be created by _init_schema

        columns = [col[1] for col in table_info]
        needs_action = "action" not in columns

        # Check if status constraint needs updating by querying sqlite_master
        # and looking for the specific constraint values we need
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'processed_messages'"
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return

        schema_sql = row[0]
        # Check for the presence of status values in the CHECK constraint
        # We look for the exact constraint pattern to avoid false matches
        has_filtered_status = "'filtered'" in schema_sql or '"filtered"' in schema_sql
        has_processing_status = "'processing'" in schema_sql or '"processing"' in schema_sql
        needs_filtered = not has_filtered_status
        needs_processing = not has_processing_status

        if not needs_filtered and not needs_processing and not needs_action:
            return
        if needs_filtered or needs_processing:
            has_action = "action" in columns
            self._conn.execute("ALTER TABLE processed_messages RENAME TO processed_messages_old")
            self._conn.execute(
                """
                CREATE TABLE processed_messages (
                    gmail_message_id TEXT PRIMARY KEY,
                    gmail_thread_id TEXT,
                    freescout_conversation_id TEXT,
                    status TEXT NOT NULL CHECK(status IN ('success', 'filtered', 'failed', 'processing')),
                    action TEXT CHECK(action IN ('create','append')),
                    error TEXT,
                    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Use conditional SQL instead of f-string for safer query construction
            if has_action:
                self._conn.execute(
                    """
                    INSERT INTO processed_messages (
                        gmail_message_id,
                        gmail_thread_id,
                        freescout_conversation_id,
                        status,
                        action,
                        error,
                        processed_at
                    )
                    SELECT
                        gmail_message_id,
                        gmail_thread_id,
                        freescout_conversation_id,
                        status,
                        action,
                        error,
                        processed_at
                    FROM processed_messages_old
                    """
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO processed_messages (
                        gmail_message_id,
                        gmail_thread_id,
                        freescout_conversation_id,
                        status,
                        action,
                        error,
                        processed_at
                    )
                    SELECT
                        gmail_message_id,
                        gmail_thread_id,
                        freescout_conversation_id,
                        status,
                        NULL,
                        error,
                        processed_at
                    FROM processed_messages_old
                    """
                )
            self._conn.execute("DROP TABLE processed_messages_old")
        elif needs_action:
            self._conn.execute(
                "ALTER TABLE processed_messages "
                "ADD COLUMN action TEXT CHECK(action IN ('create','append'))"
            )
        self._conn.commit()

    def _migrate_bot_drafts(self) -> None:
        cur = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'bot_drafts'"
        )
        row = cur.fetchone()
        if not row or not row[0]:
            return
        schema_sql = row[0]
        if "freescout_conversation_id INTEGER" not in schema_sql:
            return
        self._conn.execute("ALTER TABLE bot_drafts RENAME TO bot_drafts_old")
        self._conn.execute(
            """
            CREATE TABLE bot_drafts (
                freescout_conversation_id TEXT PRIMARY KEY,
                freescout_thread_id TEXT NULL,
                last_hash TEXT,
                last_generated_at TEXT
            )
            """
        )
        self._conn.execute(
            """
            INSERT INTO bot_drafts (
                freescout_conversation_id,
                freescout_thread_id,
                last_hash,
                last_generated_at
            )
            SELECT
                freescout_conversation_id,
                freescout_thread_id,
                last_hash,
                last_generated_at
            FROM bot_drafts_old
            """
        )
        self._conn.execute("DROP TABLE bot_drafts_old")
        self._conn.commit()

    # Valid range for processing timeout to prevent unreasonable values
    MIN_PROCESSING_TIMEOUT_MINUTES = 1
    MAX_PROCESSING_TIMEOUT_MINUTES = 1440  # 24 hours

    def mark_processing_if_new(
        self, gmail_message_id: str, gmail_thread_id: Optional[str], processing_timeout_minutes: int = 30
    ) -> bool:
        """
        Mark message as processing if new or if stuck in processing for too long.

        Args:
            gmail_message_id: The Gmail message ID
            gmail_thread_id: The Gmail thread ID
            processing_timeout_minutes: How many minutes before a stale 'processing' state can be reclaimed

        Returns:
            True if successfully claimed, False if already being processed or completed

        Raises:
            ValueError: If processing_timeout_minutes is not within valid range
        """
        # Validate timeout parameter to prevent SQL injection and unreasonable values
        if not isinstance(processing_timeout_minutes, int):
            raise ValueError(f"processing_timeout_minutes must be an integer, got {type(processing_timeout_minutes).__name__}")
        if processing_timeout_minutes < self.MIN_PROCESSING_TIMEOUT_MINUTES:
            raise ValueError(f"processing_timeout_minutes must be at least {self.MIN_PROCESSING_TIMEOUT_MINUTES}")
        if processing_timeout_minutes > self.MAX_PROCESSING_TIMEOUT_MINUTES:
            raise ValueError(f"processing_timeout_minutes must not exceed {self.MAX_PROCESSING_TIMEOUT_MINUTES}")

        # Use IMMEDIATE transaction to prevent race conditions
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            # First, try to insert a new row
            cur = self._conn.execute(
                """
                INSERT INTO processed_messages (
                    gmail_message_id,
                    gmail_thread_id,
                    freescout_conversation_id,
                    status,
                    action,
                    error,
                    processed_at
                )
                VALUES (?, ?, NULL, 'processing', NULL, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(gmail_message_id) DO NOTHING
                """,
                (gmail_message_id, gmail_thread_id),
            )

            if cur.rowcount == 1:
                # Successfully inserted new row
                self._conn.commit()
                return True

            # Message already exists. Check if it's stale and can be reclaimed
            cur = self._conn.execute(
                """
                SELECT status, processed_at
                FROM processed_messages
                WHERE gmail_message_id = ?
                """,
                (gmail_message_id,),
            )
            row = cur.fetchone()

            if not row:
                # Race condition - message was deleted? Allow retry
                self._conn.rollback()
                return False

            status, processed_at = row

            # If status is terminal (success/filtered), don't reclaim
            if status in ('success', 'filtered'):
                self._conn.rollback()
                return False

            # If status is failed, allow reclaim (retry logic)
            # If status is processing, check if it's stale
            if status == 'processing':
                # Check if the processing timestamp is too old
                # Use CAST to ensure processing_timeout_minutes is treated as an integer
                # This prevents SQL injection while still using the parameter
                cur = self._conn.execute(
                    """
                    SELECT CASE
                        WHEN datetime(processed_at, '+' || CAST(? AS INTEGER) || ' minutes') < datetime('now')
                        THEN 1 ELSE 0 END as is_stale
                    FROM processed_messages
                    WHERE gmail_message_id = ?
                    """,
                    (processing_timeout_minutes, gmail_message_id),
                )
                stale_row = cur.fetchone()
                if stale_row and stale_row[0] == 1:
                    # Reclaim stale processing state
                    self._conn.execute(
                        """
                        UPDATE processed_messages
                        SET status = 'processing',
                            processed_at = CURRENT_TIMESTAMP,
                            error = 'reclaimed from stale processing state'
                        WHERE gmail_message_id = ?
                        """,
                        (gmail_message_id,),
                    )
                    self._conn.commit()
                    return True
                else:
                    # Still being actively processed
                    self._conn.rollback()
                    return False

            # Status is 'failed', allow reclaim for retry
            self._conn.execute(
                """
                UPDATE processed_messages
                SET status = 'processing',
                    processed_at = CURRENT_TIMESTAMP
                WHERE gmail_message_id = ?
                """,
                (gmail_message_id,),
            )
            self._conn.commit()
            return True

        except Exception as e:
            self._conn.rollback()
            LOGGER.error(
                "Error in mark_processing_if_new for message %s: %s (type: %s)",
                gmail_message_id,
                e,
                type(e).__name__,
            )
            raise

    def processed_success(self, gmail_message_id: str) -> bool:
        cur = self._conn.execute(
            """
            SELECT 1
            FROM processed_messages
            WHERE gmail_message_id = ? AND status = 'success'
            """,
            (gmail_message_id,),
        )
        return cur.fetchone() is not None

    def processed_filtered(self, gmail_message_id: str) -> bool:
        cur = self._conn.execute(
            """
            SELECT 1
            FROM processed_messages
            WHERE gmail_message_id = ? AND status = 'filtered'
            """,
            (gmail_message_id,),
        )
        return cur.fetchone() is not None

    def processed_terminal(self, gmail_message_id: str) -> bool:
        cur = self._conn.execute(
            """
            SELECT 1
            FROM processed_messages
            WHERE gmail_message_id = ? AND status IN ('success', 'filtered')
            """,
            (gmail_message_id,),
        )
        return cur.fetchone() is not None

    def get_processed_status_counts(self) -> dict:
        cur = self._conn.execute(
            """
            SELECT status, COUNT(*)
            FROM processed_messages
            GROUP BY status
            """
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_recent_failures(self, limit: int = 10) -> List[dict]:
        """Get the most recent failed messages for debugging.

        Args:
            limit: Maximum number of failures to return (must be positive, default: 10)

        Returns:
            List of failure records with gmail_message_id, error, and processed_at

        Raises:
            ValueError: If limit is not a positive integer
        """
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be a positive integer, got {limit!r}")
        cur = self._conn.execute(
            """
            SELECT gmail_message_id, error, processed_at
            FROM processed_messages
            WHERE status = 'failed'
            ORDER BY processed_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "gmail_message_id": row[0],
                "error": row[1],
                "processed_at": row[2],
            }
            for row in cur.fetchall()
        ]

    def get_conv_id(self, gmail_thread_id: str) -> Optional[str]:
        cur = self._conn.execute(
            "SELECT freescout_conversation_id FROM thread_map WHERE gmail_thread_id = ?",
            (gmail_thread_id,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        return None

    def get_conversation_id_for_thread(self, gmail_thread_id: str) -> Optional[str]:
        return self.get_conv_id(gmail_thread_id)

    def upsert_thread_map(self, gmail_thread_id: str, conv_id: str) -> None:
        self._conn.execute(
            """
            INSERT INTO thread_map (
                gmail_thread_id,
                freescout_conversation_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_thread_id) DO UPDATE SET
                freescout_conversation_id = excluded.freescout_conversation_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (gmail_thread_id, conv_id),
        )
        self._conn.commit()

    def mark_success(
        self,
        gmail_message_id: str,
        gmail_thread_id: str,
        conv_id: Optional[str],
        action: str,
        error: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO processed_messages (
                gmail_message_id,
                gmail_thread_id,
                freescout_conversation_id,
                status,
                action,
                error,
                processed_at
            )
            VALUES (?, ?, ?, 'success', ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_message_id) DO UPDATE SET
                gmail_thread_id = excluded.gmail_thread_id,
                freescout_conversation_id = excluded.freescout_conversation_id,
                status = 'success',
                action = excluded.action,
                error = excluded.error,
                processed_at = CURRENT_TIMESTAMP
            """,
            (gmail_message_id, gmail_thread_id, conv_id, action, error),
        )
        self._conn.commit()

    def mark_failed(
        self,
        gmail_message_id: str,
        gmail_thread_id: Optional[str],
        error: str,
        conv_id: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO processed_messages (
                gmail_message_id,
                gmail_thread_id,
                freescout_conversation_id,
                status,
                error,
                processed_at
            )
            VALUES (?, ?, ?, 'failed', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_message_id) DO UPDATE SET
                gmail_thread_id = excluded.gmail_thread_id,
                freescout_conversation_id = excluded.freescout_conversation_id,
                status = 'failed',
                error = excluded.error,
                processed_at = CURRENT_TIMESTAMP
            """,
            (gmail_message_id, gmail_thread_id, conv_id, error),
        )
        self._conn.commit()

    def mark_filtered(
        self,
        gmail_message_id: str,
        gmail_thread_id: str,
        reason: str,
    ) -> None:
        """Record a terminal filtered state without requiring a conversation ID."""
        self._conn.execute(
            """
            INSERT INTO processed_messages (
                gmail_message_id,
                gmail_thread_id,
                freescout_conversation_id,
                status,
                error,
                processed_at
            )
            VALUES (?, ?, NULL, 'filtered', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_message_id) DO UPDATE SET
                gmail_thread_id = excluded.gmail_thread_id,
                freescout_conversation_id = NULL,
                status = 'filtered',
                error = excluded.error,
                processed_at = CURRENT_TIMESTAMP
            """,
            (gmail_message_id, gmail_thread_id, reason),
        )
        self._conn.commit()

    def get_bot_draft(self, freescout_conversation_id: str) -> Optional[dict]:
        cur = self._conn.execute(
            """
            SELECT freescout_conversation_id,
                freescout_thread_id,
                last_hash,
                last_generated_at
            FROM bot_drafts
            WHERE freescout_conversation_id = ?
            """,
            (freescout_conversation_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "freescout_conversation_id": row[0],
            "freescout_thread_id": row[1],
            "last_hash": row[2],
            "last_generated_at": row[3],
        }

    def upsert_bot_draft(
        self,
        freescout_conversation_id: str,
        freescout_thread_id: Optional[str],
        last_hash: str,
        last_generated_at: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO bot_drafts (
                freescout_conversation_id,
                freescout_thread_id,
                last_hash,
                last_generated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(freescout_conversation_id) DO UPDATE SET
                freescout_thread_id = excluded.freescout_thread_id,
                last_hash = excluded.last_hash,
                last_generated_at = excluded.last_generated_at
            """,
            (
                freescout_conversation_id,
                freescout_thread_id,
                last_hash,
                last_generated_at,
            ),
        )
        self._conn.commit()

    def get_bot_draft_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM bot_drafts")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def can_create_new_draft(self, conversation_id: str, max_drafts: Optional[int]) -> bool:
        """Check if a new draft can be created without exceeding the limit.

        This is a pre-check that should be called before creating a draft on
        the external system (FreeScout) to avoid orphaned drafts.

        Args:
            conversation_id: The conversation ID to check
            max_drafts: Maximum number of drafts allowed (None = unlimited)

        Returns:
            True if a draft can be created (either updating existing or under limit)
        """
        if max_drafts is None:
            return True

        # Check if this conversation already has a draft (update case - always allowed)
        cur = self._conn.execute(
            "SELECT 1 FROM bot_drafts WHERE freescout_conversation_id = ?",
            (conversation_id,),
        )
        if cur.fetchone() is not None:
            return True  # Updating existing draft is always allowed

        # Check if under the limit for new drafts
        cur = self._conn.execute("SELECT COUNT(*) FROM bot_drafts")
        row = cur.fetchone()
        current_count = int(row[0]) if row else 0
        return current_count < max_drafts

    def atomic_upsert_bot_draft_if_under_limit(
        self,
        freescout_conversation_id: str,
        freescout_thread_id: Optional[str],
        last_hash: str,
        last_generated_at: str,
        max_drafts: Optional[int],
    ) -> bool:
        """Atomically upsert a bot draft only if under the max draft limit.

        Uses database-level locking to prevent race conditions where concurrent
        processes could exceed the limit.

        Args:
            freescout_conversation_id: The conversation ID
            freescout_thread_id: The thread ID (optional)
            last_hash: Hash of the draft content
            last_generated_at: ISO timestamp of generation
            max_drafts: Maximum number of drafts allowed (None = unlimited)

        Returns:
            True if the draft was upserted, False if limit was reached
        """
        if max_drafts is None:
            # No limit, just upsert
            self.upsert_bot_draft(
                freescout_conversation_id,
                freescout_thread_id,
                last_hash,
                last_generated_at,
            )
            return True

        # Use IMMEDIATE transaction to acquire write lock before checking count
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            # Check if this conversation already has a draft (update case)
            cur = self._conn.execute(
                "SELECT 1 FROM bot_drafts WHERE freescout_conversation_id = ?",
                (freescout_conversation_id,),
            )
            already_exists = cur.fetchone() is not None

            if not already_exists:
                # For new drafts, check the count limit
                cur = self._conn.execute("SELECT COUNT(*) FROM bot_drafts")
                row = cur.fetchone()
                current_count = int(row[0]) if row else 0

                if current_count >= max_drafts:
                    self._conn.rollback()
                    return False

            # Either updating existing or inserting within limit
            self._conn.execute(
                """
                INSERT INTO bot_drafts (
                    freescout_conversation_id,
                    freescout_thread_id,
                    last_hash,
                    last_generated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(freescout_conversation_id) DO UPDATE SET
                    freescout_thread_id = excluded.freescout_thread_id,
                    last_hash = excluded.last_hash,
                    last_generated_at = excluded.last_generated_at
                """,
                (
                    freescout_conversation_id,
                    freescout_thread_id,
                    last_hash,
                    last_generated_at,
                ),
            )
            self._conn.commit()
            return True
        except Exception as e:
            self._conn.rollback()
            LOGGER.error(
                "Error in atomic_upsert_bot_draft_if_under_limit for conversation %s: %s (type: %s)",
                freescout_conversation_id,
                e,
                type(e).__name__,
            )
            raise

    def get_message_action(self, gmail_message_id: str) -> Optional[str]:
        """Get the action recorded for a processed message (for testing/debugging)."""
        cur = self._conn.execute(
            "SELECT action FROM processed_messages WHERE gmail_message_id = ?",
            (gmail_message_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def increment_webhook_counter(self, counter: str, amount: int = 1) -> None:
        self._conn.execute(
            """
            INSERT INTO webhook_counters (counter, value)
            VALUES (?, ?)
            ON CONFLICT(counter) DO UPDATE SET
                value = value + excluded.value
            """,
            (counter, amount),
        )
        self._conn.commit()

    def get_webhook_counters(self) -> dict:
        cur = self._conn.execute("SELECT counter, value FROM webhook_counters")
        return {row[0]: int(row[1]) for row in cur.fetchall()}

    def check_and_record_webhook_nonce(self, nonce: str, ttl_seconds: int, cache_size: int) -> bool:
        """Atomically check and store a webhook nonce for replay protection."""
        now = time.time()
        cutoff_time = now - max(ttl_seconds, 0)
        hard_limit = max(cache_size, 1)

        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "DELETE FROM webhook_nonces WHERE created_at < ?",
                (cutoff_time,),
            )
            cur = self._conn.execute(
                "SELECT 1 FROM webhook_nonces WHERE nonce = ? LIMIT 1",
                (nonce,),
            )
            if cur.fetchone():
                self._conn.rollback()
                return False

            cur = self._conn.execute("SELECT COUNT(*) FROM webhook_nonces")
            count = int(cur.fetchone()[0])
            if count >= hard_limit:
                to_remove = count - (hard_limit - 1)
                self._conn.execute(
                    """
                    DELETE FROM webhook_nonces
                    WHERE nonce IN (
                        SELECT nonce FROM webhook_nonces
                        ORDER BY created_at ASC
                        LIMIT ?
                    )
                    """,
                    (to_remove,),
                )

            self._conn.execute(
                "INSERT INTO webhook_nonces (nonce, created_at) VALUES (?, ?)",
                (nonce, now),
            )
            self._conn.commit()
            return True
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error as e:
            LOGGER.warning("Error closing database connection: %s", e)

    def __enter__(self) -> "TicketStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        # Best-effort cleanup during garbage collection. Prefer using context manager.
        try:
            if hasattr(self, "_conn") and self._conn:
                self._conn.close()
        except (sqlite3.Error, TypeError, AttributeError):
            # Suppress errors during interpreter shutdown when modules may be None
            pass
