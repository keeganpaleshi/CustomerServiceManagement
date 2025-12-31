import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class TicketStore:
    """Lightweight SQLite-backed store for idempotency and thread mapping."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or "csm.sqlite")
        if self.db_path.parent and not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    gmail_message_id TEXT PRIMARY KEY,
                    gmail_thread_id TEXT,
                    freescout_conversation_id INTEGER,
                    status TEXT NOT NULL,
                    error TEXT,
                    processed_at TEXT NOT NULL,
                    CHECK (status IN ('success', 'failed'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_map (
                    gmail_thread_id TEXT PRIMARY KEY,
                    freescout_conversation_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get_processed_message(self, gmail_message_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM processed_messages WHERE gmail_message_id = ?",
                (gmail_message_id,),
            ).fetchone()
            return dict(row) if row else None

    def record_processed_message(
        self,
        *,
        gmail_message_id: str,
        gmail_thread_id: Optional[str],
        freescout_conversation_id: Optional[int],
        status: str,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages (
                    gmail_message_id,
                    gmail_thread_id,
                    freescout_conversation_id,
                    status,
                    error,
                    processed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(gmail_message_id) DO UPDATE SET
                    gmail_thread_id=excluded.gmail_thread_id,
                    freescout_conversation_id=excluded.freescout_conversation_id,
                    status=excluded.status,
                    error=excluded.error,
                    processed_at=excluded.processed_at
                """,
                (
                    gmail_message_id,
                    gmail_thread_id,
                    freescout_conversation_id,
                    status,
                    error,
                    now,
                ),
            )

    def get_thread_conversation(self, gmail_thread_id: str) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT freescout_conversation_id FROM thread_map WHERE gmail_thread_id = ?",
                (gmail_thread_id,),
            ).fetchone()
            return int(row[0]) if row else None

    def upsert_thread_conversation(self, gmail_thread_id: str, conversation_id: int) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO thread_map (
                    gmail_thread_id,
                    freescout_conversation_id,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(gmail_thread_id) DO UPDATE SET
                    freescout_conversation_id=excluded.freescout_conversation_id,
                    updated_at=excluded.updated_at
                """,
                (gmail_thread_id, conversation_id, now, now),
            )
