"""SQLite-backed ticket store for idempotent processing."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from utils import get_settings


class TicketStore:
    """Single SQLite-based store for processed messages and thread mapping."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = Path(db_path or settings.get("TICKET_SQLITE_PATH") or "./csm.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT,
                    conv_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_map (
                    thread_id TEXT PRIMARY KEY,
                    conv_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def processed_success(self, message_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM processed_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            return bool(row and row[0] == "success")

    def record_success(self, message_id: str, conv_id: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages (message_id, status, error, conv_id, updated_at)
                VALUES (?, 'success', NULL, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(message_id) DO UPDATE SET
                    status=excluded.status,
                    error=excluded.error,
                    conv_id=excluded.conv_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (message_id, conv_id),
            )

    def record_failure(self, message_id: str, error: str, conv_id: Optional[str] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages (message_id, status, error, conv_id, updated_at)
                VALUES (?, 'failed', ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(message_id) DO UPDATE SET
                    status=excluded.status,
                    error=excluded.error,
                    conv_id=excluded.conv_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (message_id, error, conv_id),
            )

    def get_thread_conversation(self, thread_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT conv_id FROM thread_map WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            return row[0] if row else None

    def upsert_thread_conversation(self, thread_id: str, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO thread_map (thread_id, conv_id, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    conv_id=excluded.conv_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (thread_id, conv_id),
            )
