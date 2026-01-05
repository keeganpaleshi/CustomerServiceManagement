import sqlite3
from pathlib import Path
from typing import Optional


class TicketStore:
    """Durable SQLite store for message processing and thread mapping."""

    def __init__(self, sqlite_path: str) -> None:
        self.sqlite_path = sqlite_path
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self._ensure_table(
            "processed_messages",
            [
                "gmail_message_id",
                "gmail_thread_id",
                "freescout_conversation_id",
                "status",
                "error",
                "processed_at",
            ],
            """
            CREATE TABLE processed_messages (
                gmail_message_id TEXT PRIMARY KEY,
                gmail_thread_id TEXT,
                freescout_conversation_id TEXT,
                status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                error TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        self._ensure_table(
            "thread_map",
            [
                "gmail_thread_id",
                "freescout_conversation_id",
                "created_at",
                "updated_at",
            ],
            """
            CREATE TABLE thread_map (
                gmail_thread_id TEXT PRIMARY KEY,
                freescout_conversation_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
        )
        self._conn.commit()

    def _ensure_table(self, table: str, expected_columns: list[str], ddl: str) -> None:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        exists = cur.fetchone() is not None
        if exists:
            cur = self._conn.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cur.fetchall()]
            if columns == expected_columns:
                return
            self._conn.execute(f"DROP TABLE {table}")
        self._conn.execute(ddl)

    def processed_success(self, message_id: str) -> bool:
        cur = self._conn.execute(
            """
            SELECT 1 FROM processed_messages
            WHERE gmail_message_id = ? AND status = 'success'
            """,
            (message_id,),
        )
        return cur.fetchone() is not None

    def get_conv_id(self, thread_id: str) -> Optional[str]:
        cur = self._conn.execute(
            "SELECT freescout_conversation_id FROM thread_map WHERE gmail_thread_id = ?",
            (thread_id,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        return None

    def upsert_thread_map(self, thread_id: str, conv_id: str) -> None:
        self._conn.execute(
            """
            INSERT INTO thread_map (
                gmail_thread_id, freescout_conversation_id, created_at, updated_at
            )
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_thread_id) DO UPDATE SET
                freescout_conversation_id = excluded.freescout_conversation_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, conv_id),
        )
        self._conn.commit()

    def mark_success(
        self,
        message_id: str,
        thread_id: Optional[str] = None,
        conv_id: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO processed_messages (
                gmail_message_id, gmail_thread_id, freescout_conversation_id,
                status, error, processed_at
            )
            VALUES (?, ?, ?, 'success', NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_message_id) DO UPDATE SET
                gmail_thread_id = excluded.gmail_thread_id,
                freescout_conversation_id = excluded.freescout_conversation_id,
                status = 'success',
                error = NULL,
                processed_at = CURRENT_TIMESTAMP
            """,
            (message_id, thread_id, conv_id),
        )
        self._conn.commit()

    def mark_failed(
        self,
        message_id: str,
        thread_id: Optional[str],
        error: str,
        conv_id: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO processed_messages (
                gmail_message_id, gmail_thread_id, freescout_conversation_id,
                status, error, processed_at
            )
            VALUES (?, ?, ?, 'failed', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(gmail_message_id) DO UPDATE SET
                gmail_thread_id = excluded.gmail_thread_id,
                freescout_conversation_id = excluded.freescout_conversation_id,
                status = 'failed',
                error = excluded.error,
                processed_at = CURRENT_TIMESTAMP
            """,
            (message_id, thread_id, conv_id, error),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
