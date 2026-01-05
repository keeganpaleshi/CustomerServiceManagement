import sqlite3
from pathlib import Path
from typing import Optional


class TicketStore:
    """Durable SQLite store for message processing and thread mapping."""

    def __init__(self, db_path: str = "./csm.sqlite") -> None:
        self.db_path = db_path or "./csm.sqlite"
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id TEXT PRIMARY KEY,
                status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                error TEXT,
                conv_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_map (
                thread_id TEXT PRIMARY KEY,
                conv_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def processed_success(self, message_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = ? AND status = 'success'",
            (message_id,),
        )
        return cur.fetchone() is not None

    def record_success(self, message_id: str, conv_id: Optional[str] = None) -> None:
        self._conn.execute(
            """
            INSERT INTO processed_messages (message_id, status, error, conv_id, created_at, updated_at)
            VALUES (?, 'success', NULL, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(message_id) DO UPDATE SET
                status = 'success',
                error = NULL,
                conv_id = excluded.conv_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (message_id, conv_id),
        )
        self._conn.commit()

    def record_failure(self, message_id: str, error: str, conv_id: Optional[str] = None) -> None:
        self._conn.execute(
            """
            INSERT INTO processed_messages (message_id, status, error, conv_id, created_at, updated_at)
            VALUES (?, 'failed', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(message_id) DO UPDATE SET
                status = 'failed',
                error = excluded.error,
                conv_id = excluded.conv_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (message_id, error, conv_id),
        )
        self._conn.commit()

    def get_thread_conversation(self, thread_id: str) -> Optional[str]:
        cur = self._conn.execute(
            "SELECT conv_id FROM thread_map WHERE thread_id = ?",
            (thread_id,),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        return None

    def upsert_thread_conversation(self, thread_id: str, conv_id: str) -> None:
        self._conn.execute(
            """
            INSERT INTO thread_map (thread_id, conv_id, created_at, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id) DO UPDATE SET
                conv_id = excluded.conv_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, conv_id),
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
