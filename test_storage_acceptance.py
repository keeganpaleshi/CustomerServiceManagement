"""Minimal checks for TicketStore correctness."""
from pathlib import Path

from storage import TicketStore


def run_acceptance():
    db_path = Path("./acceptance.sqlite")
    if db_path.exists():
        db_path.unlink()

    store = TicketStore(str(db_path))

    store.mark_failed("msg-fail", "th-1", "boom")
    assert not store.processed_success("msg-fail"), "Failed messages must not count as success"

    store.mark_success("msg-ok", "th-2", "conv-1")
    assert store.processed_success("msg-ok"), "Successful message should be recorded"

    store.upsert_thread_map("th-2", "conv-1")
    assert store.get_conv_id("th-2") == "conv-1", "Thread map should round trip"

    store.close()
    db_path.unlink()


if __name__ == "__main__":
    run_acceptance()
    print("acceptance checks passed")
