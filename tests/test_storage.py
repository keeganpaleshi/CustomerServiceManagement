import os
import tempfile
import unittest

from storage import TicketStore


class TicketStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "csm.sqlite")
        self.store = TicketStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_fail_then_success(self):
        self.store.mark_failed("msg-fail", "thread-fail", "boom")
        self.assertFalse(self.store.processed_success("msg-fail"))

        self.store.mark_success("msg-success", "thread-success", "conv-123")
        self.assertTrue(self.store.processed_success("msg-success"))

    def test_thread_map_round_trip(self):
        self.store.upsert_thread_map("thread-1", "conv-a")
        self.assertEqual(self.store.get_conv_id("thread-1"), "conv-a")

        # ensure upsert updates the conversation id
        self.store.upsert_thread_map("thread-1", "conv-b")
        self.assertEqual(self.store.get_conv_id("thread-1"), "conv-b")


if __name__ == "__main__":
    unittest.main()
