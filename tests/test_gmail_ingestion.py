import unittest
from unittest.mock import Mock, call, patch

import requests

import gmail_bot


def _base_settings():
    return {
        "GMAIL_QUERY": "is:unread",
        "HTTP_TIMEOUT": 5,
        "FREESCOUT_POLL_INTERVAL": 300,
        "GMAIL_USE_CONSOLE": False,
        "MAX_DRAFTS": 10,
        "TICKET_SYSTEM": "none",
        "TICKET_SQLITE_PATH": ":memory:",
        "FREESCOUT_MAILBOX_ID": "mailbox-1",
        "FREESCOUT_GMAIL_THREAD_FIELD_ID": "field-thread",
        "FREESCOUT_GMAIL_MESSAGE_FIELD_ID": "field-message",
    }


def _make_message(message_id: str, thread_id: str) -> dict:
    return {
        "id": message_id,
        "threadId": thread_id,
        "snippet": "snippet body",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Need help"},
                {"name": "From", "value": "Customer <customer@example.com>"},
            ]
        },
    }


class GmailIngestionTests(unittest.TestCase):
    def test_skip_already_processed_skips_freescout(self):
        store = Mock()
        store.processed_success.return_value = True
        store.processed_filtered.return_value = False
        freescout = Mock()
        message = _make_message("msg-1", "thread-1")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "skipped_already_success")
        self.assertEqual(result.reason, "already processed")
        self.assertIsNone(result.freescout_conversation_id)
        store.get_conversation_id_for_thread.assert_not_called()
        freescout.add_customer_thread.assert_not_called()
        freescout.create_conversation.assert_not_called()

    def test_filtered_terminal_marks_filtered(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        freescout = Mock()
        message = _make_message("msg-2", "thread-2")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=True):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "filtered")
        self.assertEqual(result.reason, "filtered: promotional/spam")
        self.assertIsNone(result.freescout_conversation_id)
        store.processed_success.assert_called_once_with("msg-2")
        store.mark_filtered.assert_called_once_with(
            "msg-2",
            "thread-2",
            reason="filtered: promotional/spam",
        )
        store.mark_success.assert_not_called()
        store.mark_failed.assert_not_called()
        freescout.add_customer_thread.assert_not_called()
        freescout.create_conversation.assert_not_called()

    def test_append_existing_thread_marks_success_after_append(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = "conv-123"
        freescout = Mock()
        message = _make_message("msg-3", "thread-3")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "freescout_appended")
        self.assertEqual(result.reason, "append success")
        self.assertEqual(result.freescout_conversation_id, "conv-123")
        store.processed_success.assert_called_once_with("msg-3")
        freescout.add_customer_thread.assert_called_once_with("conv-123", "hello", imported=True)
        freescout.create_conversation.assert_not_called()
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_called_once_with(
            "msg-3",
            "thread-3",
            "conv-123",
            action="append",
        )

    def test_create_new_thread_marks_success_after_upsert(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = None
        freescout = Mock()
        freescout.create_conversation.return_value = {"id": "conv-456"}
        message = _make_message("msg-4", "thread-4")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "get_settings", return_value=_base_settings()):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "freescout_created")
        self.assertEqual(result.reason, "create success")
        self.assertEqual(result.freescout_conversation_id, "conv-456")
        store.processed_success.assert_called_once_with("msg-4")
        freescout.create_conversation.assert_called_once_with(
            "Need help",
            "customer@example.com",
            "hello",
            "mailbox-1",
            thread_id="thread-4",
            message_id="msg-4",
            gmail_thread_field="field-thread",
            gmail_message_field="field-message",
        )
        freescout.add_customer_thread.assert_not_called()
        store.upsert_thread_map.assert_called_once_with("thread-4", "conv-456")
        store.mark_success.assert_called_once_with(
            "msg-4",
            "thread-4",
            "conv-456",
            action="create",
        )

    def test_non_filtered_message_makes_single_freescout_call(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = "conv-321"
        freescout = Mock()
        message = _make_message("msg-7", "thread-7")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "freescout_appended")
        self.assertEqual(
            freescout.method_calls,
            [call.add_customer_thread("conv-321", "hello", imported=True)],
        )

    def test_append_failure_marks_failed(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = "conv-789"
        freescout = Mock()
        freescout.add_customer_thread.side_effect = requests.RequestException("boom")
        message = _make_message("msg-5", "thread-5")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "failed_retryable")
        self.assertEqual(result.reason, "append failed: boom")
        self.assertEqual(result.freescout_conversation_id, "conv-789")
        store.processed_success.assert_called_once_with("msg-5")
        freescout.add_customer_thread.assert_called_once_with("conv-789", "hello", imported=True)
        freescout.create_conversation.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-5", "thread-5", "boom", "conv-789")

    def test_processed_success_skips_before_other_processing(self):
        store = Mock()
        store.processed_success.return_value = True
        store.processed_filtered.return_value = False
        freescout = Mock()
        message = _make_message("msg-8", "thread-8")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "skipped_already_success")
        freescout.create_conversation.assert_not_called()
        freescout.add_customer_thread.assert_not_called()

    def test_processed_filtered_store_value_still_filters_message(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = True
        store.get_conversation_id_for_thread.return_value = "conv-901"
        freescout = Mock()
        message = _make_message("msg-9", "thread-9")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=True):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "filtered")
        self.assertEqual(result.reason, "filtered: promotional/spam")
        freescout.add_customer_thread.assert_not_called()
        freescout.create_conversation.assert_not_called()
        store.mark_filtered.assert_called_once_with(
            "msg-9",
            "thread-9",
            reason="filtered: promotional/spam",
        )

    def test_append_without_freescout_does_not_mark_success(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = "conv-111"
        message = _make_message("msg-11", "thread-11")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(message, store, None, Mock())

        self.assertEqual(result.status, "failed_retryable")
        self.assertEqual(result.reason, "freescout disabled")
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-11", "thread-11", "freescout disabled", "conv-111")

    def test_create_without_freescout_does_not_mark_success(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = None
        message = _make_message("msg-12", "thread-12")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(message, store, None, Mock())

        self.assertEqual(result.status, "failed_retryable")
        self.assertEqual(result.reason, "freescout disabled")
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-12", "thread-12", "freescout disabled")

    def test_create_failure_marks_failed(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = None
        freescout = Mock()
        freescout.create_conversation.side_effect = requests.RequestException("boom")
        message = _make_message("msg-6", "thread-6")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "get_settings", return_value=_base_settings()):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "failed_retryable")
        self.assertEqual(result.reason, "ticket creation failed: boom")
        self.assertIsNone(result.freescout_conversation_id)
        store.processed_success.assert_called_once_with("msg-6")
        freescout.create_conversation.assert_called_once()
        freescout.add_customer_thread.assert_not_called()
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-6", "thread-6", "boom")

    def test_create_failure_without_conversation_id_skips_thread_map_write(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conversation_id_for_thread.return_value = None
        freescout = Mock()
        freescout.create_conversation.return_value = {}
        message = _make_message("msg-10", "thread-10")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "get_settings", return_value=_base_settings()):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "failed_retryable")
        self.assertEqual(result.reason, "ticket creation failed")
        store.processed_success.assert_called_once_with("msg-10")
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_not_called()

if __name__ == "__main__":
    unittest.main()
