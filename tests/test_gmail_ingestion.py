import unittest
from unittest.mock import Mock, patch

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
        store.processed_terminal.return_value = True
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
        store.processed_terminal.return_value = False
        freescout = Mock()
        message = _make_message("msg-2", "thread-2")

        with patch.object(gmail_bot, "_TICKET_LABEL_ID", None), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=True):
            result = gmail_bot.process_gmail_message(message, store, freescout, Mock())

        self.assertEqual(result.status, "filtered")
        self.assertEqual(result.reason, "filtered: promotional/spam")
        self.assertIsNone(result.freescout_conversation_id)
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
        store.processed_terminal.return_value = False
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
        freescout.add_customer_thread.assert_called_once_with("conv-123", "hello", imported=True)
        freescout.create_conversation.assert_not_called()
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_called_once_with("msg-3", "thread-3", "conv-123")

    def test_create_new_thread_marks_success_after_upsert(self):
        store = Mock()
        store.processed_terminal.return_value = False
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
        store.mark_success.assert_called_once_with("msg-4", "thread-4", "conv-456")

    def test_append_failure_marks_failed(self):
        store = Mock()
        store.processed_terminal.return_value = False
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
        freescout.add_customer_thread.assert_called_once_with("conv-789", "hello", imported=True)
        freescout.create_conversation.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-5", "thread-5", "boom", "conv-789")

    def test_create_failure_marks_failed(self):
        store = Mock()
        store.processed_terminal.return_value = False
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
        freescout.create_conversation.assert_called_once()
        freescout.add_customer_thread.assert_not_called()
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-6", "thread-6", "boom")


if __name__ == "__main__":
    unittest.main()
