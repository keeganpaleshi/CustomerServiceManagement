import unittest
from unittest.mock import Mock, call, patch

import requests

import gmail_bot
from gmail_bot import ProcessResult


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


def _make_service(message: dict) -> Mock:
    service = Mock()
    service.users.return_value.messages.return_value.get.return_value.execute.return_value = (
        message
    )
    return service


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
        service = _make_service(message)

        result = gmail_bot.process_gmail_message(
            {"id": "msg-1", "threadId": "thread-1"},
            store,
            freescout,
            service,
            None,
            5,
        )

        self.assertEqual(result, ProcessResult.SKIPPED_ALREADY_SUCCESS)
        service.users.assert_not_called()
        store.get_conv_id.assert_not_called()
        freescout.method_calls.assertEqual([])

    def test_processed_filtered_returns_filtered_without_freescout(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = True
        freescout = Mock()
        message = _make_message("msg-2", "thread-2")
        service = _make_service(message)

        result = gmail_bot.process_gmail_message(
            {"id": "msg-2", "threadId": "thread-2"},
            store,
            freescout,
            service,
            None,
            5,
        )

        self.assertEqual(result, ProcessResult.FILTERED)
        service.users.assert_not_called()
        freescout.method_calls.assertEqual([])

    def test_filtered_terminal_marks_filtered(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        freescout = Mock()
        message = _make_message("msg-2", "thread-2")
        service = _make_service(message)

        with patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=True):
            result = gmail_bot.process_gmail_message(
                {"id": "msg-2", "threadId": "thread-2"},
                store,
                freescout,
                service,
                None,
                5,
            )

        self.assertEqual(result, ProcessResult.FILTERED)
        store.mark_filtered.assert_called_once_with(
            "msg-2",
            "thread-2",
            reason="filtered: promotional/spam",
        )
        store.mark_success.assert_not_called()
        store.mark_failed.assert_not_called()
        freescout.method_calls.assertEqual([])

    def test_append_existing_thread_marks_success_after_append(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conv_id.return_value = "conv-123"
        freescout = Mock()
        message = _make_message("msg-3", "thread-3")
        service = _make_service(message)

        with patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(
                {"id": "msg-3", "threadId": "thread-3"},
                store,
                freescout,
                service,
                None,
                5,
            )

        self.assertEqual(result, ProcessResult.FREESCOUT_APPENDED)
        freescout.method_calls.assertEqual(
            [call.add_customer_thread("conv-123", "hello", imported=True)]
        )
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_called_once_with("msg-3", "thread-3", "conv-123")

    def test_create_new_thread_marks_success_after_upsert(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conv_id.return_value = None
        freescout = Mock()
        freescout.create_conversation.return_value = {"id": "conv-456"}
        message = _make_message("msg-4", "thread-4")
        service = _make_service(message)

        with patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "get_settings", return_value=_base_settings()):
            result = gmail_bot.process_gmail_message(
                {"id": "msg-4", "threadId": "thread-4"},
                store,
                freescout,
                service,
                None,
                5,
            )

        self.assertEqual(result, ProcessResult.FREESCOUT_CREATED)
        freescout.method_calls.assertEqual(
            [
                call.create_conversation(
                    "Need help",
                    "customer@example.com",
                    "hello",
                    "mailbox-1",
                    thread_id="thread-4",
                    message_id="msg-4",
                    gmail_thread_field="field-thread",
                    gmail_message_field="field-message",
                )
            ]
        )
        store.upsert_thread_map.assert_called_once_with("thread-4", "conv-456")
        store.mark_success.assert_called_once_with("msg-4", "thread-4", "conv-456")

    def test_append_failure_marks_failed(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conv_id.return_value = "conv-789"
        freescout = Mock()
        freescout.add_customer_thread.side_effect = requests.RequestException("boom")
        message = _make_message("msg-5", "thread-5")
        service = _make_service(message)

        with patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False):
            result = gmail_bot.process_gmail_message(
                {"id": "msg-5", "threadId": "thread-5"},
                store,
                freescout,
                service,
                None,
                5,
            )

        self.assertEqual(result, ProcessResult.FREESCOUT_FAILED)
        freescout.method_calls.assertEqual(
            [call.add_customer_thread("conv-789", "hello", imported=True)]
        )
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-5", "thread-5", "boom", "conv-789")

    def test_create_failure_marks_failed(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conv_id.return_value = None
        freescout = Mock()
        freescout.create_conversation.side_effect = requests.RequestException("boom")
        message = _make_message("msg-6", "thread-6")
        service = _make_service(message)

        with patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "get_settings", return_value=_base_settings()):
            result = gmail_bot.process_gmail_message(
                {"id": "msg-6", "threadId": "thread-6"},
                store,
                freescout,
                service,
                None,
                5,
            )

        self.assertEqual(result, ProcessResult.FREESCOUT_FAILED)
        freescout.method_calls.assertEqual(
            [
                call.create_conversation(
                    "Need help",
                    "customer@example.com",
                    "hello",
                    "mailbox-1",
                    thread_id="thread-6",
                    message_id="msg-6",
                    gmail_thread_field="field-thread",
                    gmail_message_field="field-message",
                )
            ]
        )
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with("msg-6", "thread-6", "boom")

    def test_create_failure_without_conversation_id_does_not_upsert_thread(self):
        store = Mock()
        store.processed_success.return_value = False
        store.processed_filtered.return_value = False
        store.get_conv_id.return_value = None
        freescout = Mock()
        freescout.create_conversation.return_value = {"id": None}
        message = _make_message("msg-7", "thread-7")
        service = _make_service(message)

        with patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "get_settings", return_value=_base_settings()):
            result = gmail_bot.process_gmail_message(
                {"id": "msg-7", "threadId": "thread-7"},
                store,
                freescout,
                service,
                None,
                5,
            )

        self.assertEqual(result, ProcessResult.FREESCOUT_FAILED)
        freescout.method_calls.assertEqual(
            [
                call.create_conversation(
                    "Need help",
                    "customer@example.com",
                    "hello",
                    "mailbox-1",
                    thread_id="thread-7",
                    message_id="msg-7",
                    gmail_thread_field="field-thread",
                    gmail_message_field="field-message",
                )
            ]
        )
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once_with(
            "msg-7", "thread-7", "ticket creation failed", None
        )


if __name__ == "__main__":
    unittest.main()
