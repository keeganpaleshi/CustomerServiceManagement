import argparse
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


def _args():
    return argparse.Namespace(
        poll_freescout=False,
        poll_interval=300,
        timeout=5,
        gmail_query="is:unread",
        console_auth=False,
    )


class GmailIngestionTests(unittest.TestCase):
    def _service_with_message(self, message: dict) -> Mock:
        service = Mock()
        service.users.return_value.messages.return_value.get.return_value.execute.return_value = (
            message
        )
        service.users.return_value.threads.return_value.modify.return_value.execute.return_value = {}
        return service

    def test_skip_already_processed_skips_freescout(self):
        store = Mock()
        store.processed_success.return_value = True
        client = Mock()

        with patch.object(gmail_bot, "get_settings", return_value=_base_settings()), \
            patch.object(gmail_bot, "parse_args", return_value=_args()), \
            patch.object(gmail_bot, "TicketStore", return_value=store), \
            patch.object(gmail_bot, "fetch_all_unread_messages", return_value=[{"id": "msg-1"}]), \
            patch.object(gmail_bot, "get_gmail_service", return_value=Mock()), \
            patch.object(gmail_bot, "_build_freescout_client", return_value=client), \
            patch.object(gmail_bot, "poll_ticket_updates", return_value=[]), \
            patch.object(gmail_bot, "create_ticket") as create_ticket:
            gmail_bot.main()

        create_ticket.assert_not_called()
        client.add_customer_thread.assert_not_called()

    def test_filtered_terminal_marks_filtered(self):
        store = Mock()
        store.processed_success.return_value = False
        store.get_conv_id.return_value = None
        client = Mock()
        message = _make_message("msg-2", "thread-2")

        with patch.object(gmail_bot, "get_settings", return_value=_base_settings()), \
            patch.object(gmail_bot, "parse_args", return_value=_args()), \
            patch.object(gmail_bot, "TicketStore", return_value=store), \
            patch.object(gmail_bot, "fetch_all_unread_messages", return_value=[{"id": "msg-2"}]), \
            patch.object(gmail_bot, "get_gmail_service", return_value=self._service_with_message(message)), \
            patch.object(gmail_bot, "_build_freescout_client", return_value=client), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=True), \
            patch.object(gmail_bot, "poll_ticket_updates", return_value=[]), \
            patch.object(gmail_bot, "create_ticket") as create_ticket:
            gmail_bot.main()

        store.mark_filtered.assert_called_once_with(
            "msg-2",
            "thread-2",
            reason="filtered: promotional/spam",
        )
        store.mark_success.assert_not_called()
        create_ticket.assert_not_called()
        client.add_customer_thread.assert_not_called()

    def test_append_existing_thread_marks_success_after_append(self):
        store = Mock()
        store.processed_success.return_value = False
        store.get_conv_id.return_value = "conv-123"
        client = Mock()
        message = _make_message("msg-3", "thread-3")

        controller = Mock()
        controller.attach_mock(client, "client")
        controller.attach_mock(store, "store")

        with patch.object(gmail_bot, "get_settings", return_value=_base_settings()), \
            patch.object(gmail_bot, "parse_args", return_value=_args()), \
            patch.object(gmail_bot, "TicketStore", return_value=store), \
            patch.object(gmail_bot, "fetch_all_unread_messages", return_value=[{"id": "msg-3"}]), \
            patch.object(gmail_bot, "get_gmail_service", return_value=self._service_with_message(message)), \
            patch.object(gmail_bot, "_build_freescout_client", return_value=client), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "poll_ticket_updates", return_value=[]):
            gmail_bot.main()

        client.add_customer_thread.assert_called_once_with("conv-123", "hello", imported=True)
        store.mark_success.assert_called_once_with("msg-3", "thread-3", "conv-123")
        self.assertLess(
            controller.mock_calls.index(
                unittest.mock.call.client.add_customer_thread("conv-123", "hello", imported=True)
            ),
            controller.mock_calls.index(
                unittest.mock.call.store.mark_success("msg-3", "thread-3", "conv-123")
            ),
        )

    def test_create_new_thread_marks_success_after_upsert(self):
        store = Mock()
        store.processed_success.return_value = False
        store.get_conv_id.return_value = None
        client = Mock()
        message = _make_message("msg-4", "thread-4")

        controller = Mock()
        controller.attach_mock(store, "store")

        with patch.object(gmail_bot, "get_settings", return_value=_base_settings()), \
            patch.object(gmail_bot, "parse_args", return_value=_args()), \
            patch.object(gmail_bot, "TicketStore", return_value=store), \
            patch.object(gmail_bot, "fetch_all_unread_messages", return_value=[{"id": "msg-4"}]), \
            patch.object(gmail_bot, "get_gmail_service", return_value=self._service_with_message(message)), \
            patch.object(gmail_bot, "_build_freescout_client", return_value=client), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "poll_ticket_updates", return_value=[]), \
            patch.object(gmail_bot, "create_ticket", return_value={"id": "conv-456"}) as create_ticket:
            gmail_bot.main()

        create_ticket.assert_called_once()
        store.upsert_thread_map.assert_called_once_with("thread-4", "conv-456")
        store.mark_success.assert_called_once_with("msg-4", "thread-4", "conv-456")
        self.assertLess(
            controller.mock_calls.index(
                unittest.mock.call.store.upsert_thread_map("thread-4", "conv-456")
            ),
            controller.mock_calls.index(
                unittest.mock.call.store.mark_success("msg-4", "thread-4", "conv-456")
            ),
        )

    def test_append_failure_skips_mark_success(self):
        store = Mock()
        store.processed_success.return_value = False
        store.get_conv_id.return_value = "conv-789"
        client = Mock()
        client.add_customer_thread.side_effect = requests.RequestException("boom")
        message = _make_message("msg-5", "thread-5")

        with patch.object(gmail_bot, "get_settings", return_value=_base_settings()), \
            patch.object(gmail_bot, "parse_args", return_value=_args()), \
            patch.object(gmail_bot, "TicketStore", return_value=store), \
            patch.object(gmail_bot, "fetch_all_unread_messages", return_value=[{"id": "msg-5"}]), \
            patch.object(gmail_bot, "get_gmail_service", return_value=self._service_with_message(message)), \
            patch.object(gmail_bot, "_build_freescout_client", return_value=client), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "poll_ticket_updates", return_value=[]):
            gmail_bot.main()

        client.add_customer_thread.assert_called_once_with("conv-789", "hello", imported=True)
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once()

    def test_create_failure_skips_thread_map_write(self):
        store = Mock()
        store.processed_success.return_value = False
        store.get_conv_id.return_value = None
        client = Mock()
        message = _make_message("msg-6", "thread-6")

        with patch.object(gmail_bot, "get_settings", return_value=_base_settings()), \
            patch.object(gmail_bot, "parse_args", return_value=_args()), \
            patch.object(gmail_bot, "TicketStore", return_value=store), \
            patch.object(gmail_bot, "fetch_all_unread_messages", return_value=[{"id": "msg-6"}]), \
            patch.object(gmail_bot, "get_gmail_service", return_value=self._service_with_message(message)), \
            patch.object(gmail_bot, "_build_freescout_client", return_value=client), \
            patch.object(gmail_bot, "extract_plain_text", return_value="hello"), \
            patch.object(gmail_bot, "is_promotional_or_spam", return_value=False), \
            patch.object(gmail_bot, "poll_ticket_updates", return_value=[]), \
            patch.object(gmail_bot, "create_ticket", return_value=None) as create_ticket:
            gmail_bot.main()

        create_ticket.assert_called_once()
        store.upsert_thread_map.assert_not_called()
        store.mark_success.assert_not_called()
        store.mark_failed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
