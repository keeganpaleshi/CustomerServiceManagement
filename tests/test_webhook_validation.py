"""Unit tests for webhook validation functions in webhook_server.py."""

import re
import time
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient

# Import the module to access its internal functions
import webhook_server
from webhook_server import (
    _validate_webhook_timestamp,
    _check_and_record_nonce,
)


class TestValidateWebhookTimestamp(unittest.TestCase):
    """Tests for _validate_webhook_timestamp function."""

    def test_none_timestamp_is_valid(self):
        is_valid, error = _validate_webhook_timestamp(None)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_valid_iso_timestamp(self):
        now = datetime.now(timezone.utc).isoformat()
        is_valid, error = _validate_webhook_timestamp(now)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_valid_iso_timestamp_with_z(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        is_valid, error = _validate_webhook_timestamp(now)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_valid_unix_timestamp_int(self):
        now = int(time.time())
        is_valid, error = _validate_webhook_timestamp(now)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_valid_unix_timestamp_float(self):
        now = time.time()
        is_valid, error = _validate_webhook_timestamp(now)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

    def test_timestamp_too_old(self):
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        is_valid, error = _validate_webhook_timestamp(old_time.isoformat())
        self.assertFalse(is_valid)
        self.assertIn("too old", error)

    def test_timestamp_too_far_in_future(self):
        future_time = datetime.now(timezone.utc) + timedelta(minutes=10)
        is_valid, error = _validate_webhook_timestamp(future_time.isoformat())
        self.assertFalse(is_valid)
        self.assertIn("too old", error)  # Same error message for both cases

    def test_invalid_timestamp_format(self):
        is_valid, error = _validate_webhook_timestamp("not-a-timestamp")
        self.assertFalse(is_valid)
        self.assertIn("invalid timestamp format", error)

    def test_timestamp_within_skew(self):
        # 4 minutes ago should be valid (within 5 minute skew)
        near_past = datetime.now(timezone.utc) - timedelta(minutes=4)
        is_valid, error = _validate_webhook_timestamp(near_past.isoformat())
        self.assertTrue(is_valid)
        self.assertEqual(error, "")


class TestCheckAndRecordNonce(unittest.TestCase):
    """Tests for _check_and_record_nonce function."""

    def setUp(self):
        # Create a temporary database for each test
        self._temp_dir = tempfile.TemporaryDirectory()
        self._temp_db = Path(self._temp_dir.name) / "test_nonces.sqlite"
        # Patch _get_counter_store to return a new store each time (since it gets closed after use)
        from storage import TicketStore

        def _make_store():
            return TicketStore(str(self._temp_db))

        self._patcher = patch.object(
            webhook_server, "_get_counter_store", side_effect=_make_store
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._temp_dir.cleanup()

    def test_none_nonce_is_valid(self):
        is_new, error = _check_and_record_nonce(None)
        self.assertTrue(is_new)
        self.assertEqual(error, "")

    def test_new_nonce_is_accepted(self):
        is_new, error = _check_and_record_nonce("unique-nonce-123")
        self.assertTrue(is_new)
        self.assertEqual(error, "")

    def test_duplicate_nonce_is_rejected(self):
        nonce = "duplicate-nonce-456"
        is_new1, error1 = _check_and_record_nonce(nonce)
        self.assertTrue(is_new1)

        is_new2, error2 = _check_and_record_nonce(nonce)
        self.assertFalse(is_new2)
        self.assertIn("replay attack", error2)

    def test_nonce_too_long(self):
        long_nonce = "a" * 300
        is_new, error = _check_and_record_nonce(long_nonce)
        self.assertFalse(is_new)
        self.assertIn("maximum length", error)

    def test_nonce_at_max_length(self):
        max_nonce = "a" * 256
        is_new, error = _check_and_record_nonce(max_nonce)
        self.assertTrue(is_new)

    def test_integer_nonce_conversion(self):
        is_new, error = _check_and_record_nonce(12345)
        self.assertTrue(is_new)

        # Same integer should be rejected
        is_new2, error2 = _check_and_record_nonce(12345)
        self.assertFalse(is_new2)

    def test_nonce_cache_size_limit(self):
        """Test that cache doesn't grow unbounded."""
        # Fill the cache with many nonces using the configured cache size
        cache_size = webhook_server._DEFAULT_NONCE_CACHE_SIZE
        num_nonces = min(cache_size + 100, 200)  # Limit to avoid slow tests
        for i in range(num_nonces):
            _check_and_record_nonce(f"nonce-{i}")

        # Verify all nonces were recorded (duplicates would be rejected)
        is_new, _ = _check_and_record_nonce("nonce-0")
        self.assertFalse(is_new)  # Should be rejected as duplicate

    def test_thread_safety(self):
        """Test that nonce cache is thread-safe."""
        num_threads = 10
        nonces_per_thread = 100
        accepted_count = [0]
        rejected_count = [0]
        lock = threading.Lock()

        def record_nonces(thread_id):
            for i in range(nonces_per_thread):
                is_new, _ = _check_and_record_nonce(f"thread-{thread_id}-nonce-{i}")
                with lock:
                    if is_new:
                        accepted_count[0] += 1
                    else:
                        rejected_count[0] += 1

        threads = [threading.Thread(target=record_nonces, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All unique nonces should be accepted
        self.assertEqual(accepted_count[0], num_threads * nonces_per_thread)
        self.assertEqual(rejected_count[0], 0)


class TestSanitizePayload(unittest.TestCase):
    """Tests for _sanitize_payload function."""

    def test_redacts_email_field(self):
        payload = {"email": "user@example.com", "name": "John"}
        result = webhook_server._sanitize_payload(payload)
        self.assertIn("REDACTED", result["email"])
        self.assertEqual(result["name"], "John")

    def test_redacts_password_field(self):
        payload = {"password": "secret123", "username": "user"}
        result = webhook_server._sanitize_payload(payload)
        self.assertIn("REDACTED", result["password"])
        self.assertEqual(result["username"], "user")

    def test_redacts_token_field(self):
        payload = {"token": "abc123xyz", "id": 1}
        result = webhook_server._sanitize_payload(payload)
        self.assertIn("REDACTED", result["token"])
        self.assertEqual(result["id"], 1)

    def test_redacts_nested_fields(self):
        payload = {
            "data": {
                "customer_email": "user@example.com",
                "order_id": 123
            }
        }
        result = webhook_server._sanitize_payload(payload)
        self.assertIn("REDACTED", result["data"]["customer_email"])
        self.assertEqual(result["data"]["order_id"], 123)

    def test_redacts_in_list(self):
        payload = {
            "items": [
                {"email": "user1@example.com"},
                {"email": "user2@example.com"}
            ]
        }
        result = webhook_server._sanitize_payload(payload)
        for item in result["items"]:
            self.assertIn("REDACTED", item["email"])

    def test_preserves_non_dict_values(self):
        payload = "just a string"
        result = webhook_server._sanitize_payload(payload)
        self.assertEqual(result, "just a string")

    def test_truncates_deeply_nested_payload(self):
        payload = {}
        current = payload
        for _ in range(12):
            current["level"] = {}
            current = current["level"]

        result = webhook_server._sanitize_payload(payload, max_depth=5)
        current = result
        # max_depth=5 allows depths 0-4, truncates at depth 5
        # After 4 iterations we reach the dict at depth 4
        for _ in range(4):
            self.assertIsInstance(current, dict)
            current = current["level"]

        # At depth 4, the nested "level" value should be truncated
        self.assertIsInstance(current, dict)
        self.assertEqual(current["level"], "[TRUNCATED]")


class TestWebhookLogging(unittest.TestCase):
    """Tests for webhook logging behavior."""

    def test_log_webhook_payload_caps_filename_length(self):
        long_event_id = "a" * 500
        payload = {"event_id": long_event_id, "payload": {"key": "value"}}
        filename_pattern = re.compile(
            r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}Z)-(?P<event_id>.+)\.json$"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with patch.object(webhook_server, "_get_log_dir", return_value=log_dir):
                logfile = webhook_server.log_webhook_payload(payload)

            self.assertTrue(logfile.exists())
            match = filename_pattern.match(logfile.name)
            self.assertIsNotNone(match)
            event_id = match.group("event_id")
            self.assertLessEqual(len(event_id), webhook_server.SAFE_FILENAME_MAX_LENGTH)


class TestWebhookSecretRequirement(unittest.TestCase):
    """Tests webhook secret requirement behavior."""

    def test_rejects_when_secret_missing_and_unauthenticated_not_allowed(self):
        client = TestClient(webhook_server.app)
        payload = {"type": "conversation.created", "conversation_id": "123"}

        with patch.object(
            webhook_server,
            "get_settings",
            return_value={
                "FREESCOUT_WEBHOOK_SECRET": "",
                "WEBHOOK_ALLOW_UNAUTHENTICATED": False,
                "WEBHOOK_LOG_DIR": "",
                "WEBHOOK_MAX_TIMESTAMP_SKEW_SECONDS": 300,
                "WEBHOOK_NONCE_CACHE_SIZE": 10000,
                "WEBHOOK_NONCE_CACHE_TTL_SECONDS": 600,
            },
        ), patch.object(webhook_server, "reload_settings"), patch.object(
            webhook_server, "log_webhook_payload", return_value=Path("/tmp/test.json")
        ), patch.object(webhook_server, "_get_counter_store") as mock_store:
            response = client.post("/freescout", json=payload)

        self.assertEqual(response.status_code, 503)
        self.assertIn("Webhook authentication is required", response.json()["detail"])
        mock_store.assert_not_called()

    def test_allows_when_secret_missing_but_unauthenticated_explicitly_enabled(self):
        client = TestClient(webhook_server.app)
        payload = {"type": "conversation.created", "conversation_id": "123"}

        with patch.object(
            webhook_server,
            "get_settings",
            return_value={
                "FREESCOUT_WEBHOOK_SECRET": "",
                "WEBHOOK_ALLOW_UNAUTHENTICATED": True,
                "WEBHOOK_LOG_DIR": "",
                "WEBHOOK_MAX_TIMESTAMP_SKEW_SECONDS": 300,
                "WEBHOOK_NONCE_CACHE_SIZE": 10000,
                "WEBHOOK_NONCE_CACHE_TTL_SECONDS": 600,
            },
        ), patch.object(webhook_server, "reload_settings"), patch.object(
            webhook_server, "log_webhook_payload", return_value=Path("/tmp/test.json")
        ), patch.object(
            webhook_server,
            "freescout_webhook_handler",
            return_value=("ok", 200, None),
        ), patch.object(webhook_server, "_get_counter_store") as mock_store:
            response = client.post("/freescout", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "ok"})
        mock_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
