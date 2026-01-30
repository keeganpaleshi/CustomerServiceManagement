"""Unit tests for webhook validation functions in webhook_server.py."""

import time
import threading
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# Import the module to access its internal functions
import webhook_server
from webhook_server import (
    _validate_webhook_timestamp,
    _check_and_record_nonce,
    _nonce_cache,
    _nonce_cache_lock,
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
        # Clear the nonce cache before each test
        with _nonce_cache_lock:
            _nonce_cache.clear()

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
        # Fill the cache with many nonces
        for i in range(webhook_server.NONCE_CACHE_SIZE + 100):
            _check_and_record_nonce(f"nonce-{i}")

        with _nonce_cache_lock:
            # Cache should not exceed NONCE_CACHE_SIZE
            self.assertLessEqual(len(_nonce_cache), webhook_server.NONCE_CACHE_SIZE)

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
        for _ in range(4):
            self.assertIsInstance(current, dict)
            current = current["level"]

        self.assertEqual(current, "[TRUNCATED]")


if __name__ == "__main__":
    unittest.main()
