"""Unit tests for utils.py functions."""

import threading
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from utils import (
    OpenAICostTracker,
    SimpleRateLimiter,
    validate_conversation_id,
    sanitize_draft_reply,
)


class TestValidateConversationId(unittest.TestCase):
    """Tests for validate_conversation_id function."""

    def test_valid_alphanumeric_id(self):
        is_valid, id_str, error = validate_conversation_id("abc123")
        self.assertTrue(is_valid)
        self.assertEqual(id_str, "abc123")
        self.assertEqual(error, "")

    def test_valid_id_with_hyphens_underscores(self):
        is_valid, id_str, error = validate_conversation_id("conv-123_test")
        self.assertTrue(is_valid)
        self.assertEqual(id_str, "conv-123_test")

    def test_none_value(self):
        is_valid, id_str, error = validate_conversation_id(None)
        self.assertFalse(is_valid)
        self.assertIsNone(id_str)
        self.assertEqual(error, "missing conversation id")

    def test_empty_string(self):
        is_valid, id_str, error = validate_conversation_id("")
        self.assertFalse(is_valid)
        self.assertIsNone(id_str)
        self.assertEqual(error, "conversation id is empty")

    def test_whitespace_only(self):
        is_valid, id_str, error = validate_conversation_id("   ")
        self.assertFalse(is_valid)
        self.assertIsNone(id_str)
        self.assertEqual(error, "conversation id is empty")

    def test_strips_whitespace(self):
        is_valid, id_str, error = validate_conversation_id("  abc123  ")
        self.assertTrue(is_valid)
        self.assertEqual(id_str, "abc123")

    def test_invalid_characters(self):
        is_valid, id_str, error = validate_conversation_id("abc@123")
        self.assertFalse(is_valid)
        self.assertIsNone(id_str)
        self.assertIn("invalid characters", error)

    def test_special_characters_rejected(self):
        test_cases = [
            "abc/123",
            "abc\\123",
            "abc<script>",
            "abc;DROP TABLE",
            "abc\n123",
            "abc 123",
        ]
        for test_id in test_cases:
            is_valid, id_str, error = validate_conversation_id(test_id)
            self.assertFalse(is_valid, f"Should reject: {test_id!r}")
            self.assertIn("invalid characters", error)

    def test_exceeds_max_length(self):
        long_id = "a" * 300
        is_valid, id_str, error = validate_conversation_id(long_id)
        self.assertFalse(is_valid)
        self.assertIsNone(id_str)
        self.assertIn("exceeds maximum length", error)

    def test_at_max_length(self):
        max_id = "a" * 256
        is_valid, id_str, error = validate_conversation_id(max_id)
        self.assertTrue(is_valid)
        self.assertEqual(id_str, max_id)

    def test_integer_conversion(self):
        is_valid, id_str, error = validate_conversation_id(12345)
        self.assertTrue(is_valid)
        self.assertEqual(id_str, "12345")


class TestOpenAICostTracker(unittest.TestCase):
    """Tests for OpenAICostTracker class."""

    def test_estimate_cost_known_model(self):
        tracker = OpenAICostTracker()
        # gpt-4o: input=0.005, output=0.015 per 1K tokens
        cost = tracker.estimate_cost("gpt-4o", 1000, 500)
        expected = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
        self.assertAlmostEqual(cost, expected, places=6)

    def test_estimate_cost_unknown_model(self):
        tracker = OpenAICostTracker()
        # Should use default pricing
        cost = tracker.estimate_cost("unknown-model", 1000, 500)
        self.assertGreater(cost, 0)

    def test_record_usage_within_budget(self):
        tracker = OpenAICostTracker(daily_budget_usd=10.0, monthly_budget_usd=100.0)
        result = tracker.record_usage("gpt-4o", 100, 100)
        self.assertTrue(result)

    def test_record_usage_exceeds_daily_budget(self):
        tracker = OpenAICostTracker(daily_budget_usd=0.001, monthly_budget_usd=100.0)
        # First call may succeed
        tracker.record_usage("gpt-4o", 10000, 10000)
        # Second call should exceed budget
        result = tracker.record_usage("gpt-4o", 10000, 10000)
        self.assertFalse(result)

    def test_record_usage_exceeds_monthly_budget(self):
        tracker = OpenAICostTracker(daily_budget_usd=100.0, monthly_budget_usd=0.001)
        # First call may succeed
        tracker.record_usage("gpt-4o", 10000, 10000)
        # Second call should exceed budget
        result = tracker.record_usage("gpt-4o", 10000, 10000)
        self.assertFalse(result)

    def test_can_make_request_within_budget(self):
        tracker = OpenAICostTracker(daily_budget_usd=10.0, monthly_budget_usd=100.0)
        result = tracker.can_make_request("gpt-4o", 100, 100)
        self.assertTrue(result)

    def test_can_make_request_exceeds_budget(self):
        tracker = OpenAICostTracker(daily_budget_usd=0.001, monthly_budget_usd=100.0)
        # Exhaust budget first
        tracker.record_usage("gpt-4o", 10000, 10000)
        # Check if new request would exceed
        result = tracker.can_make_request("gpt-4o", 10000, 10000)
        self.assertFalse(result)

    def test_unlimited_budget(self):
        tracker = OpenAICostTracker(daily_budget_usd=0.0, monthly_budget_usd=0.0)
        # Should always succeed with unlimited budget
        for _ in range(10):
            result = tracker.record_usage("gpt-4o", 100000, 100000)
            self.assertTrue(result)

    def test_get_usage_stats(self):
        tracker = OpenAICostTracker(daily_budget_usd=10.0, monthly_budget_usd=100.0)
        tracker.record_usage("gpt-4o", 1000, 500)
        stats = tracker.get_usage_stats()
        self.assertEqual(stats["total_calls"], 1)
        self.assertGreater(stats["daily_cost"], 0)
        self.assertEqual(stats["daily_budget"], 10.0)
        self.assertEqual(stats["monthly_budget"], 100.0)

    def test_thread_safety(self):
        """Test that cost tracker is thread-safe."""
        tracker = OpenAICostTracker(daily_budget_usd=1000.0, monthly_budget_usd=10000.0)
        num_threads = 10
        calls_per_thread = 100

        def record_usage():
            for _ in range(calls_per_thread):
                tracker.record_usage("gpt-4o", 10, 10)

        threads = [threading.Thread(target=record_usage) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = tracker.get_usage_stats()
        self.assertEqual(stats["total_calls"], num_threads * calls_per_thread)


class TestSimpleRateLimiter(unittest.TestCase):
    """Tests for SimpleRateLimiter class."""

    def test_allows_within_limit(self):
        limiter = SimpleRateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            self.assertTrue(limiter.allow())

    def test_blocks_over_limit(self):
        limiter = SimpleRateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            self.assertTrue(limiter.allow())
        self.assertFalse(limiter.allow())

    def test_disabled_with_zero_max_requests(self):
        limiter = SimpleRateLimiter(max_requests=0, window_seconds=60)
        for _ in range(100):
            self.assertTrue(limiter.allow())

    def test_disabled_with_zero_window(self):
        limiter = SimpleRateLimiter(max_requests=5, window_seconds=0)
        for _ in range(100):
            self.assertTrue(limiter.allow())

    def test_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        limiter = SimpleRateLimiter(max_requests=100, window_seconds=60)
        allowed_count = [0]
        lock = threading.Lock()

        def try_request():
            for _ in range(20):
                if limiter.allow():
                    with lock:
                        allowed_count[0] += 1

        threads = [threading.Thread(target=try_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have allowed exactly 100 requests
        self.assertEqual(allowed_count[0], 100)


class TestSanitizeDraftReply(unittest.TestCase):
    """Tests for sanitize_draft_reply function."""

    def test_empty_input(self):
        self.assertEqual(sanitize_draft_reply(""), "")
        self.assertEqual(sanitize_draft_reply(None), "")

    def test_strips_whitespace(self):
        self.assertEqual(sanitize_draft_reply("  Hello  "), "Hello")

    def test_removes_analysis_tags(self):
        text = "<analysis>Some analysis</analysis>Hello!"
        self.assertEqual(sanitize_draft_reply(text), "Hello!")

    def test_removes_final_label(self):
        text = "Some reasoning. Final: Hello!"
        self.assertEqual(sanitize_draft_reply(text), "Hello!")

    def test_removes_reply_label(self):
        text = "Some reasoning. Reply: Hello!"
        self.assertEqual(sanitize_draft_reply(text), "Hello!")

    def test_removes_reasoning_sections(self):
        text = "Reasoning: This is reasoning\n\nHello!"
        result = sanitize_draft_reply(text)
        self.assertIn("Hello!", result)
        self.assertNotIn("This is reasoning", result)


if __name__ == "__main__":
    unittest.main()
