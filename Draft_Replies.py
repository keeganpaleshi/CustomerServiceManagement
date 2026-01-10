import argparse
from typing import Dict, Optional

from openai import OpenAI

from utils import (
    classify_email,
    create_base64_message,
    create_draft,
    create_ticket,
    critic_email,
    ensure_label,
    fetch_all_unread_messages,
    apply_label_to_thread,
    get_gmail_service,
    get_settings,
    is_promotional_or_spam,
    extract_plain_text,
    require_openai_api_key,
)


def parse_args(settings: Optional[Dict] = None):
    settings = settings or get_settings()
    parser = argparse.ArgumentParser(description="Draft Gmail replies")
    parser.add_argument("--gmail-query", default=settings["GMAIL_QUERY"],
                        help="Gmail search query")
    parser.add_argument("--timeout", type=int,
                        default=settings["HTTP_TIMEOUT"], help="HTTP request timeout")
    parser.add_argument(
        "--console-auth",
        action="store_true",
        default=settings["GMAIL_USE_CONSOLE"],
        help="Use console-based OAuth (paste auth code) instead of opening a browser",
    )
    parser.add_argument(
        "--skip-existing-drafts",
        action="store_true",
        default=False,
        help=(
            "Deprecated: Gmail-label-based draft skipping is disabled. "
            "Use DB-only idempotency for Phase 2C flows."
        ),
    )
    return parser.parse_args()


# -------------------------------------------------------
def get_header_value(message, header_name):
    """
    Extract a specific header (e.g., 'Subject', 'From') from a message detail.
    """
    headers = message.get("payload", {}).get("headers", [])
    for header in headers:
        if header.get("name", "").lower() == header_name.lower():
            return header.get("value", "")
    return ""


"""OpenAI Integration"""


# -------------------------------------------------------
def generate_ai_reply(subject, sender, snippet_or_body, email_type):
    """
    Generate a draft reply using OpenAI's new library (>=1.0.0).
    """
    settings = get_settings()
    # Create a client instance with your API key
    client = OpenAI(api_key=require_openai_api_key())

    # Example prompt; tailor as needed
    instructions = (
        f"[Email type: {email_type}]\n\n"
        "You are an AI email assistant. The user received an email.\n"
        f"Subject: {subject}\n"
        f"From: {sender}\n"
        f"Email content/snippet: {snippet_or_body}\n\n"
        "Please write a friendly and professional draft reply addressing the sender's query.\n"
        "Return only the reply text that can be sent to the customer. "
        "Do not include analysis, classification labels, or meta commentary."
    )


if __name__ == "__main__":
    main()
