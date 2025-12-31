import argparse
from typing import Dict, Optional

from openai import OpenAI

from utils import (
    classify_email,
    create_ticket,
    fetch_all_unread_messages,
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
        "Please write a friendly and professional draft reply addressing the sender's query."
    )
    try:
        response = client.chat.completions.create(
            model=settings["DRAFT_MODEL"],
            messages=[
                {"role": "system", "content": settings["DRAFT_SYSTEM_MSG"]},
                {"role": "user", "content": instructions},
            ],
            max_tokens=settings["DRAFT_MAX_TOKENS"],
            temperature=0.7,
        )
        # Extract the actual reply text
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        fallback_lines = [
            "Hello,",
            "",
            "I'm sorry, but I couldn't generate a response at this time. Please review this email manually.",
            "",
            "Best,",
            "Automated Script",
        ]
        return "\n".join(fallback_lines)


# -------------------------------------------------------
# 6) Main Flow
# -------------------------------------------------------
def main():
    settings = get_settings()
    args = parse_args(settings)
    """
    1) Authenticate & build Gmail service.
    2) Fetch unread messages & limit to the configured amount.
    3) For each message:
       - Skip if there's already a draft in the same thread.
       - Generate AI-based draft.
       - Create the draft (email remains unread).
    """
    service = get_gmail_service(use_console=args.console_auth)

    # Fetch all unread, then limit for "most recent" processing
    unread_messages = fetch_all_unread_messages(
        service, query=args.gmail_query)
    if not unread_messages:
        print("No unread messages found.")
        return

    # We assume the API returns them from newest to oldest, but that can vary.
    # If you want the strictly newest ``MAX_DRAFTS``, you may want to reverse
    # or sort by ``internalDate``.
    unread_messages = unread_messages[: settings["MAX_DRAFTS"]]
    print(
        f"Fetched {len(unread_messages)} unread messages (limited to {settings['MAX_DRAFTS']}).")

    # Process each unread message
    for msg_ref in unread_messages:
        msg_id = msg_ref["id"]
        msg_detail = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=msg_id,
                format="full",
                metadataHeaders=["From", "Subject"],
            )
            .execute()
        )

        subject = get_header_value(msg_detail, "Subject")
        sender = get_header_value(msg_detail, "From")
        thread_id = msg_detail.get("threadId")
        snippet = msg_detail.get("snippet", "")

        payload = msg_detail.get("payload", {})
        body_txt = extract_plain_text(payload)

        # Skip newsletters or spam before using any AI models
        if is_promotional_or_spam(msg_detail, body_txt):
            print(f"{msg_id[:8]}… skipped promotional/spam")
            continue

        cls = classify_email(f"Subject:{subject}\n\n{body_txt}")
        email_type = cls["type"]
        importance = cls["importance"]
        if email_type == "other":
            continue
        print(f"{msg_id[:8]}… type={email_type}, imp={importance}")

        create_ticket(
            subject,
            sender,
            body_txt,
            thread_id=thread_id,
            message_id=msg_id,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
