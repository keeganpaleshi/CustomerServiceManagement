import argparse
import base64
from typing import Optional

from openai import OpenAI

from utils import (
    classify_email,
    create_base64_message,
    create_draft,
    create_ticket,
    critic_email,
    fetch_all_unread_messages,
    get_gmail_service,
    get_settings,
    is_promotional_or_spam,
    thread_has_draft,
)


# -------------------------------------------------------
# Configuration helpers
# -------------------------------------------------------

def parse_args(settings):
    parser = argparse.ArgumentParser(description="Draft Gmail replies")
    parser.add_argument(
        "--gmail-query", default=settings.gmail_query, help="Gmail search query"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=settings.http_timeout,
        help="HTTP request timeout",
    )
    return parser.parse_args()


# -------------------------------------------------------
# Email helpers
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


# -------------------------------------------------------
# OpenAI Integration
# -------------------------------------------------------

def generate_ai_reply(subject, sender, snippet_or_body, email_type, settings=None):
    """Generate a draft reply using OpenAI's new library (>=1.0.0)."""

    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError(
            f"Please set your {settings.openai_api_key_env} environment variable."
        )

    client = OpenAI(api_key=settings.openai_api_key)

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
            model=settings.draft_model,
            messages=[
                {"role": "system", "content": settings.draft_system_message},
                {"role": "user", "content": instructions},
            ],
            max_tokens=settings.draft_max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return (
            "Hello,\n\n"
            "I'm sorry, but I couldn't generate a response at this time. "
            "Please review this email manually.\n\n"
            "Best,\nAutomated Script"
        )


# -------------------------------------------------------
# Main Flow
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
    service = get_gmail_service()

    # Fetch all unread, then limit for "most recent" processing
    unread_messages = fetch_all_unread_messages(service, query=args.gmail_query)
    if not unread_messages:
        print("No unread messages found.")
        return

    unread_messages = unread_messages[: settings.max_drafts]
    print(
        f"Fetched {len(unread_messages)} unread messages (limited to {settings.max_drafts})."
    )

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

        # Decode the plain text body if available
        body_txt = ""
        payload = msg_detail.get("payload", {})
        if "parts" in payload:
            for part in payload.get("parts", []):
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        body_txt = base64.urlsafe_b64decode(
                            data.encode("utf-8")
                        ).decode("utf-8")
                        break
        else:
            data = payload.get("body", {}).get("data", "")
            if data:
                body_txt = base64.urlsafe_b64decode(data.encode("utf-8")).decode(
                    "utf-8"
                )

        # Skip newsletters or spam before using any AI models
        if is_promotional_or_spam(msg_detail, body_txt):
            print(f"{msg_id[:8]}… skipped promotional/spam")
            continue

        cls = classify_email(f"Subject:{subject}\n\n{body_txt}", settings=settings)
        email_type = cls["type"]
        importance = cls["importance"]
        if email_type == "other":
            continue
        print(f"{msg_id[:8]}… type={email_type}, imp={importance}")

        # 1) Check if there's already a draft in this thread
        if thread_has_draft(service, thread_id):
            print(
                f"Skipping message {msg_id} (thread {thread_id}) because a draft already exists."
            )
            continue

        # 2) If not, generate a new draft
        reply_subject = f"Re: {subject}" if subject else "Re: (no subject)"
        draft_body_text = generate_ai_reply(
            subject, sender, snippet, email_type, settings=settings
        )

        # Self-grade the AI-generated reply to ensure quality
        critique = critic_email(
            draft_body_text,
            f"Subject:{subject}\n\n{body_txt}",
            settings=settings,
        )
        score = critique.get("score", 0)
        if score < settings.critic_threshold:
            print(
                f"Draft for message {msg_id} scored {score} (<{settings.critic_threshold}). Creating ticket."
            )
            create_ticket(
                subject, sender, body_txt, timeout=args.timeout, settings=settings
            )
            continue
        else:
            print(f"Draft score {score} >= {settings.critic_threshold}; saving draft.")

        draft_message = create_base64_message(
            "me", sender, reply_subject, draft_body_text
        )

        draft = create_draft(service, "me", draft_message, thread_id=thread_id)
        if draft:
            print(
                f"Draft created for unread email from {sender} (subject: '{subject}')"
            )
        else:
            print(f"Failed to create draft for message ID {msg_id}")


if __name__ == "__main__":
    main()
