import os
import base64
import json
import yaml
from openai import OpenAI
from gmail_utils import (
    get_gmail_service,
    fetch_all_unread_messages,
    create_base64_message,
    create_draft,
    is_promotional_or_spam,
    classify_email,
)


# -------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------
# Load YAML config
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)

# Gmail API scope
SCOPES = CFG["gmail"]["scopes"]

# Load OpenAI API key from env (never store in plain text!)
OPENAI_API_KEY = os.getenv(CFG["openai"]["api_key_env"])
if not OPENAI_API_KEY:
    raise ValueError(f"Please set your {CFG['openai']['api_key_env']} environment variable.")

# Default to the o3 model unless overridden in config
DRAFT_MODEL = CFG["openai"]["draft_model"]
DRAFT_MAX_TOKENS = CFG["openai"].get("draft_max_tokens", 16384)
DRAFT_SYSTEM_MSG = CFG["openai"].get("draft_system_message", "")

# Use the same model for general OpenAI calls by default
OPENAI_MODEL = DRAFT_MODEL

# Model used when classifying incoming emails
CLASSIFY_MODEL = CFG["openai"]["classify_model"]
CLASSIFY_MAX_TOKENS = CFG["openai"].get("classify_max_tokens", 50)

# Critic settings
CRITIC_THRESHOLD = CFG["thresholds"]["critic_threshold"]
MAX_RETRIES      = CFG["thresholds"]["max_retries"]

MAX_DRAFTS = CFG.get("limits", {}).get("max_drafts", 100)

# Labels that indicate promotional or spam content. Any message with these
# Gmail labels will be skipped.
# We'll use the new v1.0.0+ style:
from openai import OpenAI


# -------------------------------------------------------
# Module 1 - Classify incoming email
# -------------------------------------------------------


# The classify_email function is defined later with error handling.


# -------------------------------------------------------
# Module 3 - Evaluate AI Drafts
# -------------------------------------------------------
def critic_email(draft: str, original: str) -> dict:
    """Self-grade a draft reply using GPT-4.1."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY JSON {\"score\":1-10,\"feedback\":\"...\"} "
                    "rating on correctness, tone, length."
                ),
            },
            {"role": "assistant", "content": draft},
            {"role": "user", "content": f"Original email:\n\n{original}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# -------------------------------------------------------
# 2) Gmail Service Setup
# -------------------------------------------------------

# -------------------------------------------------------
# 5) OpenAI Integration
# -------------------------------------------------------
def generate_ai_reply(subject, sender, snippet_or_body, email_type):
    """
    Generate a draft reply using OpenAI's new library (>=1.0.0).
    """
    # Create a client instance with your API key
    client = OpenAI(api_key=OPENAI_API_KEY)

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
            model=DRAFT_MODEL,
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM_MSG},
                {"role": "user", "content": instructions},
            ],
            max_tokens=DRAFT_MAX_TOKENS,
            temperature=0.7,
        )
        # Extract the actual reply text
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
# 6) Main Flow
# -------------------------------------------------------
def main():
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
    unread_messages = fetch_all_unread_messages(service)
    if not unread_messages:
        print("No unread messages found.")
        return

    # We assume the API returns them from newest to oldest, but that can vary.
    # If you want the strictly newest ``MAX_DRAFTS``, you may want to reverse
    # or sort by ``internalDate``.
    unread_messages = unread_messages[:MAX_DRAFTS]
    print(f"Fetched {len(unread_messages)} unread messages (limited to {MAX_DRAFTS}).")

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
                        body_txt = base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8")
                        break
        else:
            data = payload.get("body", {}).get("data", "")
            if data:
                body_txt = base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8")

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

        # 1) Check if there's already a draft in this thread
        if thread_has_draft(service, thread_id):
            print(
                f"Skipping message {msg_id} (thread {thread_id}) because a draft already exists."
            )
            continue


        # 2) If not, generate a new draft
        reply_subject = f"Re: {subject}" if subject else "Re: (no subject)"
        draft_body_text = generate_ai_reply(subject, sender, snippet, email_type)

        # Self-grade the AI-generated reply to ensure quality
        critique = critic_email(
            draft_body_text,
            f"Subject:{subject}\n\n{body_txt}",
        )
        score = critique.get("score", 0)
        if score < CRITIC_THRESHOLD:
            print(
                f"Draft for message {msg_id} scored {score} (<{CRITIC_THRESHOLD}). Creating ticket."
            )
            create_ticket(subject, sender, body_txt)
            continue
        else:
            print(f"Draft score {score} >= {CRITIC_THRESHOLD}; saving draft.")

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
