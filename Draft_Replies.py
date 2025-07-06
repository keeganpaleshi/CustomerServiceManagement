import os
import pickle
import base64
import json
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import yaml

# -------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------
# Load YAML config
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)

# Gmail API scope
SCOPES = CFG["gmail"]["scopes"]
GMAIL_QUERY = CFG["gmail"].get("query", "is:unread")

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
PROMO_LABELS = {
    "SPAM",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}


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
def get_gmail_service(
    creds_filename: str | None = None,
    token_filename: str | None = None,
):
    """Authenticate with Gmail API and return a service resource.

    Filenames may be supplied via arguments or will default to the values
    specified in ``config.yaml``.
    """
    creds_filename = creds_filename or CFG["gmail"]["client_secret_file"]
    token_filename = token_filename or CFG["gmail"]["token_file"]
    creds = None

    # Load token if it exists
    if os.path.exists(token_filename):
        with open(token_filename, "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, prompt for login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_filename, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the token for future runs
        with open(token_filename, "wb") as token:
            pickle.dump(creds, token)

    # Build the Gmail service
    service = build("gmail", "v1", credentials=creds)
    return service


# -------------------------------------------------------
# 3) Fetching Unread Messages
# -------------------------------------------------------
def fetch_all_unread_messages(service):
    """Fetch messages matching the configured Gmail query."""
    unread_messages = []
    page_token = None
    query = GMAIL_QUERY

    while True:
        response = (
            service.users()
            .messages()
            .list(userId="me", q=query, pageToken=page_token)
            .execute()
        )
        messages_page = response.get("messages", [])
        if not messages_page:
            break

        unread_messages.extend(messages_page)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return unread_messages


# -------------------------------------------------------
# 4) Email Processing Helpers
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


def create_base64_message(sender, to, subject, body_text):
    """
    Create a MIMEText email and encode it in base64 for the Gmail API.
    """
    msg = MIMEText(body_text)
    msg["to"] = to
    msg["from"] = sender
    msg["subject"] = subject

    raw_bytes = base64.urlsafe_b64encode(msg.as_bytes())
    return {"raw": raw_bytes.decode("utf-8")}


def create_draft(service, user_id, message_body, thread_id=None):
    """
    Create and insert a draft email.
    """
    try:
        body = {"message": message_body}
        if thread_id:
            body["message"]["threadId"] = thread_id

        draft = service.users().drafts().create(userId=user_id, body=body).execute()
        return draft
    except Exception as error:
        print(f"An error occurred creating the draft: {error}")
        return None


def is_promotional_or_spam(message, body_text: str) -> bool:
    """Return True if the message looks like a newsletter or spam."""
    labels = set(message.get("labelIds", []))
    if labels & PROMO_LABELS:
        return True
    headers = {h.get("name", "").lower(): h.get("value", "") for h in message.get("payload", {}).get("headers", [])}
    if "list-unsubscribe" in headers or "list-id" in headers:
        return True
    if "unsubscribe" in body_text.lower():
        return True
    return False


def classify_email(text: str) -> dict:
    """Classify an email and return a dict with type and importance."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Categorize the email as lead, customer, or other. Return ONLY JSON {\"type\":\"lead|customer|other\",\"importance\":1-10}. NO other text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=CLASSIFY_MAX_TOKENS,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error classifying email: {e}")
        return {"type": "other", "importance": 0}


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
