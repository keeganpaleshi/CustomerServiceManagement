import os
import pickle
import base64
import openai
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# -------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------
# Gmail API scope
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Load OpenAI API key from env (never store in plain text!)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# EXAMPLE model name   verify you have access to this model
OPENAI_MODEL = "gpt-4.5-preview"

# We'll use the new v1.0.0+ style:
from openai import OpenAI
import json


# -------------------------------------------------------
# Module 1 - Classify incoming email
# -------------------------------------------------------
CLASSIFY_MODEL = OPENAI_MODEL


def classify_email(raw_txt: str) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY JSON with keys: "
                    '{"type":"lead|customer|other","importance":1-10}. No extra text.'
                ),
            },
            {"role": "user", "content": raw_txt},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# -------------------------------------------------------
# 2) Gmail Service Setup
# -------------------------------------------------------
def get_gmail_service(
    creds_filename: str = "client_secret_106355235075-nsep78srfr4f7g4noa2lfmjdemvc3q7h.apps.googleusercontent.com.json",
    token_filename: str = "token.pickle",
):
    """
    Authenticate with Gmail API and return a service resource.
    """
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
    """
    Fetch all unread messages in the Gmail inbox, handling pagination.
    We'll then slice to only the first 100 (likely the newest) in main().
    """
    unread_messages = []
    page_token = None

    while True:
        response = (
            service.users()
            .messages()
            .list(userId="me", q="is:unread", pageToken=page_token)
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


# -------------------------------------------------------
# 5) OpenAI Integration
# -------------------------------------------------------
def generate_ai_reply(subject, sender, snippet_or_body):
    """
    Generate a draft reply using OpenAI's new library (>=1.0.0).
    """
    # Create a client instance with your API key
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Example prompt; tailor as needed
    instructions = (
        "You are an AI email assistant. The user received an email.\n"
        f"Subject: {subject}\n"
        f"From: {sender}\n"
        f"Email content/snippet: {snippet_or_body}\n\n"
        "Please write a friendly and professional draft reply addressing the sender's query."
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Context: This is a business email for Cruising Solutions. You are replying "
                        "to customers who have concerns or questions some about orders they've placed, "
                        "others about products they're considering purchasing. You should reply in the "
                        "name of David, lead Customer Service Member, with the phone number 843-222-3660.\n\n"
                        "Style Guidelines:\n\n"
                        "    Write in an email format.\n"
                        "    Be kind, courteous, and polite.\n"
                        "    Recognize any urgency in the customer's message.\n"
                        "    Provide helpful, succinct responses (most customers appreciate concise emails).\n"
                        "    Avoid giving specific dates or times when you will follow up (e.g., no  today,  "
                        " tomorrow,  or exact deadlines). Instead, use phrases such as:\n"
                        "         as soon as possible \n"
                        "         at your earliest convenience \n"
                        "    Occasionally use nautical terms, as most customers are sailors.\n"
                        "    If you don t have an answer to their question immediately, let them know you re "
                        "checking into it and will respond once you have the information."
                    ),
                },
                {"role": "user", "content": instructions},
            ],
            max_tokens=16384,
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
    2) Fetch unread messages & limit to 100 (likely the newest).
    3) For each message:
       - Skip if there's already a draft in the same thread.
       - Generate AI-based draft.
       - Create the draft (email remains unread).
    """
    service = get_gmail_service()

    # Fetch all unread, then limit to 100 for "most recent" processing
    unread_messages = fetch_all_unread_messages(service)
    if not unread_messages:
        print("No unread messages found.")
        return

    # We assume the API returns them from newest to oldest, but that can vary.
    # If you want the strictly newest 100, you may want to reverse or sort by internalDate.
    unread_messages = unread_messages[:100]
    print(f"Fetched {len(unread_messages)} unread messages (limited to 100).")

    # Process each unread message
    for msg_ref in unread_messages:
        msg_id = msg_ref["id"]
        msg_detail = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=msg_id,
                format="metadata",
                metadataHeaders=["From", "Subject"],
            )
            .execute()
        )

        subject = get_header_value(msg_detail, "Subject")
        sender = get_header_value(msg_detail, "From")
        thread_id = msg_detail.get("threadId")
        snippet = msg_detail.get("snippet", "")

        # 1) Check if there's already a draft in this thread
        thread_data = service.users().threads().get(userId="me", id=thread_id).execute()
        messages_in_thread = thread_data.get("messages", [])
        # If any message in the thread has a label "DRAFT", skip
        already_has_draft = any(
            "DRAFT" in (m.get("labelIds") or []) for m in messages_in_thread
        )
        if already_has_draft:
            print(
                f"Skipping message {msg_id} (thread {thread_id}) because a draft already exists."
            )
            continue

        # 2) If not, generate a new draft
        reply_subject = f"Re: {subject}" if subject else "Re: (no subject)"
        draft_body_text = generate_ai_reply(subject, sender, snippet)
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
