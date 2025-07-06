import os

# additional imports
import json
import requests
import base64

from gmail_utils import (
    get_gmail_service,
    fetch_all_unread_messages,
    create_base64_message,
    create_draft,
    is_promotional_or_spam,
    classify_email,
)

from openai import OpenAI
from Draft_Replies import generate_ai_reply
import yaml

# Load YAML config
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CFG = yaml.safe_load(f)


# OpenAI models & API key
OPENAI_API_KEY       = os.getenv(CFG["openai"]["api_key_env"])
CLASSIFY_MODEL       = CFG["openai"]["classify_model"]
CLASSIFY_MAX_TOKENS  = CFG["openai"].get("classify_max_tokens", 50)

# Critic settings
CRITIC_THRESHOLD     = CFG["thresholds"]["critic_threshold"]
MAX_RETRIES          = CFG["thresholds"]["max_retries"]

MAX_DRAFTS           = CFG.get("limits", {}).get("max_drafts", 100)

# Gmail label IDs that indicate promotional or spammy content. Messages with
# any of these labels will be skipped entirely.
PROMO_LABELS = {
    "SPAM",
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_UPDATES",
    "CATEGORY_FORUMS",
}

# Ticketing
TICKET_SYSTEM        = CFG["ticket"]["system"]
FREESCOUT_URL        = CFG["ticket"]["freescout_url"]
FREESCOUT_KEY        = CFG["ticket"]["freescout_key"]

if not OPENAI_API_KEY:
    raise ValueError(f"Please set your {CFG['openai']['api_key_env']} environment variable.")

def create_ticket(subject: str, sender: str, body: str):
    if TICKET_SYSTEM != "freescout":
        return
    url = f"{FREESCOUT_URL.rstrip('/')}/api/conversations"
    payload = {
        "type": "email",
        "subject": subject or "(no subject)",
        "customer": {"email": sender},
        "threads": [{"type": "customer", "text": body}],
    }
    r = requests.post(
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-FreeScout-API-Key": FREESCOUT_KEY,
        },
        json=payload,
        timeout=15,
    )
    r.raise_for_status()


def main():

    svc = get_gmail_service()
    for ref in fetch_all_unread_messages(svc)[:MAX_DRAFTS]:
        msg = (
            svc.users()
            .messages()
            .get(userId="me", id=ref["id"], format="full")
            .execute()
        )
        subject = next(
            (h["value"] for h in msg["payload"]["headers"] if h["name"] == "Subject"),
            "",
        )
        sender = next(
            (h["value"] for h in msg["payload"]["headers"] if h["name"] == "From"),
            "",
        )
        thread = msg["threadId"]

        # decode first text/plain part
        part = msg["payload"]["parts"][0]["body"]["data"]
        body = base64.urlsafe_b64decode(part).decode("utf-8", "ignore")
        snippet = msg.get("snippet", "")

        # Skip obvious newsletters or spam before any AI calls
        if is_promotional_or_spam(msg, body):
            print(f"{ref['id'][:8]}… skipped promotional/spam")
            continue

        # ---- classification ----
        cls = classify_email(f"Subject:{subject}\n\n{body}")
        email_type, importance = cls["type"], cls["importance"]

        # skip others
        if email_type == "other":
            continue

        # ---- draft creation with critic ----
        if not thread_has_draft(svc, thread):
            draft_text = generate_ai_reply(subject, sender, snippet, email_type)
            for _ in range(MAX_RETRIES):
                rating = critic_email(draft_text, body)
                if rating["score"] >= CRITIC_THRESHOLD:
                    break
                draft_text = generate_ai_reply(
                    subject,
                    sender,
                    f"{snippet}\n\nCritic feedback: {rating['feedback']}",
                    email_type,
                )
            msg_draft = create_base64_message(
                "me", sender, f"Re: {subject}", draft_text
            )
            create_draft(svc, "me", msg_draft, thread_id=thread)

        # ---- ticket for lead/customer ----
        create_ticket(subject, sender, body)

        print(f"{ref['id'][:8]}… {email_type:<8} imp={importance}")


if __name__ == "__main__":
    main()

