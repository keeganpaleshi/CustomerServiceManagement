import base64

from Draft_Replies import generate_ai_reply
from common import (
    CFG,
    OPENAI_API_KEY,
    CRITIC_THRESHOLD,
    MAX_RETRIES,
    critic_email,
    classify_email,
    create_base64_message,
    create_draft,
    create_ticket,
    fetch_all_unread_messages,
    get_gmail_service,
    is_promotional_or_spam,
    thread_has_draft,
)

MAX_DRAFTS = CFG.get("limits", {}).get("max_drafts", 100)

if not OPENAI_API_KEY:
    raise ValueError(
        f"Please set your {CFG['openai']['api_key_env']} environment variable."
    )


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
