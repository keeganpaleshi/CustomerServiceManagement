import base64
import argparse
from utils import (
    get_gmail_service, fetch_all_unread_messages, create_base64_message,
    create_draft, thread_has_draft, is_promotional_or_spam,
    critic_email, classify_email, create_ticket,
    GMAIL_QUERY, HTTP_TIMEOUT, MAX_DRAFTS, CRITIC_THRESHOLD, MAX_RETRIES
)

from Draft_Replies import generate_ai_reply


def parse_args():
    parser = argparse.ArgumentParser(description="Process Gmail messages")
    parser.add_argument("--gmail-query", default=GMAIL_QUERY, help="Gmail search query")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP request timeout")
    return parser.parse_args()


def main():
    args = parse_args()

    svc = get_gmail_service()
    for ref in fetch_all_unread_messages(svc, query=args.gmail_query)[:MAX_DRAFTS]:
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
        create_ticket(subject, sender, body, timeout=args.timeout)

        print(f"{ref['id'][:8]}… {email_type:<8} imp={importance}")


if __name__ == "__main__":
    main()
