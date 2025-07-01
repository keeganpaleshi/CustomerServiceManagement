# CustomerServiceManagement

This project automates drafting Gmail replies using OpenAI's API. It can also create customer service tickets in FreeScout.

## Prerequisites

1. **Python 3.10+** with the dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. **Google credentials** for the Gmail API stored as `client_secret.json` in this directory.

## Required Environment Variables

- `OPENAI_API_KEY` – API key used to call OpenAI models.
- `FREESCOUT_URL` – URL of your FreeScout instance (optional).
- `FREESCOUT_KEY` – API key for FreeScout (optional).
- `DRAFT_MODEL` – OpenAI model for generating replies (defaults to `o3`).
- `CLASSIFY_MODEL` – Model for categorizing emails (defaults to the value of `DRAFT_MODEL`).

## Creating Gmail Credentials

1. Create a project at <https://console.cloud.google.com/> and enable the **Gmail API**.
2. Configure an OAuth consent screen for an internal or testing user.
3. Create an **OAuth client ID** of type *Desktop app*.
4. Download the resulting JSON file and save it as `client_secret.json` in this repository.
5. The first time you run the script, a browser window will open to authorize access and a `token.pickle` file will be created for future runs.

## Running the Script

Set the required environment variables and run `Draft_Replies.py`:

```bash
export OPENAI_API_KEY=sk-...
export FREESCOUT_URL=https://freescout.example.com # optional
export FREESCOUT_KEY=your-api-key               # optional
python Draft_Replies.py
```

## Usage

Authentication uses OAuth2 credentials. By default the scripts look for
`client_secret.json` and create a `token.pickle` file. You can override these
filenames with environment variables before running any of the tools:

```bash
export GMAIL_CLIENT_SECRET=/path/to/client_secret.json
export GMAIL_TOKEN_FILE=/path/to/token.pickle
```

Alternatively, both `Draft_Replies.get_gmail_service()` and
`gmail_bot.get_gmail_service()` accept `creds_filename` and `token_filename`
arguments to explicitly set the paths in code.

The script fetches up to 100 unread emails, generates a draft response for each using OpenAI, and leaves the drafts in Gmail without sending.

## Example Workflow

1. Install dependencies and create `client_secret.json` as described above.
2. Set your `OPENAI_API_KEY` and any FreeScout variables.
3. Run `python Draft_Replies.py`.
4. Review the drafts created in your Gmail account, edit as needed, and send them manually.


