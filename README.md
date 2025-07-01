# CustomerServiceManagement


## Setup

Install Python dependencies required for the Gmail draft automation:

```bash
pip install -r requirements.txt
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
