# CustomerServiceManagement


## Setup

Install Python dependencies required for the Gmail draft automation:

```bash
pip install -r requirements.txt
```

Set your OpenAI credentials:

```bash
export OPENAI_API_KEY=<your-key>
```

The scripts default to using `gpt-4-1106-preview` for both drafting and
classifying emails. You need access to this GPT-4 model. If you do not have
access, set the `DRAFT_MODEL` and `CLASSIFY_MODEL` environment variables to a
model you can use.
