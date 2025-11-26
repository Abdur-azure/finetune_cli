# HuggingFace Upload

Share your fine-tuned models with the community.

## Prerequisites

1.  **HuggingFace Account**: Create one at [huggingface.co](https://huggingface.co).
2.  **Access Token**: Get a **Write** token from [Settings > Tokens](https://huggingface.co/settings/tokens).

## The Upload Process

At the end of the interactive workflow, the CLI asks:

```
🤗 Upload model to HuggingFace Hub? [y/N]:
```

If you select `y`:

1.  **Repository Name**: Enter `username/repo-name`.
2.  **Token**: Paste your Write token (input will be hidden).
3.  **Privacy**: Choose public or private repository.

The CLI will:
1.  Login to HuggingFace.
2.  Create the repository if it doesn't exist.
3.  Upload all model artifacts (adapters, tokenizer configs).
4.  Provide a direct link to your model.

## Using Uploaded Models

Users can load your model directly:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "username/repo-name")
```
