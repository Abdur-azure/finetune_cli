# Model Hub

The `xlmtec hub` commands let you search, browse, and inspect HuggingFace models without leaving the terminal.

No API key required — uses the public HuggingFace API.

---

## Search

Find models by keyword with optional filters:

```bash
# Basic search
xlmtec hub search "bert"

# Filter by task
xlmtec hub search "bert" --task text-classification

# Sort by likes, limit results
xlmtec hub search "gpt" --sort likes --limit 20

# Find multilingual models
xlmtec hub search "multilingual" --task translation
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--task`, `-t` | — | Pipeline tag filter (see common tasks below) |
| `--limit`, `-n` | `10` | Number of results (1–100) |
| `--sort`, `-s` | `downloads` | Sort by: `downloads`, `likes`, `lastModified` |

**Common task filters:**

- `text-classification`
- `text-generation`
- `token-classification`
- `question-answering`
- `summarization`
- `translation`
- `fill-mask`
- `text2text-generation`

---

## Info

Show full details for a specific model:

```bash
xlmtec hub info google/bert-base-uncased
xlmtec hub info mistralai/Mistral-7B-v0.1
xlmtec hub info meta-llama/Llama-2-7b-hf
```

Output includes:

- Model ID, author, task, library
- Download count and likes
- Model size (MB)
- Tags and languages
- Last modified date
- Direct link to HuggingFace page

---

## Trending

Show the most downloaded models right now:

```bash
xlmtec hub trending
xlmtec hub trending --limit 20
```

---

## Typical workflow

```bash
# 1. Find candidate models
xlmtec hub search "bert" --task text-classification --limit 5

# 2. Inspect the best candidate
xlmtec hub info google/bert-base-uncased

# 3. Generate a training config for it
xlmtec ai-suggest "fine-tune google/bert-base-uncased for sentiment analysis" --provider claude

# 4. Validate and train
xlmtec config validate config.yaml
xlmtec train --config config.yaml --dry-run
xlmtec train --config config.yaml
```