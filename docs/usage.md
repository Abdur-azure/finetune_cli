# Usage

## Getting started

```bash
xlmtec --help        # list all commands
xlmtec --version     # show installed version
```

---

## ai-suggest — Generate a config from plain English

Describe your fine-tuning task and get a ready-to-run YAML config:

```bash
xlmtec ai-suggest "fine-tune GPT-2 for sentiment analysis" --provider claude
xlmtec ai-suggest "qlora on llama for code generation" --provider gemini
xlmtec ai-suggest "instruction tune for customer support" --provider codex --save config.yaml
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--provider` | `claude` | AI provider: `claude`, `gemini`, `codex` |
| `--api-key` | env var | API key (falls back to env variable) |
| `--save` | — | Save generated YAML to a file |

See [AI Integrations](ai_integrations.md) for full provider setup.

---

## hub — Browse HuggingFace models

### Search

```bash
xlmtec hub search "bert"
xlmtec hub search "llama" --task text-generation --limit 10
xlmtec hub search "gpt" --sort likes
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--task` | — | Filter by pipeline tag |
| `--limit` | `10` | Number of results (max 100) |
| `--sort` | `downloads` | Sort by: `downloads`, `likes`, `lastModified` |

### Info

```bash
xlmtec hub info google/bert-base-uncased
xlmtec hub info mistralai/Mistral-7B-v0.1
```

### Trending

```bash
xlmtec hub trending
xlmtec hub trending --limit 20
```

See [Model Hub](hub.md) for more.

---

## config validate — Check a config before training

```bash
xlmtec config validate config.yaml
xlmtec config validate config.yaml --strict    # fail on warnings too
```

Reports all validation errors at once. Exits 0 on success, 1 on any error.

---

## train — Fine-tune a model

### Dry run (no model loading)

```bash
xlmtec train --config config.yaml --dry-run
```

Validates the config and prints the full training plan without loading any model.

### Training

```bash
xlmtec train --config config.yaml
xlmtec train --config config.yaml --method lora
```

---

## recommend — Get a method recommendation

```bash
xlmtec recommend --model-size 7b --vram 8
```

---

## evaluate — Evaluate a model

```bash
xlmtec evaluate --model output/run1 --dataset data/test.jsonl
```

---

## benchmark — Compare runs

```bash
xlmtec benchmark --runs output/run1 output/run2
```

---

## merge — Merge LoRA adapter

```bash
xlmtec merge --adapter output/run1 --output merged_model/
```

---

## upload — Upload to HuggingFace Hub

```bash
xlmtec upload --model output/run1 --repo-id your-username/my-model
```