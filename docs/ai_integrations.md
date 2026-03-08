# AI Integrations

The `xlmtec ai-suggest` command uses an AI provider to turn a plain-English task description into a ready-to-run training config.

---

## Supported providers

| Provider | Package | Env variable |
|----------|---------|-------------|
| Claude (Anthropic) | `xlmtec[claude]` | `ANTHROPIC_API_KEY` |
| Gemini (Google) | `xlmtec[gemini]` | `GEMINI_API_KEY` |
| GPT (OpenAI) | `xlmtec[codex]` | `OPENAI_API_KEY` |

---

## Setup

### Claude

```bash
pip install xlmtec[claude]
export ANTHROPIC_API_KEY=sk-ant-...
xlmtec ai-suggest "fine-tune for sentiment analysis" --provider claude
```

### Gemini

```bash
pip install xlmtec[gemini]
export GEMINI_API_KEY=...
xlmtec ai-suggest "fine-tune for sentiment analysis" --provider gemini
```

### GPT (Codex)

```bash
pip install xlmtec[codex]
export OPENAI_API_KEY=sk-...
xlmtec ai-suggest "fine-tune for sentiment analysis" --provider codex
```

### All providers at once

```bash
pip install xlmtec[ai]
```

---

## Usage

```bash
# Basic
xlmtec ai-suggest "fine-tune GPT-2 for customer support classification"

# Specify provider
xlmtec ai-suggest "qlora llama for code generation" --provider gemini

# Save config to file
xlmtec ai-suggest "instruction tune for QA" --provider claude --save config.yaml

# Pass API key directly
xlmtec ai-suggest "summarise medical notes" --provider codex --api-key sk-...
```

---

## Output

Each suggestion includes:

- **Method** — recommended fine-tuning method (LoRA, QLoRA, full, instruction, DPO)
- **YAML config** — complete, ready-to-run `xlmtec train` configuration
- **Explanation** — why this config was chosen for your task
- **Command** — exact command to run

Example output:

```
Recommendation
──────────────
Method: lora
Why: LoRA is ideal for text classification — low VRAM requirement,
     fast convergence, and strong performance on GPT-2 scale models.

Generated config:
model:
  name: gpt2
dataset:
  source: local_file
  path: data/train.jsonl
lora:
  r: 16
  alpha: 32
training:
  output_dir: output/run1
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4

Run this:
xlmtec train --method lora --config config.yaml --output-dir output/run1
```

---

## How it works

1. Your task description is sent to the chosen AI provider
2. The provider returns a structured JSON response with method, YAML, and explanation
3. xlmtec parses and displays the result — no raw JSON exposed
4. Optionally saves the YAML to disk with `--save`

All three providers use the same system prompt and return identical output structure, so results are comparable across providers.