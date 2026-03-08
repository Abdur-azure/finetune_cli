# xlmtec

[![PyPI version](https://img.shields.io/pypi/v/xlmtec.svg)](https://pypi.org/project/xlmtec/)
[![Python](https://img.shields.io/pypi/pyversions/xlmtec.svg)](https://pypi.org/project/xlmtec/)
[![License](https://img.shields.io/pypi/l/xlmtec.svg)](https://github.com/Abdur-azure/xlmtec/blob/main/LICENSE)

**xlmtec** is a command-line toolkit for fine-tuning large language models. Describe your task in plain English, get a ready-to-run config, browse HuggingFace models, and train — all from the terminal.

---

## Features

- **AI-powered config generation** — describe your task, get a YAML config from Claude, Gemini, or GPT
- **Model Hub browser** — search and inspect HuggingFace models without leaving the terminal
- **5 fine-tuning methods** — LoRA, QLoRA, Full, Instruction, DPO
- **Config validation** — catch errors before training starts
- **Dry-run mode** — preview your training plan without loading a model
- **Rich terminal UI** — progress bars, panels, colour output throughout

---

## Installation

```bash
# Core (lightweight — no ML deps)
pip install xlmtec

# With training support
pip install xlmtec[ml]

# With AI suggestions (pick your provider)
pip install xlmtec[claude]    # Anthropic
pip install xlmtec[gemini]    # Google
pip install xlmtec[codex]     # OpenAI
pip install xlmtec[ai]        # All three

# Everything
pip install xlmtec[full]
```

---

## Quickstart

### 1. Get an AI-generated config

```bash
xlmtec ai-suggest "fine-tune a small model for customer support" --provider claude
```

Outputs a ready-to-run YAML config and the exact command to run.

### 2. Browse models on HuggingFace

```bash
xlmtec hub search "bert" --task text-classification --limit 5
xlmtec hub trending
xlmtec hub info google/bert-base-uncased
```

### 3. Validate your config

```bash
xlmtec config validate config.yaml
```

### 4. Train

```bash
# Preview without loading model
xlmtec train --config config.yaml --dry-run

# Start training
xlmtec train --config config.yaml
```

---

## Commands

| Command | Description |
|---------|-------------|
| `xlmtec ai-suggest "<task>"` | Generate a config from plain English |
| `xlmtec hub search "<query>"` | Search HuggingFace models |
| `xlmtec hub info <model-id>` | Show model details |
| `xlmtec hub trending` | Top trending models |
| `xlmtec config validate <file>` | Validate a YAML config |
| `xlmtec train --config <file>` | Fine-tune a model |
| `xlmtec train --config <file> --dry-run` | Preview training plan |
| `xlmtec recommend` | Get method recommendation for your hardware |
| `xlmtec evaluate` | Evaluate a fine-tuned model |
| `xlmtec benchmark` | Compare multiple runs |
| `xlmtec merge` | Merge LoRA adapter into base model |
| `xlmtec upload` | Upload model to HuggingFace Hub |
| `xlmtec --version` | Show installed version |

---

## Fine-tuning methods

| Method | VRAM | Best for |
|--------|------|----------|
| `lora` | Low (4–8 GB) | Most tasks, fast convergence |
| `qlora` | Very low (4 GB) | Large models on limited hardware |
| `full` | High (24 GB+) | Best quality, small models |
| `instruction` | Low (4–8 GB) | Prompt/response style tasks |
| `dpo` | Low (4–8 GB) | Preference learning from pairs |

---

## AI Providers

Set your API key as an environment variable, then pass `--provider`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
xlmtec ai-suggest "summarise legal documents" --provider claude

export GEMINI_API_KEY=...
xlmtec ai-suggest "summarise legal documents" --provider gemini

export OPENAI_API_KEY=sk-...
xlmtec ai-suggest "summarise legal documents" --provider codex
```

---

## Example config

```yaml
model:
  name: gpt2

dataset:
  source: local_file
  path: data/train.jsonl

lora:
  r: 16
  alpha: 32
  target_modules: [c_attn]

training:
  output_dir: output/run1
  num_epochs: 3
  batch_size: 4
  learning_rate: 2e-4
```

---

## Development

```bash
git clone https://github.com/Abdur-azure/xlmtec.git
cd xlmtec
pip install -e ".[full,dev]"
pytest tests/ -v --ignore=tests/test_integration.py
```

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full release history.

## License

MIT