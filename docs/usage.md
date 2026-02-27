# Usage Guide

This guide covers all six `finetune-cli` commands with real examples.

---

## 5-minute quickstart

**Step 1 — Install:**
```bash
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
pip install -e .
```

**Step 2 — Generate sample data (no network required):**
```bash
python examples/generate_sample_data.py
# Creates:
#   data/sample.jsonl          (500 rows, causal LM)
#   data/instructions.jsonl    (300 rows, alpaca format)
#   data/dpo_sample.jsonl      (200 rows, prompt/chosen/rejected)
```

**Step 3 — Not sure which method to use? Ask:**
```bash
finetune-cli recommend gpt2 --output my_config.yaml
finetune-cli train --config my_config.yaml
```

**Step 4 — Or use a ready-made config:**
```bash
finetune-cli train --config examples/configs/lora_gpt2.yaml
finetune-cli train --config examples/configs/instruction_tuning.yaml
finetune-cli train --config examples/configs/full_finetuning.yaml
finetune-cli train --config examples/configs/dpo.yaml        # requires: pip install trl>=0.7.0
```

---

## Command reference

| Command | What it does |
|---------|-------------|
| `finetune-cli train` | Fine-tune using a YAML/JSON config or inline flags |
| `finetune-cli evaluate` | Score a saved checkpoint (ROUGE, BLEU, Perplexity) |
| `finetune-cli benchmark` | Before/after comparison: base vs fine-tuned |
| `finetune-cli upload` | Push adapter or merged model to HuggingFace Hub |
| `finetune-cli merge` | Merge LoRA adapter into base model — standalone model |
| `finetune-cli recommend` | Inspect model size + VRAM, output optimal YAML config |

---

## `train` — Fine-tune a model

### Using a config file (recommended)

```bash
finetune-cli train --config examples/configs/lora_gpt2.yaml
```

### Using flags (quick experiments)

```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/sample.jsonl \
  --method lora \
  --epochs 3 \
  --output ./output
```

### Training methods

#### LoRA (default — general purpose)
```bash
finetune-cli train --model gpt2 --dataset ./data/sample.jsonl --method lora --epochs 3
```

#### QLoRA (4-bit — large models on limited VRAM)
```bash
finetune-cli train \
  --model meta-llama/Llama-3.2-1B \
  --dataset ./data/sample.jsonl \
  --method qlora --4bit --fp16 --epochs 2
```

#### Instruction tuning (alpaca-style `{instruction, input, response}` data)
```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/instructions.jsonl \
  --method instruction_tuning --epochs 3
# or: finetune-cli train --config examples/configs/instruction_tuning.yaml
```

#### Full fine-tuning (all parameters — small models only)
```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data/sample.jsonl \
  --method full_finetuning --lr 1e-5 --epochs 3
# or: finetune-cli train --config examples/configs/full_finetuning.yaml
```

> **Warning:** Full fine-tuning trains every parameter. Only safe on models ≤300M params
> unless you have 24GB+ VRAM. Run `finetune-cli recommend` first if unsure.

#### DPO (Direct Preference Optimization — `{prompt, chosen, rejected}` data)
```bash
pip install trl>=0.7.0   # one-time

finetune-cli train \
  --model gpt2 \
  --dataset ./data/dpo_sample.jsonl \
  --method dpo --lr 5e-5 --epochs 1
# or: finetune-cli train --config examples/configs/dpo.yaml
```

### All `train` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | YAML or JSON config file (overrides all flags) |
| `--model` | `gpt2` | HuggingFace model id |
| `--dataset` | — | Path to local dataset file |
| `--hf-dataset` | — | HuggingFace dataset id |
| `--output` | `./output` | Output directory |
| `--method` | `lora` | `lora` / `qlora` / `instruction_tuning` / `full_finetuning` / `dpo` |
| `--lora-r` | `8` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.1` | LoRA dropout |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--lr` | `2e-4` | Learning rate |
| `--max-length` | `512` | Max token sequence length |
| `--max-samples` | — | Limit dataset to N samples |
| `--4bit` | off | Load model in 4-bit (QLoRA) |
| `--fp16` | off | Mixed precision FP16 |

---

## `recommend` — Get an optimal config

Inspects model parameter count and available VRAM, outputs a ready-to-use YAML config.

```bash
# Print to stdout
finetune-cli recommend gpt2

# Save and train
finetune-cli recommend gpt2 --output my_config.yaml
finetune-cli train --config my_config.yaml
```

Decision logic:

| Model size | VRAM | Recommended method |
|-----------|------|--------------------|
| > 7B | any | qlora + grad checkpoint |
| > 1B | ≥ 16 GB | lora, r=16 |
| > 1B | < 16 GB | qlora, r=8 |
| ≤ 300M | ≥ 4 GB | lora, r=8 |
| ≤ 300M | < 4 GB | full_finetuning |

---

## `merge` — Merge adapter into standalone model

Bakes a LoRA adapter into its base model, saving a standard HuggingFace model
that runs without PEFT installed.

```bash
finetune-cli merge ./output/lora ./output/merged \
  --base-model gpt2 \
  --dtype float16
```

| Flag | Default | Description |
|------|---------|-------------|
| `--base-model` | required | HuggingFace base model id |
| `--dtype` | `float32` | `float32` / `float16` / `bfloat16` |

Use merge when:
- Sharing the model publicly (no PEFT dependency for inference)
- Deploying to servers that don't support PEFT
- Reducing inference latency (no adapter overhead)

Typical merge → upload workflow:
```bash
finetune-cli merge ./output/lora ./output/merged --base-model gpt2 --dtype float16
finetune-cli upload ./output/merged username/my-gpt2
```

---

## `evaluate` — Score a saved checkpoint

```bash
finetune-cli evaluate \
  --model-path ./output \
  --dataset ./test.jsonl \
  --metrics rougeL,bleu
```

Output:
```
Results:
  rougeL               0.4231
  bleu                 0.1876
```

---

## `benchmark` — Compare base vs fine-tuned

```bash
finetune-cli benchmark gpt2 ./output \
  --dataset ./test.jsonl \
  --metrics rougeL,bleu \
  --num-samples 200
```

Output:
```
Metric       Base      Fine-tuned   Delta
rougeL       0.3012    0.4231       ▲ 0.1219
bleu         0.1204    0.1876       ▲ 0.0672
```

---

## `upload` — Push to HuggingFace Hub

```bash
# Upload LoRA adapter
finetune-cli upload ./output username/my-model --token $HF_TOKEN

# Merge then upload (recommended for sharing)
finetune-cli upload ./output username/my-model \
  --merge-adapter --base-model gpt2 --token $HF_TOKEN

# Private repository
finetune-cli upload ./output username/my-model --private
```

Set `HF_TOKEN` in your environment to avoid passing `--token` each time:
```bash
export HF_TOKEN=hf_...
```

---

## Supported dataset formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JSON Lines | `.jsonl` | One JSON object per line — recommended |
| JSON | `.json` | Array of objects |
| CSV | `.csv` | Auto-detects text columns |
| Parquet | `.parquet` | Columnar format |
| Plain text | `.txt` | One sample per line |
| HuggingFace Hub | — | Any public dataset via `--hf-dataset` |

### Column requirements by method

| Method | Required columns |
|--------|-----------------|
| `lora`, `qlora`, `full_finetuning` | Any string column (auto-detected) |
| `instruction_tuning` | `instruction`, `input` (optional), `response` |
| `dpo` | `prompt`, `chosen`, `rejected` |

Generate offline sample data for all methods:
```bash
python examples/generate_sample_data.py
```