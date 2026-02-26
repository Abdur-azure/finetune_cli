# Usage Guide

This guide covers all four `finetune-cli` subcommands with real examples.

---

## 5-minute quickstart

**Step 1 — Install:**
```bash
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
pip install -e .
```

**Step 2 — Generate sample data:**
```bash
python examples/generate_sample_data.py
# Creates: data/sample.jsonl (500 rows, causal LM)
#          data/instructions.jsonl (300 rows, alpaca format)
```

**Step 3 — Train:**
```bash
# LoRA on GPT-2 (CPU-safe, ~2 min)
finetune-cli train --config examples/configs/lora_gpt2.yaml

# Instruction tuning (alpaca format)
finetune-cli train --config examples/configs/instruction_tuning.yaml

# Full fine-tuning (small model only)
finetune-cli train --config examples/configs/full_finetuning.yaml
```

**Step 4 — Not sure which config to use? Ask:**
```bash
finetune-cli recommend gpt2 --output my_config.yaml
finetune-cli train --config my_config.yaml
```

---

---

## Installation check

```bash
finetune-cli --help
```

---

## `train` — Fine-tune a model

### Using flags (quick experiments)

```bash
finetune-cli train \
  --model gpt2 \
  --dataset ./data.jsonl \
  --lora-r 8 \
  --lora-alpha 32 \
  --epochs 3 \
  --batch-size 4 \
  --output ./output
```

### Using a config file (recommended for reproducibility)

```bash
finetune-cli train --config examples/configs/lora_gpt2.yaml
```

### QLoRA (4-bit, memory-efficient)

```bash
finetune-cli train \
  --model meta-llama/Llama-3.2-1B \
  --dataset ./data.jsonl \
  --4bit \
  --fp16 \
  --lora-r 16 \
  --epochs 2 \
  --output ./output_qlora
```

### All `train` flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | — | YAML or JSON config file (overrides all flags) |
| `--model` | `gpt2` | HuggingFace model id |
| `--dataset` | — | Path to local dataset file |
| `--hf-dataset` | — | HuggingFace dataset id |
| `--output` | `./output` | Output directory |
| `--method` | `lora` | Training method (`lora`, `qlora`) |
| `--lora-r` | `8` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.1` | LoRA dropout |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--lr` | `2e-4` | Learning rate |
| `--max-length` | `512` | Max token sequence length |
| `--max-samples` | — | Limit dataset to N samples |
| `--4bit` | off | Load model in 4-bit (QLoRA) |
| `--fp16` | off | Mixed precision FP16 |

---

## `evaluate` — Score a saved checkpoint

```bash
finetune-cli evaluate \
  --model-path ./output \
  --dataset ./test.jsonl \
  --metrics rougeL,bleu
```

Prints a score table:

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

Prints a before/after delta report:

```
Metric       Base      Fine-tuned   Delta
rougeL       0.3012    0.4231       ▲ 0.1219
bleu         0.1204    0.1876       ▲ 0.0672
```

---

## `upload` — Push to HuggingFace Hub

### Upload LoRA adapter (default)

```bash
finetune-cli upload ./output username/my-model --token $HF_TOKEN
```

### Merge adapter into base model, then upload

```bash
finetune-cli upload ./output username/my-model \
  --merge-adapter \
  --base-model gpt2 \
  --token $HF_TOKEN
```

The merged model is a standard HuggingFace model — no PEFT dependency required for inference.

### Private repository

```bash
finetune-cli upload ./output username/my-model --private
```

!!! tip "Token via environment variable"
    Set `HF_TOKEN` in your environment instead of passing `--token` every time:
    ```bash
    export HF_TOKEN=hf_...
    finetune-cli upload ./output username/my-model
    ```

---

## Supported dataset formats

| Format | Extension | Notes |
|--------|-----------|-------|
| JSON Lines | `.jsonl` | One JSON object per line — recommended |
| JSON | `.json` | Array of objects or single dict |
| CSV | `.csv` | Auto-detects text columns |
| Parquet | `.parquet` | Columnar format |
| Plain text | `.txt` | One sample per line |
| HuggingFace | — | Any public Hub dataset via `--hf-dataset` |

Text columns are auto-detected. To specify explicitly, use a config file with `dataset.text_columns`.