# finetune-cli

**Production-grade LLM fine-tuning from the command line.**

`finetune-cli` is a modular Python framework for fine-tuning large language models using LoRA and QLoRA. It provides a type-safe configuration system, a composable trainer stack, a benchmarking pipeline, and a clean CLI — all fully tested and CI-verified.

---

## Quick start

```bash
# Install
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
pip install -e .

# Fine-tune GPT-2 on a local dataset
finetune-cli train --model gpt2 --dataset ./data.jsonl --epochs 3

# Or use a config file (recommended)
finetune-cli train --config examples/configs/lora_gpt2.yaml
```

---

## What's included

| Component | Description |
|-----------|-------------|
| `finetune_cli train` | LoRA / QLoRA training with auto-detected target modules |
| `finetune_cli evaluate` | ROUGE, BLEU, Perplexity scoring on a saved checkpoint |
| `finetune_cli benchmark` | Before/after comparison report with delta indicators |
| `finetune_cli upload` | Push adapter or merged model to HuggingFace Hub |
| `ConfigBuilder` | Fluent Python API for building validated pipeline configs |
| `DataPipeline` | Loads JSON/JSONL/CSV/Parquet/HF datasets, tokenizes, splits |
| `TrainerFactory` | Single entry point — selects trainer for lora / qlora / instruction_tuning / full_finetuning |
| `BenchmarkRunner` | Model-agnostic evaluation with comparison reports |
| `finetune_cli merge` | Merge LoRA adapter into base model → standalone model |
| `finetune_cli recommend` | Inspect model size + VRAM, output optimal YAML config |
| `DPOTrainer` | Direct Preference Optimization on prompt/chosen/rejected datasets (requires trl) |

---

## Navigation

- **[Installation](installation.md)** — requirements, GPU setup, HuggingFace login
- **[Usage Guide](usage.md)** — all CLI subcommands with examples
- **[Configuration](configuration.md)** — YAML config reference and parameter guide
- **[API Reference](api.md)** — Python API for programmatic use
- **[Examples](examples.md)** — common workflows and recipes
- **[Troubleshooting](troubleshooting.md)** — OOM, NaN loss, dataset errors

---

## Migrating from v1?

See [CHANGELOG.md](https://github.com/Abdur-azure/finetune_cli/blob/main/CHANGELOG.md) for the full v1 → v2 migration guide.

!!! note "v1 is deprecated"
    Running `python finetune_cli.py` will display a migration message. The v1 interactive wizard has been replaced by the `finetune-cli` subcommands documented here.

---

## Project status

- **Version:** 2.6.0
- **Tests:** 85+ unit tests + integration tests (all green)
- **CI:** pytest matrix across Python 3.10 / 3.11 / 3.12
- **License:** MIT