---
title: xlmtec — LLM Fine-Tuning CLI
description: >
  xlmtec is a production-grade Python CLI for fine-tuning Large Language Models
  using LoRA, QLoRA, DPO, instruction tuning, knowledge distillation, structured
  pruning, hyperparameter sweeps, and model export. Open-source, MIT licensed.
---

# xlmtec — LLM Fine-Tuning CLI

**xlmtec** is an open-source Python framework and CLI tool for fine-tuning
Large Language Models (LLMs) on custom datasets — without writing boilerplate
training code.

```bash
pip install xlmtec[ml]
xlmtec train --model gpt2 --dataset data/train.jsonl --method lora
```

---

## Why xlmtec?

Most LLM fine-tuning code is either a single-script notebook or a heavyweight
framework. xlmtec sits in between: **a modular CLI** that handles the full
pipeline — from dataset loading and tokenization to training, evaluation,
hyperparameter search, and model export — with sensible defaults and rich
terminal output.

---

## Supported fine-tuning methods

| Method | Description | Best for |
|--------|-------------|---------|
| **LoRA** | Low-rank adapter injection via PEFT | Most models, most tasks |
| **QLoRA** | 4-bit quantized LoRA | Large models on consumer GPUs |
| **Instruction tuning** | Alpaca-format fine-tuning | Chat / instruction following |
| **DPO** | Direct Preference Optimization | Alignment without reward model |
| **Vanilla distillation** | Response-level knowledge distillation | Model compression |
| **Feature distillation** | Hidden-state KD from teacher model | High-quality compression |
| **Structured pruning** | Magnitude-based head / FFN pruning | Inference speedup |
| **WANDA pruning** | Weight-and-activation unstructured pruning | State-of-the-art sparsity |

---

## Full pipeline in one tool

```
xlmtec train      → fine-tune with LoRA / QLoRA / DPO / distillation
xlmtec sweep      → Optuna hyperparameter search over lr, batch size, LoRA rank
xlmtec evaluate   → ROUGE, BLEU, perplexity benchmarks
xlmtec export     → save as ONNX, GGUF (llama.cpp), or safetensors
xlmtec predict    → batch inference on JSONL / CSV datasets
xlmtec dashboard  → compare training runs
xlmtec recommend  → AI-assisted config suggestions (Claude / Gemini / GPT-4)
xlmtec hub        → search and browse HuggingFace model hub
xlmtec template   → ready-made configs for sentiment, QA, summarisation, DPO
xlmtec resume     → resume from checkpoint
xlmtec plugin     → extend with custom trainers and providers
xlmtec tui        → interactive terminal UI
```

---

## Installation

```bash
pip install xlmtec              # core CLI only (no GPU libs)
pip install xlmtec[ml]          # + PyTorch, Transformers, PEFT, Accelerate
pip install xlmtec[ml,sweep]    # + Optuna hyperparameter sweep
pip install xlmtec[ml,dpo]      # + TRL for DPO training
pip install xlmtec[full]        # everything
```

See the [Installation guide](installation.md) for GPU setup and platform notes.

---

## Cite xlmtec

If you use xlmtec in your research or project, please cite it:

```bibtex
@software{xlmtec,
  author  = {Rahman, Abdur},
  title   = {xlmtec: Production-Grade LLM Fine-Tuning CLI},
  year    = {2026},
  version = {3.28.0},
  url     = {https://github.com/Abdur-azure/xlmtec},
  license = {MIT}
}
```

A `CITATION.cff` file is included in the repository root for automated citation
by GitHub, Zenodo, and LLM citation tools.

---

## License

MIT — free to use, modify, and distribute.  
[GitHub](https://github.com/Abdur-azure/xlmtec) ·
[PyPI](https://pypi.org/project/xlmtec) ·
[Issues](https://github.com/Abdur-azure/xlmtec/issues)