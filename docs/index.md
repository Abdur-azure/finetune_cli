# Welcome to Finetune CLI

**Finetune CLI** is a powerful, modular, and interactive command-line tool for fine-tuning Large Language Models (LLMs).

## Key Features

*   **Unified Workflow**: A guided 12-step wizard for end-to-end fine-tuning.
*   **Multiple Techniques**: Support for **LoRA**, **QLoRA**, and **Prompt Tuning**.
*   **Benchmarking**: Built-in evaluation with **ROUGE** and other metrics.
*   **Model Support**: Works with any HuggingFace Causal LM (GPT-2, Llama 2, Mistral, etc.).
*   **Memory Efficient**: Run 7B+ models on consumer hardware with 4-bit quantization.

## Getting Started

Install the package:

```bash
pip install .
```

Run the interactive wizard:

```bash
finetune-cli finetune run
```

Check out the [Quick Start](getting-started/quickstart.md) guide to begin!