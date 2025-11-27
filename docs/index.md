# 🤖 Finetune CLI

A comprehensive command-line tool for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation), with automatic ROUGE benchmarking and HuggingFace integration.

[![Build](https://img.shields.io/github/actions/workflow/status/Abdur-azure/finetune_cli/deploy_docs.yml)](https://github.com/Abdur-azure/finetune_cli/actions)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Abdur-azure/finetune_cli/blob/main/LICENSE)

## Overview

This tool simplifies the process of fine-tuning large language models by providing an interactive CLI interface with built-in benchmarking capabilities. Whether you're working with local datasets or HuggingFace repositories, this tool handles the complexity of LoRA configuration, training, and evaluation.

## Key Features

- **🎯 LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning with automatic target module detection
- **📊 Auto-benchmarking**: ROUGE score comparison before and after training to measure improvements
- **🔍 Smart Dataset Loading**: Automatically detects text columns and handles multiple data formats
- **📁 Flexible Data Sources**: Support for local files (JSON, JSONL, CSV, TXT) and HuggingFace datasets
- **🎛️ Selective Loading**: Load specific files from large repositories to optimize memory usage
- **🚀 HuggingFace Integration**: Push fine-tuned models directly to HuggingFace Hub
- **🧠 Auto-detection**: Automatically identifies target modules for any model architecture

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive CLI
python finetune_cli.py
```

The tool will guide you through:

1. Model selection from HuggingFace
2. Dataset loading and preparation
3. Pre-training benchmark
4. LoRA configuration
5. Training process
6. Post-training evaluation
7. Optional upload to HuggingFace Hub

## Why Use This Tool?

- **Simplified Workflow**: No need to write complex training scripts
- **Best Practices Built-in**: Automatically handles tokenization, padding, and data collation
- **Memory Efficient**: LoRA reduces memory requirements significantly
- **Reproducible**: Consistent configuration and benchmarking across experiments
- **Educational**: Learn fine-tuning concepts through interactive prompts

## Getting Started

Check out the [Installation Guide](installation.md) to set up your environment, then follow the [Usage Guide](usage.md) to start fine-tuning your first model.

## Documentation Structure

- **[Installation](installation.md)**: Setup instructions and prerequisites
- **[Usage Guide](usage.md)**: Detailed walkthrough of all features
- **[Configuration](configuration.md)**: Understanding LoRA parameters and training settings
- **[API Reference](api.md)**: Technical documentation of core classes and methods
- **[Examples](examples.md)**: Common use cases and recipes
- **[Troubleshooting](troubleshooting.md)**: Solutions to common issues

## Project Status

This tool is actively maintained and open for contributions. If you encounter any issues or have suggestions, please open an issue on [GitHub](https://github.com/Abdur-azure/finetune_cli/issues).

## License

MIT License - see [LICENSE](https://github.com/Abdur-azure/finetune_cli/blob/main/LICENSE) for details.