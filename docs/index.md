# 🤖 LLM Fine-Tuning CLI Tool

Welcome to the documentation for the **LLM Fine-Tuning CLI Tool** - a comprehensive command-line application for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **🎯 LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- **📊 Auto-benchmarking**: ROUGE score comparison before/after training
- **🔍 Smart Dataset Loading**: Auto-detect columns and handle multiple formats
- **📁 Flexible Data Sources**: Local files or HuggingFace datasets
- **🎛️ Selective Loading**: Load specific files from large repositories
- **🚀 HuggingFace Upload**: Push models directly to HuggingFace Hub

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the CLI tool
python finetune_cli.py
```

## 📚 Documentation Sections

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Getting Started__

    ---

    Learn how to install and run your first fine-tuning job

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Comprehensive guide to all features and configurations

    [:octicons-arrow-right-24: User Guide](user-guide/overview.md)

-   :material-code-braces:{ .lg .middle } __Examples__

    ---

    Real-world examples and use cases

    [:octicons-arrow-right-24: Examples](examples/basic.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Detailed API documentation for developers

    [:octicons-arrow-right-24: API Docs](api/llm-finetuner.md)

</div>

## 💡 Why Use This Tool?

- **Easy to Use**: Interactive CLI guides you through every step
- **Flexible**: Works with any HuggingFace model and dataset
- **Efficient**: LoRA fine-tuning uses minimal GPU memory
- **Comprehensive**: Includes benchmarking and deployment
- **Production-Ready**: Robust error handling and validation

## 🎯 Use Cases

- Fine-tune models for specific domains (medical, legal, technical)
- Adapt pre-trained models to your custom data
- Create specialized chatbots and assistants
- Improve model performance on specific tasks
- Experiment with different training configurations

## 🤝 Community

- [GitHub Repository](https://github.com/Abdur-azure/finetune_cli)
- [Report Issues](https://github.com/Abdur-azure/finetune_cli/issues)
- [Contribute](contributing.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Abdur-azure/finetune_cli/blob/main/LICENSE) file for details.