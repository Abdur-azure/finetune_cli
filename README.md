# finetune_cli:
A comprehensive CLI application for fine-tuning LLMs with LoRA, benchmarking, and uploading to Hugging Face. This will be a production-ready tool with proper error handling and user-friendly interface.

## 🎯 Key Features:

Model Loading - Supports any HuggingFace model with automatic GPU/CPU detection
Dataset Loading - Load from local files (JSON/CSV) or HuggingFace datasets
LoRA Fine-tuning - Configurable LoRA parameters (rank, alpha, dropout, target modules)
ROUGE Benchmarking - Automatic before/after performance comparison
HuggingFace Upload - Push to existing repos or create new ones

## 📦 Installation:
```
# Install required dependencies
pip install torch transformers datasets peft rouge-score huggingface-hub tqdm pandas
```

## 🚀 Usage:
```
python finetune_cli.py
```

The tool will interactively guide you through:

Model selection (e.g., gpt2, facebook/opt-125m)
Dataset configuration (local file or HF dataset like wikitext)
Pre-training benchmark
LoRA configuration (r, alpha, dropout)
Training parameters (epochs, batch size, learning rate)
Post-training benchmark with comparison
Optional HuggingFace upload

## 📊 Example Workflow:

Enter model: gpt2
Enter dataset: wikitext (config: wikitext-2-raw-v1)
Text column: text
Benchmark on base model
Configure LoRA (r=8, alpha=32)
Train for 3 epochs
Benchmark fine-tuned model
View ROUGE score improvements
Upload to HuggingFace (optional)
