# 🤖 Finetune CLI Tool

A comprehensive command-line tool for fine-tuning Large Language Models using LoRA (Low-Rank Adaptation), with automatic ROUGE benchmarking and HuggingFace integration.

![Build](https://img.shields.io/github/actions/workflow/status/Abdur-azure/finetune_cli/deploy_docs.yml)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Stars](https://img.shields.io/github/stars/Abdur-azure/finetune_cli)
![Issues](https://img.shields.io/github/issues/Abdur-azure/finetune_cli)

## ✨ Features

- 🎯 **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning
- 📊 **Auto-benchmarking**: ROUGE score comparison before/after training
- 🔍 **Smart Dataset Loading**: Auto-detect columns and handle multiple formats
- 📁 **Flexible Data Sources**: Local files (JSON, JSONL, CSV, TXT) or HuggingFace datasets
- 🎛️ **Selective Loading**: Load specific files from large repositories
- 🚀 **HuggingFace Upload**: Push models directly to HuggingFace Hub
- 🧠 **Auto-detection**: Automatically finds target modules for any model architecture

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start
```bash
python finetune_cli.py
```

The tool will guide you through an interactive setup:

1. **Model Selection**: Choose any HuggingFace model (e.g., `gpt2`, `facebook/opt-125m`)
2. **Dataset Loading**: Load from local files or HuggingFace datasets
3. **Pre-training Benchmark**: Automatic ROUGE scoring on base model
4. **LoRA Configuration**: Set rank, alpha, and dropout parameters
5. **Training**: Fine-tune with custom hyperparameters
6. **Post-training Benchmark**: Compare performance improvements
7. **Upload**: Optionally push to HuggingFace Hub

## 📚 Usage Examples

### Example 1: Fine-tune GPT-2 on Local Data
```bash
python finetune_cli.py

# Follow prompts:
Model name: gpt2
Dataset path: ./my_data.jsonl
Limit samples: yes
Number of samples: 1000
Max sequence length: 512
LoRA r: 8
LoRA alpha: 32
Epochs: 3
```

### Example 2: Fine-tune with HuggingFace Dataset
```bash
python finetune_cli.py

# Follow prompts:
Model name: facebook/opt-125m
Dataset name: wikitext
Dataset config: wikitext-2-raw-v1
Limit samples: yes
Number of samples: 5000
```

### Example 3: Load Specific File from Large Repository
```bash
python finetune_cli.py

# For repositories with multiple files:
Dataset name: HuggingFaceH4/ultrachat_200k
Load specific file: yes
File path: data/train_sft-00000-of-00004.parquet
Number of samples: 2000
```

## 📊 Supported Dataset Formats

### Local Files
- **JSON**: Standard JSON format
- **JSONL**: JSON Lines (one object per line)
- **CSV**: Comma-separated values
- **TXT**: Plain text (one sample per line)

### HuggingFace Datasets
- Any public HuggingFace dataset
- Specific file selection from large repositories
- Multiple split support (train/test/validation)

## 🎯 LoRA Parameter Guide

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **r (rank)** | Adapter size | 4 (light), 8 (balanced), 16 (strong), 32 (heavy) |
| **alpha** | Scaling factor | 2x rank (16, 32, 64) |
| **dropout** | Regularization | 0.05 (low), 0.1 (balanced), 0.2 (high) |

## 🔑 HuggingFace Upload

To upload models to HuggingFace:

1. Get your token from: https://huggingface.co/settings/tokens
2. When prompted, enter your token or login via CLI:
```bash
   huggingface-cli login
```

## 📁 Project Structure
```
llm-finetune-cli/
├── finetune_cli.py       # Main application
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── .gitignore           # Git ignore rules
```

## ⚙️ Configuration Options

### Training Parameters
- **Epochs**: Number of training iterations
- **Batch Size**: Samples per gradient update
- **Learning Rate**: Step size for optimization
- **Max Length**: Maximum sequence length

### LoRA Parameters
- **r**: Rank of adaptation matrices
- **alpha**: Scaling factor for LoRA weights
- **dropout**: Dropout probability for regularization

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Decrease max sequence length
- Use smaller LoRA rank (r)
- Limit number of samples

### Module Not Found
```bash
pip install --upgrade -r requirements.txt
```

### Authentication Error (HuggingFace)
```bash
huggingface-cli login
```

## 📈 Performance Tips

1. **Start Small**: Test with 1000 samples before full training
2. **Monitor Memory**: Watch GPU memory usage
3. **Adjust LoRA Rank**: Higher rank = better adaptation but more memory
4. **Use Gradient Accumulation**: Effective larger batch sizes
5. **Select Relevant Data**: Quality > Quantity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this tool for any purpose.

## 🙏 Acknowledgments

- Built with [Transformers](https://github.com/huggingface/transformers)
- LoRA implementation from [PEFT](https://github.com/huggingface/peft)
- Evaluation using [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge)

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

Made with ❤️ for the AI community
