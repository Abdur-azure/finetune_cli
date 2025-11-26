# 🤖 FinetuneCLI — Modular LLM Fine-Tuning Toolkit

A comprehensive, modular command-line tool for fine-tuning Large Language Models with support for multiple parameter-efficient techniques including LoRA, QLoRA, and Prompt Tuning. Features an interactive 12-step workflow, automatic benchmarking, and HuggingFace integration.

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

![Version](https://img.shields.io/badge/version-0.2.1-blue.svg)

## ✨ Features

### 🎯 Fine-Tuning Techniques
- **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning
- **QLoRA (Quantized LoRA)**: 4-bit/8-bit quantization for memory-efficient training
- **Prompt Tuning**: Soft prompt learning with minimal parameters
- **Auto-Detection**: Automatically identifies target modules for any model architecture (GPT-2, Llama, etc.)

### 🚀 Interactive Workflow
- **Unified CLI**: 12-step interactive wizard (`finetune-cli finetune run`)
- **Hierarchical Menus**: Organized technique selection (Quantization/Distillation/Pruning)
- **Smart Defaults**: Pre-configured parameters with scientific notation support

### 📊 Benchmarking & Evaluation
- **ROUGE Metrics**: Automatic before/after comparison
- **Multiple Metrics**: Support for BLEU, BERTScore, Perplexity, and more
- **Real-time Feedback**: Progress tracking during training

### 📁 Data Handling
- **Flexible Sources**: Local files (JSON, JSONL, CSV, TXT) or HuggingFace datasets
- **Smart Loading**: Auto-detect columns and handle multiple formats
- **Selective Loading**: Load specific files from large repositories
- **Dataset Sampling**: Efficient sampling for benchmarking

### � Additional Features
- **HuggingFace Upload**: Push models directly to HuggingFace Hub
- **Modular Architecture**: Clean separation of concerns
- **Comprehensive Docs**: Full documentation with examples and troubleshooting


## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
<<<<<<< Updated upstream
- CUDA-capable GPU (optional, but recommended)
=======
- CUDA-capable GPU (optional, but recommended for QLoRA)

### Install from Source
```bash
# Clone the repository
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install in editable mode
pip install -e .
```

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
=======

### Interactive Workflow (Recommended)
```bash
finetune-cli finetune run
```

This launches the interactive 12-step wizard:
1. **Model Selection**: Choose any HuggingFace model
2. **Dataset Selection**: Local or HuggingFace datasets
3. **Technique Selection**: LoRA, QLoRA, or Prompt Tuning
4. **Benchmark Selection**: ROUGE, BLEU, etc.
5. **Output Directory**: Where to save the model
6. **Base Model Benchmarking**: Evaluate before training
7. **Parameter Configuration**: Set technique-specific params
8. **Fine-Tuning**: Train the model
9. **Fine-Tuned Benchmarking**: Evaluate after training
10. **Performance Comparison**: Before/after metrics
11. **Model Saving**: Confirmation and file listing
12. **HuggingFace Upload**: Optional cloud deployment

### Alternative: Direct Training
```bash
# LoRA training
finetune-cli train lora --model gpt2 --dataset ./data.jsonl

# Benchmarking
finetune-cli benchmark --model ./finetuned_model --dataset ./test.jsonl

# Quantization
finetune-cli quantize --help
```

### Python Module
```bash
python -m finetunecli
```

## 📚 Usage Examples

### Example 1: QLoRA Fine-tuning with GPT-2
```bash
finetune-cli finetune run

# Follow prompts:
Model name: gpt2
Dataset path: ./train1000.jsonl
Technique: Quantization → QLoRA
Benchmark: ROUGE
LoRA rank (r): 8
LoRA alpha: 16
Quantization bits: 4
Quantization type: nf4
Epochs: 1
Batch size: 1
Learning rate: 2e-4
```

### Example 2: Prompt Tuning with Custom Initialization
```bash
finetune-cli finetune run

# Follow prompts:
Model name: facebook/opt-125m
Dataset: wikitext
Technique: Quantization → Prompt Tuning
Number of virtual tokens: 20
Initialization: TEXT
Init text: "Summarize the following text:"
Epochs: 5
Learning rate: 3e-2
```

### Example 3: LoRA with HuggingFace Dataset
```bash
finetune-cli finetune run

# Follow prompts:
Model name: gpt2
Dataset: squad
Technique: Quantization → LoRA
LoRA r: 16
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
=======
## 🎯 Technique Parameter Guides

### LoRA Parameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **r (rank)** | Adapter size | 4 (light), 8 (balanced), 16 (strong), 32 (heavy) |
| **alpha** | Scaling factor | 2x rank (16, 32, 64) |
| **dropout** | Regularization | 0.05 (low), 0.1 (balanced), 0.2 (high) |
| **learning rate** | Optimizer step size | 2e-4 (default), 1e-4 to 5e-4 |

### QLoRA Parameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **bits** | Quantization precision | 4 (recommended), 8 |
| **quant_type** | Quantization method | nf4 (Normal Float 4), fp4 |
| **r, alpha, dropout** | Same as LoRA | See LoRA table |
| **batch_size** | Smaller due to quantization | 1-2 |

### Prompt Tuning Parameters
| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| **num_virtual_tokens** | Soft prompt length | 8-100 (20 default) |
| **init_method** | Initialization strategy | TEXT, RANDOM |
| **init_text** | Seed text for initialization | Task-specific prompt |
| **learning rate** | Higher than LoRA | 1e-2 to 5e-2 (3e-2 default) |

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

### Dataset Structure
Your dataset should have `input` and `output` fields:
```json
{"input": "Question or prompt", "output": "Expected response"}
```

## 📁 Project Structure
```
finetune_cli/
├── finetunecli/              # Main package
│   ├── __init__.py          # Package entry point with app
│   ├── cli/                 # CLI commands
│   │   ├── unified_cli.py   # Interactive workflow
│   │   ├── train_cli.py     # Training commands
│   │   ├── benchmark_cli.py # Benchmarking
│   │   └── quantize_cli.py  # Quantization
│   ├── config/              # Configuration classes
│   ├── quantization/        # Fine-tuning techniques
│   │   ├── lora/           # LoRA implementation
│   │   ├── qlora/          # QLoRA implementation
│   │   └── prompt_tuning/  # Prompt Tuning
│   ├── benchmarking/        # Evaluation metrics
│   ├── training/            # Training utilities
│   └── utils/              # Helper functions
├── tests/                   # Test suite
│   ├── test_benchmark_fix.py
│   └── test_qlora_modules.py
├── docs/                    # Documentation
├── finetune_cli.py         # Legacy standalone script
├── finetunecli.py          # Package wrapper
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🐛 Troubleshooting

### Command Not Found: finetune-cli
**Solution**: Activate your virtual environment or use the full path:
```bash
# Activate venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Or use full path
.\venv\Scripts\finetune-cli  # Windows
./venv/bin/finetune-cli      # Linux/Mac

# Or use Python module
python -m finetunecli
```

### ImportError: cannot import name 'app'
**Solution**: Reinstall the package:
```bash
pip install -e .
```

### CUDA Out of Memory
**Solutions**:
- Use QLoRA with 4-bit quantization
- Reduce batch size to 1
- Decrease max sequence length
- Use smaller LoRA rank (r=4)
- Limit number of samples

### QLoRA Target Modules Error
**Fixed in v0.2.1**: The system now auto-detects target modules for any model architecture.

### Benchmark AttributeError
**Fixed in v0.2.1**: Dataset sampling now correctly handles HuggingFace Datasets.

For more issues, see [docs/troubleshooting/common-issues.md](docs/troubleshooting/common-issues.md)

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

2. When prompted in the workflow, enter your token
3. Or login via CLI:
>>>>>>> Stashed changes
```bash
huggingface-cli login
```

## 📈 Performance Tips

1. **Start Small**: Test with 1000 samples before full training
2. **Monitor Memory**: Watch GPU memory usage
3. **Adjust LoRA Rank**: Higher rank = better adaptation but more memory
4. **Use Gradient Accumulation**: Effective larger batch sizes
5. **Select Relevant Data**: Quality > Quantity

2. **Use QLoRA**: For large models, 4-bit quantization saves memory
3. **Monitor Memory**: Watch GPU memory usage
4. **Adjust LoRA Rank**: Higher rank = better adaptation but more memory
5. **Scientific Notation**: Use `2e-4` format for learning rates
6. **Auto-Detection**: Let the system find target modules automatically

## 🧪 Testing

Run all tests:
```bash
python -m unittest discover tests
```

Run specific tests:
```bash
python -m unittest tests.test_benchmark_fix
python -m unittest tests.test_qlora_modules
```

## 📝 What's New in v0.2.1

### Bug Fixes
- ✅ Fixed `ImportError: cannot import name 'app'`
- ✅ Fixed `finetune-cli` command not found issue
- ✅ Fixed `AttributeError` in benchmarking with HuggingFace Datasets
- ✅ Fixed QLoRA target module detection for non-Llama models

### Improvements
- 🎯 Auto-detection of target modules for any model architecture
- 📊 Better dataset sampling for benchmarking
- 🔢 Scientific notation support for learning rate input
- 📁 Organized test suite in `tests/` directory
- 📚 Comprehensive documentation updates

See [docs/changelog.md](docs/changelog.md) for full changelog.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - feel free to use this tool for any purpose.

## 🙏 Acknowledgments

- Built with [Transformers](https://github.com/huggingface/transformers)
- LoRA implementation from [PEFT](https://github.com/huggingface/peft)
- Evaluation using [ROUGE Score](https://github.com/google-research/google-research/tree/master/rouge)

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

- Quantization with [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## 📧 Support

For issues, questions, or suggestions:
- 📖 Check the [documentation](docs/)
- 🐛 Open an [issue on GitHub](https://github.com/Abdur-azure/finetune_cli/issues)
- 💬 Start a [discussion](https://github.com/Abdur-azure/finetune_cli/discussions)

## 📚 Documentation

Full documentation is available at: [https://abdur-azure.github.io/finetune_cli/](https://abdur-azure.github.io/finetune_cli/)

---

Made with ❤️ for the AI community
