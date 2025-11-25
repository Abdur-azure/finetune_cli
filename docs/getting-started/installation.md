# Installation

This guide will help you set up the LLM Fine-Tuning CLI Tool on your system.

## Prerequisites

Before installing, ensure you have:

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) CUDA-capable GPU for faster training

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
```

### 2. Create Virtual Environment (Recommended)

=== "Linux/macOS"

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

=== "Windows"

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python finetune_cli.py --help
```

## GPU Setup (Optional)

For CUDA support:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

!!! warning "Common Issues"
    
    **Issue**: `ModuleNotFoundError`
    
    **Solution**: Ensure all dependencies are installed:
    ```bash
    pip install -r requirements.txt
    ```

!!! tip "GPU Not Detected"
    
    The tool will automatically fall back to CPU if no GPU is detected. To force GPU usage:
    ```bash
    export CUDA_VISIBLE_DEVICES=0
    ```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Your First Fine-tune](first-finetune.md)
