# Installation Guide

## Prerequisites

Before installing the LLM Fine-Tuning CLI Tool, ensure you have the following:

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU (optional but highly recommended)
  - Minimum 8GB VRAM for small models (GPT-2, OPT-125M)
  - 16GB+ VRAM recommended for larger models
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB+ free space for model downloads and checkpoints

### Software Dependencies

- CUDA Toolkit 11.8+ (for GPU acceleration)
- pip package manager
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Abdur-azure/finetune_cli.git
cd finetune_cli
```

### 2. Create Virtual Environment (Recommended)

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using `conda`:

```bash
conda create -n finetune python=3.10
conda activate finetune
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:

- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.35.0` - HuggingFace Transformers library
- `datasets>=2.14.0` - HuggingFace Datasets library
- `peft>=0.7.0` - Parameter-Efficient Fine-Tuning library
- `rouge-score>=0.1.2` - ROUGE metric for evaluation
- `huggingface-hub>=0.19.0` - HuggingFace Hub integration
- `accelerate>=0.24.0` - Distributed training utilities
- Additional utilities (tqdm, pandas, sentencepiece, protobuf)

### 4. Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
Transformers: 4.x.x
CUDA Available: True  # or False if no GPU
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU with CUDA

1. Install NVIDIA drivers for your GPU
2. Install CUDA Toolkit (11.8 or later):
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. Verify CUDA installation:

```bash
nvcc --version
nvidia-smi
```

### PyTorch with CUDA

If you need a specific CUDA version, install PyTorch separately:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## HuggingFace Setup (Optional)

To upload models to HuggingFace Hub, you'll need an account and token:

1. Create account at [HuggingFace](https://huggingface.co/join)
2. Generate token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Login via CLI:

```bash
huggingface-cli login
```

Or set environment variable:

```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

## Troubleshooting Installation

### Issue: CUDA Out of Memory

**Solution**: Install CPU-only PyTorch or use a smaller model:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Module Not Found Errors

**Solution**: Upgrade pip and reinstall dependencies:

```bash
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### Issue: Slow Installation

**Solution**: Use a different PyPI mirror:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Issue: Permission Denied

**Solution**: Install in user space:

```bash
pip install --user -r requirements.txt
```

## Docker Installation (Alternative)

For a containerized environment:

```bash
# Build image
docker build -t finetune-cli .

# Run container
docker run -it --gpus all -v $(pwd):/workspace finetune-cli
```

Create a `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "finetune_cli.py"]
```

## Next Steps

Once installation is complete, proceed to the [Usage Guide](usage.md) to start fine-tuning your first model.

## Updating

To update to the latest version:

```bash
git pull origin main
pip install --upgrade -r requirements.txt
```