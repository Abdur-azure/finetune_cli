# Installation

## Requirements

- Python 3.10 or higher
- pip

---

## Install from PyPI

### Lightweight install (no ML dependencies)

Good for using `ai-suggest`, `hub`, and `config validate` without downloading PyTorch:

```bash
pip install xlmtec
```

### With training support

```bash
pip install xlmtec[ml]
```

Installs: `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`

### With AI provider support

```bash
pip install xlmtec[claude]    # Anthropic Claude
pip install xlmtec[gemini]    # Google Gemini
pip install xlmtec[codex]     # OpenAI GPT
pip install xlmtec[ai]        # All three providers
```

### Everything

```bash
pip install xlmtec[full]
```

---

## Install from source

```bash
git clone https://github.com/Abdur-azure/xlmtec.git
cd xlmtec
pip install -e ".[full,dev]"
```

---

## Verify installation

```bash
xlmtec --version
xlmtec --help
```

---

## GPU setup (optional)

xlmtec works on CPU but training is significantly faster with a CUDA GPU.

```bash
# Check if CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

For a specific CUDA version:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## HuggingFace token (optional)

Required only for uploading models or accessing private repos:

```bash
huggingface-cli login
# or
export HF_TOKEN=hf_...
```