# Export Formats

Export a trained model to safetensors, ONNX, or GGUF for deployment.

---

## Quick start

```bash
# Export to safetensors (always available, no extra install)
xlmtec export output/run1 --format safetensors --output exported/

# Export to ONNX with fp16 quantisation
xlmtec export output/run1 --format onnx --quantize fp16 --output exported/

# Export to GGUF with 4-bit quantisation
xlmtec export output/run1 --format gguf --quantize q4_0 --output exported/

# Preview without writing any files
xlmtec export output/run1 --format safetensors --dry-run
```

---

## Supported formats

| Format | Install required | Quantise options | Best for |
|--------|-----------------|-----------------|----------|
| `safetensors` | none | — | Safe storage, fast loading |
| `onnx` | `pip install xlmtec[onnx]` | `fp32`, `fp16`, `int8` | Cross-platform inference, ONNX Runtime |
| `gguf` | `pip install xlmtec[gguf]` + llama.cpp | `q4_0`, `q4_1`, `q8_0`, `f16` | llama.cpp / LM Studio / Ollama |

---

## Installing extras

```bash
# ONNX export
pip install "xlmtec[onnx]"

# GGUF export
pip install "xlmtec[gguf]"
# Also requires llama.cpp — clone and build, then pass --llama-cpp-dir
git clone https://github.com/ggerganov/llama.cpp
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | required | `safetensors`, `onnx`, or `gguf` |
| `--output` | `exported/` | Directory to write exported files |
| `--quantize` | none | Quantisation type (format-specific) |
| `--llama-cpp-dir` | auto-detect | Path to llama.cpp repo root (GGUF only) |
| `--dry-run` | off | Validate only — do not write files |

---

## Dry run

`--dry-run` validates your model directory and options without loading the model or writing any files. Useful to confirm everything is correct before a long export:

```bash
xlmtec export output/run1 --format onnx --quantize fp16 --dry-run
```

```
┌─ Export Plan ─────────────────────────────────────┐
│  Source   : output/run1                           │
│  Format   : onnx                                  │
│  Quantise : fp16                                  │
│  Output   : exported/                             │
│  Deps     : optimum ✓  onnx ✓  onnxruntime ✓     │
└───────────────────────────────────────────────────┘
Remove --dry-run to export.
```

---

## GGUF workflow

GGUF export calls llama.cpp's `convert_hf_to_gguf.py` script:

```bash
# 1. Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 2. Export with xlmtec
xlmtec export output/run1 \
  --format gguf \
  --quantize q4_0 \
  --llama-cpp-dir ~/llama.cpp \
  --output exported/
```

The resulting `.gguf` file can be loaded directly by LM Studio, Ollama, or llama.cpp's `main` binary.

---

## Merge adapter before export

If your run used LoRA/QLoRA, merge the adapter into the base model first:

```bash
xlmtec merge output/run1 --output merged/
xlmtec export merged/ --format gguf --quantize q4_0 --output exported/
```