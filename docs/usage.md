# Configuration Guide

This guide explains all configuration parameters and how to optimize them for your use case.

## LoRA Parameters

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds trainable rank decomposition matrices to existing weights.

### Rank (r)

The rank of the low-rank matrices added to model layers.

**What it controls**: The capacity of the adapter to learn new patterns.

**Values and Use Cases:**

| Rank | Trainable Params | Memory | Use Case |
|------|-----------------|---------|-----------|
| 4 | ~0.1-0.5M | Low | Quick experiments, simple tasks |
| 8 | ~0.5-2M | Moderate | General purpose, balanced performance |
| 16 | ~2-8M | Higher | Complex tasks, significant adaptation |
| 32 | ~8-32M | High | Maximum quality, specialized domains |

**Choosing rank:**

```python
# Simple classification or entity extraction
r = 4

# General text generation or summarization
r = 8

# Complex reasoning or domain adaptation
r = 16

# Specialized medical/legal/technical domains
r = 32
```

**Trade-offs:**

- ✅ Higher rank: Better adaptation, handles complex patterns
- ❌ Higher rank: More memory, longer training, risk of overfitting

### Alpha (α)

Scaling factor applied to LoRA weights.

**What it controls**: The influence of LoRA updates relative to pre-trained weights.

**Formula**: `scaling = alpha / r`

**Recommended values:**

- Standard: `alpha = 2 × r` (e.g., r=8, alpha=16)
- Conservative: `alpha = r` (less aggressive updates)
- Aggressive: `alpha = 4 × r` (stronger adaptation)

**Examples:**

```python
# Conservative (maintains more of base model)
r = 8, alpha = 8    # scaling = 1.0

# Standard (recommended)
r = 8, alpha = 16   # scaling = 2.0

# Aggressive (stronger fine-tuning)
r = 8, alpha = 32   # scaling = 4.0
```

**When to adjust:**

- Increase alpha if model isn't adapting enough
- Decrease alpha if model forgets pre-trained knowledge

### Dropout

Regularization technique to prevent overfitting.

**What it controls**: Probability of randomly disabling LoRA parameters during training.

**Values:**

- `0.0`: No dropout (risk of overfitting on small datasets)
- `0.05`: Light regularization (large, diverse datasets)
- `0.1`: Standard regularization (general purpose)
- `0.2`: Heavy regularization (small or noisy datasets)

**Choosing dropout:**

```python
# Large dataset (> 50k samples), clean data
dropout = 0.05

# Medium dataset (5k-50k samples)
dropout = 0.1

# Small dataset (< 5k samples) or noisy data
dropout = 0.2
```

### Target Modules

Specifies which model layers to apply LoRA to.

**Auto-detection**: The tool automatically identifies optimal target modules.

**Common patterns:**

```python
# Attention layers only (memory efficient)
["q_proj", "v_proj"]

# Full attention (recommended)
["q_proj", "k_proj", "v_proj", "o_proj"]

# Attention + MLP (maximum adaptation)
["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
```

**Manual override** (advanced):

You can modify the code to specify custom targets:

```python
target_modules = ["q_proj", "v_proj"]  # Attention queries and values only
```

## Training Parameters

### Number of Epochs

Complete passes through the training dataset.

**Guidelines by dataset size:**

| Dataset Size | Recommended Epochs |
|--------------|-------------------|
| < 1,000 samples | 5-10 |
| 1,000-5,000 | 3-7 |
| 5,000-50,000 | 3-5 |
| > 50,000 | 1-3 |

**Signs of:**

- **Underfitting**: Loss still decreasing, ROUGE scores improving
  - Solution: Increase epochs
  
- **Overfitting**: Training loss decreases but validation loss increases
  - Solution: Decrease epochs, increase dropout

### Batch Size

Number of samples processed before updating model weights.

**Memory constraints:**

| GPU VRAM | Model Size | Max Batch Size |
|----------|-----------|----------------|
| 8GB | Small (< 500M params) | 2-4 |
| 12GB | Small-Medium | 4-8 |
| 16GB | Medium (1-3B params) | 4-8 |
| 24GB | Large (7B params) | 8-16 |

**Effective batch size** with gradient accumulation:

```python
# Config in training_args
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
# Effective batch size = 4 × 4 = 16
```

**Tips:**

- Start with batch_size=4 and adjust based on memory
- Smaller batches = more frequent updates, noisier gradients
- Larger batches = more stable gradients, better generalization

### Learning Rate

Step size for weight updates.

**Common values:**

| Learning Rate | Use Case |
|--------------|----------|
| 1e-5 | Very conservative, large models |
| 5e-5 | Conservative, stable training |
| 1e-4 | Moderate, good starting point |
| 2e-4 | Standard for LoRA (recommended) |
| 5e-4 | Aggressive, small models |
| 1e-3 | Very aggressive, risk of instability |

**Learning rate schedule:**

The tool uses a constant learning rate. For advanced use, you can modify to use:

- Linear decay
- Cosine decay
- Warmup + decay

**Signs of poor learning rate:**

- **Too high**: Loss oscillates or diverges, NaN values
  - Solution: Reduce by 50% (e.g., 2e-4 → 1e-4)

- **Too low**: Loss decreases very slowly
  - Solution: Increase by 2x (e.g., 1e-4 → 2e-4)

### Maximum Sequence Length

Maximum number of tokens per training sample.

**Choosing max length:**

```python
# Short texts (tweets, titles, Q&A)
max_length = 128

# Medium texts (paragraphs, summaries)
max_length = 512

# Long texts (articles, documents)
max_length = 1024

# Very long texts (research papers)
max_length = 2048
```

**Trade-offs:**

- ✅ Longer sequences: Better context understanding
- ❌ Longer sequences: Quadratic memory increase, slower training

**Memory impact:**

```
Memory ∝ batch_size × max_length²
```

Doubling max_length quadruples memory usage!

## Advanced Configuration

### Gradient Accumulation

Simulate larger batch sizes without more memory.

**In the code** (line 222):

```python
gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps
```

**Calculation:**

```
Effective Batch Size = batch_size × gradient_accumulation_steps × num_gpus
```

### Mixed Precision (FP16)

Use 16-bit floating point for faster training and less memory.

**Automatically enabled** when CUDA is available:

```python
fp16 = self.device == "cuda"  # Line 228
```

**Benefits:**

- 50% less memory
- 2-3x faster training
- Minimal accuracy loss

### Model Quantization

For very large models, you can enable quantization:

```python
# Add to model loading (line 59)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # Quantize to 8-bit
    device_map="auto"
)
```

## Configuration Recipes

### Recipe 1: Quick Experimentation

```
Samples: 1000
Max Length: 256
LoRA r: 4
LoRA alpha: 16
Dropout: 0.1
Epochs: 3
Batch Size: 4
Learning Rate: 2e-4
```

**Best for**: Testing ideas, rapid iteration

### Recipe 2: Balanced Quality

```
Samples: 10000
Max Length: 512
LoRA r: 8
LoRA alpha: 32
Dropout: 0.1
Epochs: 3
Batch Size: 8
Learning Rate: 2e-4
```

**Best for**: Production models, general tasks

### Recipe 3: Maximum Quality

```
Samples: 50000+
Max Length: 1024
LoRA r: 16
LoRA alpha: 64
Dropout: 0.1
Epochs: 3
Batch Size: 8
Learning Rate: 1e-4
```

**Best for**: Specialized domains, publication-quality results

### Recipe 4: Memory-Constrained

```
Samples: 5000
Max Length: 256
LoRA r: 4
LoRA alpha: 16
Dropout: 0.1
Epochs: 5
Batch Size: 2
Learning Rate: 2e-4
```

**Best for**: Limited GPU memory (< 8GB)

## Optimization Tips

1. **Start conservative**: Use lower rank, smaller batch, fewer epochs
2. **Monitor metrics**: Watch loss curves and ROUGE scores
3. **Iterate gradually**: Increase one parameter at a time
4. **Save checkpoints**: Keep best performing configurations
5. **Profile memory**: Use `nvidia-smi` to track GPU usage

## Next Steps

- See practical [Examples](examples.md)
- Understand [Troubleshooting](troubleshooting.md) common issues
- Explore [API Reference](api.md) for programmatic usage