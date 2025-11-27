# API Reference

Complete reference for the `LLMFineTuner` class and its methods.

## LLMFineTuner Class

The main class for fine-tuning language models with LoRA.

### Constructor

```python
LLMFineTuner(model_name: str, output_dir: str = "./finetuned_model")
```

**Parameters:**

- `model_name` (str): HuggingFace model identifier (e.g., "gpt2", "facebook/opt-125m")
- `output_dir` (str, optional): Directory to save fine-tuned model. Default: "./finetuned_model"

**Attributes:**

- `model_name` (str): Name of the base model
- `output_dir` (str): Output directory path
- `device` (str): Device for training ("cuda" or "cpu")
- `tokenizer` (AutoTokenizer): HuggingFace tokenizer instance
- `model` (AutoModelForCausalLM): Base model instance
- `peft_model` (PeftModel): LoRA-adapted model instance

**Example:**

```python
from finetune_cli import LLMFineTuner

finetuner = LLMFineTuner(
    model_name="gpt2",
    output_dir="./my_model"
)
```

## Methods

### load_model()

Load the base model and tokenizer from HuggingFace.

```python
def load_model() -> None
```

**Returns:** None

**Side Effects:**

- Initializes `self.tokenizer`
- Initializes `self.model`
- Sets pad_token if not present

**Example:**

```python
finetuner = LLMFineTuner("gpt2")
finetuner.load_model()
```

**Notes:**

- Automatically uses FP16 on CUDA devices
- Sets device_map="auto" for multi-GPU support
- Uses low_cpu_mem_usage for efficient loading

---

### load_dataset_from_source()

Load dataset from local file or HuggingFace Hub.

```python
def load_dataset_from_source(
    dataset_source: str,
    dataset_config: Optional[str] = None,
    split: str = "train",
    num_samples: Optional[int] = None,
    data_files: Optional[str] = None
) -> Dataset
```

**Parameters:**

- `dataset_source` (str): Local file path or HuggingFace dataset name
- `dataset_config` (str, optional): Dataset configuration/subset name
- `split` (str): Dataset split to load. Default: "train"
- `num_samples` (int, optional): Limit number of samples to load
- `data_files` (str, optional): Specific files to load from repository

**Returns:** `Dataset` object

**Supported Formats:**

- Local: `.json`, `.jsonl`, `.csv`, `.txt`
- HuggingFace: Any public dataset

**Example:**

```python
# Load local file
dataset = finetuner.load_dataset_from_source(
    dataset_source="./data.jsonl",
    num_samples=1000
)

# Load HuggingFace dataset
dataset = finetuner.load_dataset_from_source(
    dataset_source="wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="train",
    num_samples=5000
)

# Load specific file from large repo
dataset = finetuner.load_dataset_from_source(
    dataset_source="HuggingFaceH4/ultrachat_200k",
    data_files="data/train_sft-00000-of-00004.parquet",
    num_samples=2000
)
```

---

### detect_text_columns()

Automatically detect text columns in a dataset.

```python
def detect_text_columns(dataset: Dataset) -> List[str]
```

**Parameters:**

- `dataset` (Dataset): Dataset to analyze

**Returns:** List of column names containing text data

**Detection Strategy:**

1. Checks for common text column names
2. Inspects data types of columns
3. Returns all string-type columns

**Common Names Detected:**

- text, content, input, output
- prompt, response, instruction
- question, answer

**Example:**

```python
dataset = finetuner.load_dataset_from_source("./data.jsonl")
text_cols = finetuner.detect_text_columns(dataset)
print(f"Found columns: {text_cols}")
# Output: Found columns: ['prompt', 'response']
```

---

### prepare_dataset()

Tokenize and prepare dataset for training.

```python
def prepare_dataset(
    dataset: Dataset,
    text_columns: Optional[List[str]] = None,
    max_length: int = 512
) -> Tuple[Dataset, List[str]]
```

**Parameters:**

- `dataset` (Dataset): Raw dataset to prepare
- `text_columns` (List[str], optional): Columns to use. Auto-detects if None
- `max_length` (int): Maximum sequence length. Default: 512

**Returns:** Tuple of (tokenized_dataset, text_columns_used)

**Processing Steps:**

1. Auto-detects text columns if not provided
2. Combines multiple columns if present
3. Tokenizes with truncation and padding
4. Removes original columns

**Example:**

```python
dataset = finetuner.load_dataset_from_source("./data.jsonl")

# Auto-detect columns
tokenized, cols = finetuner.prepare_dataset(dataset, max_length=512)

# Manual column specification
tokenized, cols = finetuner.prepare_dataset(
    dataset,
    text_columns=["prompt", "response"],
    max_length=256
)
```

---

### get_target_modules()

Automatically detect target modules for LoRA based on model architecture.

```python
def get_target_modules() -> Union[List[str], str]
```

**Returns:** List of module names or "all-linear" as fallback

**Detection Patterns:**

- GPT-2 style: `["c_attn", "c_proj"]`
- Transformer style: `["q_proj", "v_proj", "k_proj", "o_proj"]`
- Alternative: `["query", "value", "key", "dense"]`

**Example:**

```python
finetuner.load_model()
modules = finetuner.get_target_modules()
print(f"Detected modules: {modules}")
# Output: Detected modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj']
```

**Notes:**

- Automatically called by `setup_lora()`
- Fallback to "all-linear" if no pattern matches

---

### setup_lora()

Configure and apply LoRA to the model.

```python
def setup_lora(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> None
```

**Parameters:**

- `r` (int): LoRA rank. Default: 8
- `lora_alpha` (int): LoRA alpha scaling. Default: 32
- `lora_dropout` (float): Dropout probability. Default: 0.1
- `target_modules` (List[str], optional): Modules to apply LoRA. Auto-detects if None

**Returns:** None

**Side Effects:**

- Creates `self.peft_model` with LoRA adapters
- Prints trainable parameter statistics

**Example:**

```python
finetuner.load_model()

# Default configuration
finetuner.setup_lora()

# Custom configuration
finetuner.setup_lora(
    r=16,
    lora_alpha=64,
    lora_dropout=0.15,
    target_modules=["q_proj", "v_proj"]
)
```

**Output:**

```
trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.2370
```

---

### train()

Train the model with LoRA adapters.

```python
def train(
    train_dataset: Dataset,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
) -> None
```

**Parameters:**

- `train_dataset` (Dataset): Tokenized training dataset
- `num_epochs` (int): Number of training epochs. Default: 3
- `batch_size` (int): Per-device batch size. Default: 4
- `learning_rate` (float): Learning rate. Default: 2e-4

**Returns:** None

**Side Effects:**

- Trains the model
- Saves checkpoints to output_dir
- Saves final model and tokenizer

**Training Configuration:**

- Gradient accumulation: 4 steps
- FP16: Enabled on CUDA
- Logging: Every 10 steps
- Save strategy: Per epoch

**Example:**

```python
finetuner.load_model()
dataset = finetuner.load_dataset_from_source("./data.jsonl")
tokenized, _ = finetuner.prepare_dataset(dataset)
finetuner.setup_lora()

finetuner.train(
    train_dataset=tokenized,
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-4
)
```

---

### benchmark()

Benchmark model performance using ROUGE scores.

```python
def benchmark(
    test_prompts: List[str],
    use_finetuned: bool = False
) -> Dict[str, float]
```

**Parameters:**

- `test_prompts` (List[str]): List of prompts to evaluate
- `use_finetuned` (bool): Use fine-tuned model. Default: False (uses base model)

**Returns:** Dictionary with ROUGE scores

**Metrics Computed:**

- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence

**Example:**

```python
test_prompts = [
    "What is machine learning?",
    "Explain neural networks.",
    "Define artificial intelligence."
]

# Benchmark base model
base_scores = finetuner.benchmark(test_prompts, use_finetuned=False)

# After training
finetuned_scores = finetuner.benchmark(test_prompts, use_finetuned=True)

# Compare
for metric in base_scores:
    improvement = (finetuned_scores[metric] - base_scores[metric]) / base_scores[metric] * 100
    print(f"{metric}: {improvement:+.2f}% improvement")
```

**Output Format:**

```python
{
    'rouge1': 0.3245,
    'rouge2': 0.2134,
    'rougeL': 0.2987
}
```

---

### upload_to_huggingface()

Upload fine-tuned model to HuggingFace Hub.

```python
def upload_to_huggingface(
    repo_name: str,
    token: Optional[str] = None,
    create_new: bool = False,
    private: bool = False
) -> None
```

**Parameters:**

- `repo_name` (str): Repository name (format: "username/repo-name")
- `token` (str, optional): HuggingFace API token. Uses cached login if None
- `create_new` (bool): Create new repository. Default: False
- `private` (bool): Make repository private. Default: False

**Returns:** None

**Requirements:**

- Valid HuggingFace token with write permissions
- Trained model in output_dir

**Example:**

```python
# After training
finetuner.upload_to_huggingface(
    repo_name="myusername/gpt2-finetuned",
    token="hf_xxxxxxxxxxxxx",
    create_new=True,
    private=False
)
```

**Successful Upload:**

```
✅ Model uploaded successfully to: https://huggingface.co/myusername/gpt2-finetuned
```

## Usage Patterns

### Complete Training Pipeline

```python
from finetune_cli import LLMFineTuner

# Initialize
finetuner = LLMFineTuner("gpt2", "./my_model")

# Load model
finetuner.load_model()

# Load and prepare data
dataset = finetuner.load_dataset_from_source(
    "./data.jsonl",
    num_samples=5000
)
tokenized, cols = finetuner.prepare_dataset(dataset, max_length=512)

# Pre-training benchmark
test_prompts = ["Sample prompt 1", "Sample prompt 2"]
base_scores = finetuner.benchmark(test_prompts, use_finetuned=False)

# Setup LoRA
finetuner.setup_lora(r=8, lora_alpha=32, lora_dropout=0.1)

# Train
finetuner.train(tokenized, num_epochs=3, batch_size=4, learning_rate=2e-4)

# Post-training benchmark
finetuned_scores = finetuner.benchmark(test_prompts, use_finetuned=True)

# Upload
finetuner.upload_to_huggingface(
    "username/model-name",
    create_new=True
)
```

### Loading Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./my_model")

# Inference
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Merging Adapters (Optional)

```python
from peft import PeftModel

# Load model with adapters
model = PeftModel.from_pretrained(base_model, "./my_model")

# Merge adapters into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

## Helper Functions

### get_user_input()

Interactive prompt with optional default value.

```python
def get_user_input(prompt: str, default: Optional[str] = None) -> str
```

**Example:**

```python
model_name = get_user_input("Enter model name", "gpt2")
# Prompt: Enter model name [gpt2]:
```

## Type Definitions

```python
from typing import Optional, Dict, List, Tuple
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
```

## Error Handling

The tool includes comprehensive error handling:

```python
try:
    finetuner.train(dataset)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Reduce batch size or sequence length")
    raise
except KeyboardInterrupt:
    print("Training interrupted")
    sys.exit(0)
```

## Next Steps

- See [Usage Guide](usage.md) for CLI workflow
- Check [Examples](examples.md) for practical use cases
- Review [Configuration](configuration.md) for parameter tuning