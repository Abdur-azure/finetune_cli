# API Reference

Python API for programmatic use. All public classes are importable from their respective subpackages.

---

## Configuration — `finetune_cli.core.config`

### `ConfigBuilder`

Fluent builder for constructing a validated `PipelineConfig`.

```python
from finetune_cli.core.config import ConfigBuilder
from finetune_cli.core.types import TrainingMethod, DatasetSource

config = (
    ConfigBuilder()
    .with_model("gpt2", torch_dtype="float32")
    .with_dataset("./data.jsonl", source=DatasetSource.LOCAL_FILE, max_samples=1000)
    .with_tokenization(max_length=512)
    .with_training(TrainingMethod.LORA, "./output", num_epochs=3, batch_size=4)
    .with_lora(r=8, lora_alpha=32, lora_dropout=0.1)
    .build()
)
```

**Methods:**

| Method | Key kwargs | Description |
|--------|-----------|-------------|
| `.with_model(name, **kwargs)` | `torch_dtype`, `load_in_4bit`, `load_in_8bit` | Set model config |
| `.with_dataset(path, source, **kwargs)` | `max_samples`, `text_columns`, `shuffle` | Set dataset config |
| `.with_tokenization(**kwargs)` | `max_length`, `truncation`, `padding` | Set tokenization config |
| `.with_training(method, output_dir, **kwargs)` | `num_epochs`, `batch_size`, `learning_rate`, `fp16` | Set training config |
| `.with_lora(**kwargs)` | `r`, `lora_alpha`, `lora_dropout`, `target_modules` | Set LoRA config |
| `.with_evaluation(metrics, **kwargs)` | `batch_size`, `num_samples` | Set evaluation config |
| `.build()` | — | Validate and return `PipelineConfig` |

### `PipelineConfig`

Pydantic model holding the full pipeline config. Supports JSON and YAML I/O.

```python
# Load from file
config = PipelineConfig.from_yaml(Path("config.yaml"))
config = PipelineConfig.from_json(Path("config.json"))

# Save to file
config.to_yaml(Path("config.yaml"))
config.to_json(Path("config.json"))
```

---

## Data Pipeline — `finetune_cli.data`

### `quick_load`

One-liner for loading and tokenizing a dataset.

```python
from finetune_cli.data import quick_load
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = quick_load("./data.jsonl", tokenizer, max_samples=500, max_length=512)
# Returns: datasets.Dataset with input_ids, attention_mask, labels
```

### `prepare_dataset`

Full pipeline with optional train/validation split.

```python
from finetune_cli.data import prepare_dataset

result = prepare_dataset(
    dataset_config=config.dataset.to_config(),
    tokenization_config=config.tokenization.to_config(),
    tokenizer=tokenizer,
    split_for_validation=True,
    validation_ratio=0.1,
)
# result["train"], result["validation"]
```

### `DataPipeline`

Stateful pipeline class — use when you need statistics or to save processed data.

```python
from finetune_cli.data import DataPipeline

pipeline = DataPipeline(dataset_config, tokenization_config, tokenizer)
dataset = pipeline.run(split_for_validation=False)
stats = pipeline.get_statistics()   # {"num_samples": 1000, "avg_words": 42, ...}
pipeline.save_processed(Path("./data/processed"))
```

---

## Model Loading — `finetune_cli.models.loader`

```python
from finetune_cli.models.loader import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(config.model.to_config())
```

Handles device mapping, 4-bit/8-bit quantization, and `pad_token` setup automatically.

---

## Trainers — `finetune_cli.trainers`

### `TrainerFactory.train` (recommended)

Single entry point — selects the right trainer based on `TrainingMethod`.

```python
from finetune_cli.trainers import TrainerFactory

result = TrainerFactory.train(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    training_config=config.training.to_config(),
    lora_config=config.lora.to_config(),
)
# result.output_dir, result.train_loss, result.steps_completed
```

### `LoRATrainer` / `QLoRATrainer` (direct use)

```python
from finetune_cli.trainers import LoRATrainer

trainer = LoRATrainer(model, tokenizer, training_config, lora_config)
result = trainer.train(dataset)
```

### `TrainingResult`

Frozen dataclass returned by all trainers.

| Field | Type | Description |
|-------|------|-------------|
| `output_dir` | `Path` | Where the adapter was saved |
| `train_loss` | `float` | Final training loss |
| `steps_completed` | `int` | Total training steps |
| `elapsed_seconds` | `float` | Wall-clock training time |

---

## Evaluation — `finetune_cli.evaluation`

### `BenchmarkRunner`

```python
from finetune_cli.evaluation import BenchmarkRunner
from finetune_cli.core.types import EvaluationConfig, EvaluationMetric

eval_cfg = EvaluationConfig(
    metrics=[EvaluationMetric.ROUGE_L, EvaluationMetric.BLEU],
    num_samples=100,
    generation_max_length=100,
)
runner = BenchmarkRunner(eval_cfg, tokenizer)

# Single model score
result = runner.evaluate(model, dataset, label="fine-tuned")
print(result.scores)  # {"rougeL": 0.42, "bleu": 0.19}

# Before/after comparison
report = runner.run_comparison(base_model, ft_model, dataset)
print(report.summary())
print(report.improvements)  # {"rougeL": +0.12, "bleu": +0.07}
```

### Available metrics

| `EvaluationMetric` value | Class | Notes |
|--------------------------|-------|-------|
| `rouge1`, `rouge2`, `rougeL` | `RougeMetric` | Token overlap |
| `bleu` | `BleuMetric` | N-gram precision (requires `nltk punkt`) |
| `perplexity` | `PerplexityMetric` | Requires model |

---

## Exceptions — `finetune_cli.core.exceptions`

All exceptions inherit from `FineTuneError`.

```python
from finetune_cli.core.exceptions import (
    InvalidConfigError,      # Bad config value
    MissingConfigError,      # Required field absent
    IncompatibleConfigError, # Conflicting options (e.g. fp16 + bf16)
    DatasetNotFoundError,    # File path doesn't exist
    EmptyDatasetError,       # Dataset loaded but has 0 rows
    NoTextColumnsError,      # No string columns found
    TrainingError,           # Training loop failure
    ModelLoadError,          # Model download / load failed
)
```

---

## Trainers — full reference

### All training methods

| `TrainingMethod` | Trainer class | Needs `lora_config` | Notes |
|------------------|--------------|---------------------|-------|
| `lora` | `LoRATrainer` | Yes | Default. Attaches LoRA adapters, freezes base. |
| `qlora` | `QLoRATrainer` | Yes | 4-bit quantised base + LoRA. Set `load_in_4bit: true`. |
| `instruction_tuning` | `InstructionTrainer` | Yes | Auto-formats `{instruction, input, response}` datasets. Skip if `input_ids` present. |
| `full_finetuning` | `FullFineTuner` | No | Trains all parameters. VRAM warning for >1B param models. |

### `FullFineTuner`

```python
from finetune_cli.trainers import FullFineTuner

trainer = FullFineTuner(model, tokenizer, training_config)
result = trainer.train(dataset)
# Issues ResourceWarning if model has >1B parameters
```

### `InstructionTrainer`

```python
from finetune_cli.trainers import InstructionTrainer, format_instruction_dataset
from datasets import Dataset

# Format raw alpaca-style data
raw = Dataset.from_list([
    {"instruction": "Explain X.", "input": "", "response": "X is ..."},
])
formatted = format_instruction_dataset(raw)
# formatted has a single "text" column with the alpaca prompt template

# Or pass raw data directly — InstructionTrainer formats automatically
trainer = InstructionTrainer(model, tokenizer, training_config, lora_config)
result = trainer.train(raw)  # formats then trains
```

---

## CLI commands — reference

### `finetune-cli train`

```
finetune-cli train [OPTIONS]

Options:
  --config, -c PATH        YAML/JSON config file (takes precedence over flags)
  --model, -m TEXT         HuggingFace model id          [default: gpt2]
  --dataset, -d PATH       Local dataset path
  --hf-dataset TEXT        HuggingFace dataset id
  --output, -o PATH        Output directory              [default: ./output]
  --method TEXT            lora | qlora | instruction_tuning | full_finetuning
  --lora-r INT             LoRA rank                     [default: 8]
  --lora-alpha INT         LoRA alpha                    [default: 32]
  --epochs, -e INT         Number of epochs              [default: 3]
  --batch-size, -b INT     Per-device batch size         [default: 4]
  --lr FLOAT               Learning rate                 [default: 2e-4]
  --max-length INT         Max token length              [default: 512]
  --4bit                   Load model in 4-bit (QLoRA)
  --fp16                   Mixed precision FP16
```

### `finetune-cli merge`

```
finetune-cli merge ADAPTER_DIR OUTPUT_DIR [OPTIONS]

Arguments:
  ADAPTER_DIR   Path to saved LoRA adapter directory
  OUTPUT_DIR    Directory to save the merged standalone model

Options:
  --base-model, -b TEXT    Base HuggingFace model id     [required]
  --dtype TEXT             float32 | float16 | bfloat16  [default: float32]
```

The merged model runs without PEFT installed and is ready for direct inference or HuggingFace Hub upload.

```bash
finetune-cli merge ./outputs/gpt2_lora ./outputs/gpt2_merged \
  --base-model gpt2 --dtype float16
```

### `finetune-cli recommend`

```
finetune-cli recommend MODEL [OPTIONS]

Arguments:
  MODEL         HuggingFace model id (e.g. gpt2, meta-llama/Llama-3.2-1B)

Options:
  --dataset, -d PATH       Optional local dataset path
  --output, -o PATH        Save generated YAML config to file
  --vram FLOAT             Available VRAM in GB (auto-detect if omitted)
```

Decision logic:

| Model size | VRAM | Recommended method |
|-----------|------|--------------------|
| >7B | any | qlora, r=16, grad_ckpt |
| >1B | ≥16GB | lora, r=16 |
| >1B | <16GB | qlora, r=8, grad_ckpt |
| >300M | ≥8GB | lora, r=8 |
| >300M | <8GB | lora, r=4 |
| ≤300M | ≥4GB | lora, r=8 |
| ≤300M | <4GB | full_finetuning |

```bash
finetune-cli recommend gpt2 --output my_config.yaml
finetune-cli train --config my_config.yaml
```

### `finetune-cli upload`

```
finetune-cli upload MODEL_PATH REPO_ID [OPTIONS]

Options:
  --token, -t TEXT         HF API token (or set HF_TOKEN env var)
  --private                Make repository private
  --message, -m TEXT       Commit message
  --merge-adapter          Merge LoRA adapter before uploading
  --base-model TEXT        Base model id (required with --merge-adapter)
```