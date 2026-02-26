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