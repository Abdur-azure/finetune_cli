"""
Core type definitions for the fine-tuning framework.

This module defines all shared types, enums, and protocols used across
the framework. It serves as the single source of truth for type contracts.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Union
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


# ============================================================================
# ENUMERATIONS
# ============================================================================


class TrainingMethod(str, Enum):
    """Available training methods following LLM fine-tuning taxonomy."""
    
    # Full Fine-Tuning
    FULL_FINETUNING = "full_finetuning"
    
    # Parameter-Efficient Fine-Tuning (PEFT)
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"
    PROMPT_TUNING = "prompt_tuning"
    IA3 = "ia3"
    
    # Instruction & Alignment Fine-Tuning
    INSTRUCTION_TUNING = "instruction_tuning"
    RLHF = "rlhf"
    DPO = "dpo"
    RLAIF = "rlaif"
    
    # Contrastive Fine-Tuning
    SIMCSE = "simcse"
    CONTRASTIVE_ALIGNMENT = "contrastive_alignment"
    
    # Knowledge Distillation
    VANILLA_DISTILLATION = "vanilla_distillation"
    FEATURE_DISTILLATION = "feature_distillation"
    SELF_DISTILLATION = "self_distillation"
    
    # Quantization-Aware Training
    QAT_INT8 = "qat_int8"
    QAT_INT4 = "qat_int4"
    
    # Pruning-Aware Training
    MAGNITUDE_PRUNING = "magnitude_pruning"
    STRUCTURED_PRUNING = "structured_pruning"


class DatasetSource(str, Enum):
    """Dataset source types."""
    LOCAL_FILE = "local_file"
    HUGGINGFACE_HUB = "huggingface_hub"
    CUSTOM = "custom"


class FileFormat(str, Enum):
    """Supported file formats for local datasets."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    TXT = "txt"


class EvaluationMetric(str, Enum):
    """Available evaluation metrics."""
    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_L = "rougeL"
    BLEU = "bleu"
    PERPLEXITY = "perplexity"
    ACCURACY = "accuracy"
    F1 = "f1"
    EXACT_MATCH = "exact_match"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DeviceType(str, Enum):
    """Compute device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model loading and initialization."""
    
    name: str
    """HuggingFace model identifier or local path."""
    
    device: DeviceType = DeviceType.AUTO
    """Device to load model on."""
    
    torch_dtype: Optional[torch.dtype] = None
    """Data type for model weights."""
    
    load_in_8bit: bool = False
    """Enable 8-bit quantization."""
    
    load_in_4bit: bool = False
    """Enable 4-bit quantization."""
    
    use_flash_attention: bool = False
    """Enable Flash Attention 2 if available."""
    
    trust_remote_code: bool = False
    """Allow custom model code execution."""
    
    revision: Optional[str] = None
    """Model revision/branch to use."""
    
    cache_dir: Optional[Path] = None
    """Directory for caching downloaded models."""


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    source: DatasetSource
    """Dataset source type."""
    
    path: str
    """Path to dataset (local file or HuggingFace identifier)."""
    
    split: str = "train"
    """Dataset split to use."""
    
    config_name: Optional[str] = None
    """Dataset configuration/subset name."""
    
    data_files: Optional[Union[str, List[str]]] = None
    """Specific files to load from repository."""
    
    text_columns: Optional[List[str]] = None
    """Columns containing text data."""
    
    max_samples: Optional[int] = None
    """Maximum number of samples to load."""
    
    streaming: bool = False
    """Enable streaming mode for large datasets."""
    
    shuffle: bool = True
    """Shuffle dataset after loading."""
    
    seed: int = 42
    """Random seed for reproducibility."""


@dataclass(frozen=True)
class TokenizationConfig:
    """Configuration for text tokenization."""
    
    max_length: int = 512
    """Maximum sequence length."""
    
    truncation: bool = True
    """Enable truncation for long sequences."""
    
    padding: Literal["max_length", "longest", "do_not_pad"] = "max_length"
    """Padding strategy."""
    
    add_special_tokens: bool = True
    """Add special tokens (BOS, EOS, etc.)."""
    
    return_attention_mask: bool = True
    """Return attention masks."""


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA training."""
    
    r: int = 8
    """LoRA rank."""
    
    lora_alpha: int = 32
    """LoRA scaling parameter."""
    
    lora_dropout: float = 0.1
    """Dropout probability for LoRA layers."""
    
    target_modules: Optional[List[str]] = None
    """Modules to apply LoRA to. Auto-detected if None."""
    
    bias: Literal["none", "all", "lora_only"] = "none"
    """Bias training strategy."""
    
    fan_in_fan_out: bool = False
    """Set for Conv1D layers (e.g., GPT-2)."""
    
    init_lora_weights: Union[bool, Literal["gaussian", "loftq"]] = True
    """LoRA weight initialization strategy."""


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training process."""
    
    method: TrainingMethod
    """Training method to use."""
    
    output_dir: Path
    """Directory to save model and checkpoints."""
    
    num_epochs: int = 3
    """Number of training epochs."""
    
    batch_size: int = 4
    """Per-device batch size."""
    
    gradient_accumulation_steps: int = 4
    """Steps to accumulate gradients."""
    
    learning_rate: float = 2e-4
    """Learning rate."""
    
    weight_decay: float = 0.01
    """Weight decay coefficient."""
    
    warmup_ratio: float = 0.1
    """Ratio of training for learning rate warmup."""
    
    lr_scheduler_type: str = "cosine"
    """Learning rate scheduler type."""
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""
    
    fp16: bool = False
    """Enable mixed precision training (FP16)."""
    
    bf16: bool = False
    """Enable mixed precision training (BF16)."""
    
    logging_steps: int = 10
    """Log metrics every N steps."""
    
    save_steps: Optional[int] = None
    """Save checkpoint every N steps."""
    
    save_strategy: Literal["no", "epoch", "steps"] = "epoch"
    """Checkpoint saving strategy."""
    
    evaluation_strategy: Literal["no", "epoch", "steps"] = "epoch"
    """Evaluation strategy."""
    
    load_best_model_at_end: bool = True
    """Load best model after training."""
    
    seed: int = 42
    """Random seed for reproducibility."""
    
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing to save memory."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    metrics: List[EvaluationMetric]
    """Metrics to compute."""
    
    batch_size: int = 8
    """Batch size for evaluation."""
    
    num_samples: Optional[int] = None
    """Number of samples to evaluate on."""
    
    generation_max_length: int = 100
    """Maximum length for text generation."""
    
    generation_temperature: float = 0.7
    """Temperature for text generation."""
    
    generation_top_p: float = 0.9
    """Top-p sampling parameter."""
    
    generation_do_sample: bool = True
    """Enable sampling during generation."""


# ============================================================================
# PROTOCOLS (INTERFACES)
# ============================================================================


class ModelLoader(Protocol):
    """Protocol for model loading implementations."""
    
    def load_model(self, config: ModelConfig) -> PreTrainedModel:
        """Load a model based on configuration."""
        ...
    
    def load_tokenizer(self, config: ModelConfig) -> PreTrainedTokenizer:
        """Load a tokenizer based on configuration."""
        ...


class DatasetLoader(Protocol):
    """Protocol for dataset loading implementations."""
    
    def load(self, config: DatasetConfig) -> Any:
        """Load dataset based on configuration."""
        ...
    
    def prepare(self, dataset: Any, tokenizer: PreTrainedTokenizer, 
                config: TokenizationConfig) -> Any:
        """Prepare dataset for training."""
        ...


class Trainer(Protocol):
    """Protocol for trainer implementations."""
    
    def train(self, model: PreTrainedModel, dataset: Any, 
              config: TrainingConfig) -> Dict[str, Any]:
        """Train the model."""
        ...
    
    def save(self, output_dir: Path) -> None:
        """Save trained model."""
        ...


class Evaluator(Protocol):
    """Protocol for evaluation implementations."""
    
    def evaluate(self, model: PreTrainedModel, dataset: Any,
                 config: EvaluationConfig) -> Dict[str, float]:
        """Evaluate model on dataset."""
        ...


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass
class TrainingResult:
    """Results from training process."""
    
    method: TrainingMethod
    final_loss: float
    best_loss: float
    num_epochs_completed: int
    total_steps: int
    training_time_seconds: float
    metrics: Dict[str, List[float]]
    output_dir: Path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "num_epochs": self.num_epochs_completed,
            "total_steps": self.total_steps,
            "training_time": self.training_time_seconds,
            "metrics": self.metrics,
            "output_dir": str(self.output_dir)
        }


@dataclass
class EvaluationResult:
    """Results from evaluation process."""
    
    metrics: Dict[str, float]
    num_samples: int
    evaluation_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "evaluation_time": self.evaluation_time_seconds
        }


# ============================================================================
# TYPE ALIASES
# ============================================================================

PathLike = Union[str, Path]
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float]