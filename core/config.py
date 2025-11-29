"""
Configuration management for the fine-tuning framework.

Provides Pydantic-based configuration with validation, serialization,
and composition of configuration objects.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from .types import (
    TrainingMethod, DatasetSource, DeviceType, EvaluationMetric,
    ModelConfig, DatasetConfig, TokenizationConfig, LoRAConfig,
    TrainingConfig, EvaluationConfig
)
from .exceptions import (
    InvalidConfigError, MissingConfigError, IncompatibleConfigError
)


T = TypeVar('T', bound=BaseModel)


# ============================================================================
# CONFIGURATION MODELS (Pydantic)
# ============================================================================


class ModelConfigModel(BaseModel):
    """Pydantic model for ModelConfig with validation."""
    
    name: str = Field(..., description="HuggingFace model identifier")
    device: DeviceType = Field(DeviceType.AUTO, description="Compute device")
    torch_dtype: Optional[str] = Field(None, description="Model dtype (float32, float16, bfloat16)")
    load_in_8bit: bool = Field(False, description="Load in 8-bit quantization")
    load_in_4bit: bool = Field(False, description="Load in 4-bit quantization")
    use_flash_attention: bool = Field(False, description="Enable Flash Attention 2")
    trust_remote_code: bool = Field(False, description="Allow custom code execution")
    revision: Optional[str] = Field(None, description="Model revision")
    cache_dir: Optional[str] = Field(None, description="Cache directory")
    
    @field_validator('torch_dtype', mode='before')
    @classmethod
    def validate_dtype(cls, v: Optional[str]) -> Optional[str]:
        """Validate torch dtype string."""
        if v is None:
            return None
        valid_dtypes = ['float32', 'float16', 'bfloat16', 'auto']
        if v not in valid_dtypes:
            raise InvalidConfigError(
                f"Invalid torch_dtype '{v}'. Must be one of {valid_dtypes}"
            )
        return v
    
    @model_validator(mode='after')
    def validate_quantization(self) -> 'ModelConfigModel':
        """Validate quantization options are not both enabled."""
        if self.load_in_8bit and self.load_in_4bit:
            raise IncompatibleConfigError(
                "Cannot enable both 8-bit and 4-bit quantization",
                ["load_in_8bit", "load_in_4bit"]
            )
        return self
    
    def to_config(self) -> ModelConfig:
        """Convert to immutable ModelConfig dataclass."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'auto': None
        }
        return ModelConfig(
            name=self.name,
            device=self.device,
            torch_dtype=dtype_map.get(self.torch_dtype) if self.torch_dtype else None,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            use_flash_attention=self.use_flash_attention,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            cache_dir=Path(self.cache_dir) if self.cache_dir else None
        )


class DatasetConfigModel(BaseModel):
    """Pydantic model for DatasetConfig with validation."""
    
    source: DatasetSource = Field(..., description="Dataset source type")
    path: str = Field(..., description="Dataset path or identifier")
    split: str = Field("train", description="Dataset split")
    config_name: Optional[str] = Field(None, description="Dataset config name")
    data_files: Optional[Union[str, list[str]]] = Field(None, description="Specific files")
    text_columns: Optional[list[str]] = Field(None, description="Text column names")
    max_samples: Optional[int] = Field(None, ge=1, description="Maximum samples")
    streaming: bool = Field(False, description="Enable streaming mode")
    shuffle: bool = Field(True, description="Shuffle dataset")
    seed: int = Field(42, ge=0, description="Random seed")
    
    @field_validator('max_samples')
    @classmethod
    def validate_max_samples(cls, v: Optional[int]) -> Optional[int]:
        """Validate max_samples is positive."""
        if v is not None and v <= 0:
            raise InvalidConfigError("max_samples must be positive")
        return v
    
    @model_validator(mode='after')
    def validate_source_path(self) -> 'DatasetConfigModel':
        """Validate path based on source type."""
        if self.source == DatasetSource.LOCAL_FILE:
            path = Path(self.path)
            if not path.exists():
                raise InvalidConfigError(f"Local file not found: {self.path}")
        return self
    
    def to_config(self) -> DatasetConfig:
        """Convert to immutable DatasetConfig dataclass."""
        return DatasetConfig(
            source=self.source,
            path=self.path,
            split=self.split,
            config_name=self.config_name,
            data_files=self.data_files,
            text_columns=self.text_columns,
            max_samples=self.max_samples,
            streaming=self.streaming,
            shuffle=self.shuffle,
            seed=self.seed
        )


class TokenizationConfigModel(BaseModel):
    """Pydantic model for TokenizationConfig with validation."""
    
    max_length: int = Field(512, ge=1, le=8192, description="Maximum sequence length")
    truncation: bool = Field(True, description="Enable truncation")
    padding: str = Field("max_length", description="Padding strategy")
    add_special_tokens: bool = Field(True, description="Add special tokens")
    return_attention_mask: bool = Field(True, description="Return attention mask")
    
    @field_validator('padding')
    @classmethod
    def validate_padding(cls, v: str) -> str:
        """Validate padding strategy."""
        valid = ['max_length', 'longest', 'do_not_pad']
        if v not in valid:
            raise InvalidConfigError(f"Invalid padding '{v}'. Must be one of {valid}")
        return v
    
    def to_config(self) -> TokenizationConfig:
        """Convert to immutable TokenizationConfig dataclass."""
        return TokenizationConfig(
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,  # type: ignore
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=self.return_attention_mask
        )


class LoRAConfigModel(BaseModel):
    """Pydantic model for LoRAConfig with validation."""
    
    r: int = Field(8, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(32, ge=1, description="LoRA alpha")
    lora_dropout: float = Field(0.1, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: Optional[list[str]] = Field(None, description="Target modules")
    bias: str = Field("none", description="Bias training strategy")
    fan_in_fan_out: bool = Field(False, description="Conv1D layer mode")
    init_lora_weights: Union[bool, str] = Field(True, description="Weight init strategy")
    
    @field_validator('bias')
    @classmethod
    def validate_bias(cls, v: str) -> str:
        """Validate bias strategy."""
        valid = ['none', 'all', 'lora_only']
        if v not in valid:
            raise InvalidConfigError(f"Invalid bias '{v}'. Must be one of {valid}")
        return v
    
    @field_validator('init_lora_weights')
    @classmethod
    def validate_init(cls, v: Union[bool, str]) -> Union[bool, str]:
        """Validate initialization strategy."""
        if isinstance(v, str):
            valid = ['gaussian', 'loftq']
            if v not in valid:
                raise InvalidConfigError(f"Invalid init '{v}'. Must be one of {valid}")
        return v
    
    @model_validator(mode='after')
    def validate_alpha_ratio(self) -> 'LoRAConfigModel':
        """Warn if alpha/r ratio is unusual."""
        ratio = self.lora_alpha / self.r
        if ratio < 0.5 or ratio > 8.0:
            import warnings
            warnings.warn(
                f"Unusual lora_alpha/r ratio: {ratio:.2f}. "
                f"Typical range is 0.5-8.0 (recommended: 2.0-4.0)"
            )
        return self
    
    def to_config(self) -> LoRAConfig:
        """Convert to immutable LoRAConfig dataclass."""
        return LoRAConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,  # type: ignore
            fan_in_fan_out=self.fan_in_fan_out,
            init_lora_weights=self.init_lora_weights  # type: ignore
        )


class TrainingConfigModel(BaseModel):
    """Pydantic model for TrainingConfig with validation."""
    
    method: TrainingMethod = Field(..., description="Training method")
    output_dir: str = Field(..., description="Output directory")
    num_epochs: int = Field(3, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Gradient accumulation")
    learning_rate: float = Field(2e-4, gt=0.0, le=1.0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="Weight decay")
    warmup_ratio: float = Field(0.1, ge=0.0, le=0.5, description="Warmup ratio")
    lr_scheduler_type: str = Field("cosine", description="LR scheduler")
    max_grad_norm: float = Field(1.0, gt=0.0, description="Max gradient norm")
    fp16: bool = Field(False, description="Enable FP16")
    bf16: bool = Field(False, description="Enable BF16")
    logging_steps: int = Field(10, ge=1, description="Logging frequency")
    save_steps: Optional[int] = Field(None, description="Save checkpoint steps")
    save_strategy: str = Field("epoch", description="Save strategy")
    evaluation_strategy: str = Field("epoch", description="Evaluation strategy")
    load_best_model_at_end: bool = Field(True, description="Load best model")
    seed: int = Field(42, ge=0, description="Random seed")
    gradient_checkpointing: bool = Field(False, description="Gradient checkpointing")
    
    @field_validator('lr_scheduler_type')
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        """Validate scheduler type."""
        valid = ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant']
        if v not in valid:
            raise InvalidConfigError(f"Invalid scheduler '{v}'. Must be one of {valid}")
        return v
    
    @field_validator('save_strategy', 'evaluation_strategy')
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate save/eval strategy."""
        valid = ['no', 'epoch', 'steps']
        if v not in valid:
            raise InvalidConfigError(f"Invalid strategy '{v}'. Must be one of {valid}")
        return v
    
    @model_validator(mode='after')
    def validate_mixed_precision(self) -> 'TrainingConfigModel':
        """Validate mixed precision options."""
        if self.fp16 and self.bf16:
            raise IncompatibleConfigError(
                "Cannot enable both FP16 and BF16",
                ["fp16", "bf16"]
            )
        return self
    
    @model_validator(mode='after')
    def validate_save_steps(self) -> 'TrainingConfigModel':
        """Validate save_steps when strategy is 'steps'."""
        if self.save_strategy == 'steps' and self.save_steps is None:
            raise MissingConfigError("save_steps", "TrainingConfig")
        return self
    
    def to_config(self) -> TrainingConfig:
        """Convert to immutable TrainingConfig dataclass."""
        return TrainingConfig(
            method=self.method,
            output_dir=Path(self.output_dir),
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            max_grad_norm=self.max_grad_norm,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_strategy=self.save_strategy,  # type: ignore
            evaluation_strategy=self.evaluation_strategy,  # type: ignore
            load_best_model_at_end=self.load_best_model_at_end,
            seed=self.seed,
            gradient_checkpointing=self.gradient_checkpointing
        )


class EvaluationConfigModel(BaseModel):
    """Pydantic model for EvaluationConfig with validation."""
    
    metrics: list[EvaluationMetric] = Field(..., description="Evaluation metrics")
    batch_size: int = Field(8, ge=1, description="Batch size")
    num_samples: Optional[int] = Field(None, ge=1, description="Number of samples")
    generation_max_length: int = Field(100, ge=1, description="Max generation length")
    generation_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")
    generation_top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p")
    generation_do_sample: bool = Field(True, description="Enable sampling")
    
    def to_config(self) -> EvaluationConfig:
        """Convert to immutable EvaluationConfig dataclass."""
        return EvaluationConfig(
            metrics=self.metrics,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            generation_max_length=self.generation_max_length,
            generation_temperature=self.generation_temperature,
            generation_top_p=self.generation_top_p,
            generation_do_sample=self.generation_do_sample
        )


# ============================================================================
# COMPOSITE CONFIGURATION
# ============================================================================


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    model: ModelConfigModel
    dataset: DatasetConfigModel
    tokenization: TokenizationConfigModel
    training: TrainingConfigModel
    evaluation: Optional[EvaluationConfigModel] = None
    lora: Optional[LoRAConfigModel] = None
    
    @model_validator(mode='after')
    def validate_method_config(self) -> 'PipelineConfig':
        """Validate method-specific config is present."""
        if self.training.method == TrainingMethod.LORA and self.lora is None:
            raise MissingConfigError("lora", "PipelineConfig")
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'PipelineConfig':
        """Load from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'PipelineConfig':
        """Load from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: Path) -> None:
        """Save to JSON file."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save to YAML file."""
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# ============================================================================
# CONFIGURATION BUILDER
# ============================================================================


class ConfigBuilder:
    """Builder for constructing configurations programmatically."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
    
    def with_model(self, name: str, **kwargs) -> 'ConfigBuilder':
        """Add model configuration."""
        self._config['model'] = {'name': name, **kwargs}
        return self
    
    def with_dataset(self, path: str, source: DatasetSource = DatasetSource.LOCAL_FILE,
                     **kwargs) -> 'ConfigBuilder':
        """Add dataset configuration."""
        self._config['dataset'] = {'source': source, 'path': path, **kwargs}
        return self
    
    def with_tokenization(self, **kwargs) -> 'ConfigBuilder':
        """Add tokenization configuration."""
        self._config['tokenization'] = kwargs
        return self
    
    def with_training(self, method: TrainingMethod, output_dir: str,
                      **kwargs) -> 'ConfigBuilder':
        """Add training configuration."""
        self._config['training'] = {'method': method, 'output_dir': output_dir, **kwargs}
        return self
    
    def with_lora(self, **kwargs) -> 'ConfigBuilder':
        """Add LoRA configuration."""
        self._config['lora'] = kwargs
        return self
    
    def with_evaluation(self, metrics: list[EvaluationMetric],
                        **kwargs) -> 'ConfigBuilder':
        """Add evaluation configuration."""
        self._config['evaluation'] = {'metrics': metrics, **kwargs}
        return self
    
    def build(self) -> PipelineConfig:
        """Build and validate final configuration."""
        return PipelineConfig.from_dict(self._config)