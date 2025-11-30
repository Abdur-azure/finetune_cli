"""
QLoRA (Quantized LoRA) trainer implementation.

Implements LoRA training on quantized models (4-bit or 8-bit)
for extreme memory efficiency.
"""

from typing import Optional
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from ..core.types import TrainingConfig, TrainingResult, LoRAConfig as LoRAConfigType, ModelConfig
from ..core.exceptions import TrainingError, UnsupportedModelError
from ..utils.logging import get_logger
from .lora_trainer import LoRATrainer


logger = get_logger(__name__)


# ============================================================================
# QLORA TRAINER
# ============================================================================


class QLoRATrainer(LoRATrainer):
    """
    Trainer for QLoRA (Quantized LoRA) fine-tuning.
    
    Extends LoRATrainer to work with quantized models.
    The model must be loaded with quantization (4-bit or 8-bit)
    before passing to this trainer.
    
    Key differences from LoRA:
    - Works with quantized base model
    - Uses special optimizers (paged_adamw_32bit)
    - Requires gradient checkpointing
    - More memory efficient but slightly slower
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType,
        model_config: ModelConfig
    ):
        """
        Initialize QLoRA trainer.
        
        Args:
            model: Quantized base model
            tokenizer: Tokenizer for the model
            training_config: Training configuration
            lora_config: LoRA configuration
            model_config: Model configuration (to check quantization)
        """
        self.model_config = model_config
        
        # Validate model is quantized
        self._validate_quantization()
        
        super().__init__(model, tokenizer, training_config, lora_config)
        
        self.logger.info("Initialized QLoRA trainer with quantized model")
    
    def _validate_quantization(self) -> None:
        """Validate that model is properly quantized."""
        if not (self.model_config.load_in_4bit or self.model_config.load_in_8bit):
            raise UnsupportedModelError(
                self.model_config.name,
                "unknown",
                "QLoRA requires model to be loaded with 4-bit or 8-bit quantization. "
                "Set load_in_4bit=True or load_in_8bit=True in ModelConfig."
            )
        
        quant_type = "4-bit" if self.model_config.load_in_4bit else "8-bit"
        self.logger.info(f"Model is quantized with {quant_type}")
    
    def _validate_setup(self) -> None:
        """Validate QLoRA-specific setup requirements."""
        super()._validate_setup()
        
        # QLoRA requires gradient checkpointing
        if not self.config.gradient_checkpointing:
            self.logger.warning(
                "Gradient checkpointing is disabled but recommended for QLoRA. "
                "Enabling automatically."
            )
            # Create new config with gradient checkpointing
            from dataclasses import replace
            self.config = replace(self.config, gradient_checkpointing=True)
        
        # Log memory savings
        self._log_memory_savings()
    
    def _log_memory_savings(self) -> None:
        """Log estimated memory savings from quantization."""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate memory usage
        if self.model_config.load_in_4bit:
            quant_memory = total_params * 0.5 / 1e9  # 4-bit = 0.5 bytes per param
            full_memory = total_params * 4 / 1e9     # FP32 = 4 bytes per param
            savings_ratio = 8.0
        else:  # 8-bit
            quant_memory = total_params * 1 / 1e9    # 8-bit = 1 byte per param
            full_memory = total_params * 4 / 1e9     # FP32 = 4 bytes per param
            savings_ratio = 4.0
        
        self.logger.info("QLoRA Memory Savings:")
        self.logger.info(f"  Quantized model: ~{quant_memory:.2f} GB")
        self.logger.info(f"  Full precision would be: ~{full_memory:.2f} GB")
        self.logger.info(f"  Memory savings: ~{savings_ratio:.1f}x")
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare quantized model with LoRA adapters.
        
        Returns:
            Quantized model with LoRA adapters
        """
        # Call parent implementation
        peft_model = super().prepare_model()
        
        # Additional QLoRA-specific setup
        self.logger.info("Configuring QLoRA-specific optimizations...")
        
        # Enable gradient checkpointing for quantized model
        if hasattr(peft_model, 'enable_input_require_grads'):
            peft_model.enable_input_require_grads()
        
        return peft_model
    
    def get_training_args(self):
        """
        Build training arguments optimized for QLoRA.
        
        Returns:
            TrainingArguments with QLoRA-specific settings
        """
        # Get base arguments
        args = super().get_training_args()
        
        # Override optimizer for QLoRA
        # paged_adamw_32bit is recommended for quantized models
        from transformers import TrainingArguments
        
        return TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            max_grad_norm=args.max_grad_norm,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            evaluation_strategy=args.evaluation_strategy,
            load_best_model_at_end=args.load_best_model_at_end,
            report_to=args.report_to,
            seed=args.seed,
            gradient_checkpointing=True,  # Always true for QLoRA
            # QLoRA-specific optimizer
            optim="paged_adamw_32bit",
            # Additional settings
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=args.remove_unused_columns,
            label_names=args.label_names,
        )
    
    def merge_and_save(self, output_dir: Optional[Path] = None) -> None:
        """
        Merge and dequantize model before saving.
        
        Note: This will dequantize the model, so the output will be
        larger than the quantized version but can be used without
        quantization libraries.
        
        Args:
            output_dir: Directory to save merged model
        """
        self.logger.warning(
            "Merging QLoRA adapters will dequantize the model. "
            "The output model will require more memory to load."
        )
        super().merge_and_save(output_dir)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def train_with_qlora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    training_config: TrainingConfig,
    lora_config: LoRAConfigType,
    model_config: ModelConfig,
    eval_dataset: Optional[Dataset] = None
) -> TrainingResult:
    """
    Convenience function to train with QLoRA.
    
    Args:
        model: Quantized base model
        tokenizer: Tokenizer
        train_dataset: Training data
        training_config: Training configuration
        lora_config: LoRA configuration
        model_config: Model configuration (with quantization info)
        eval_dataset: Optional evaluation data
    
    Returns:
        Training results
    
    Example:
        >>> from finetune_cli.models.loader import load_model_and_tokenizer
        >>> from finetune_cli.core.config import ConfigBuilder
        >>> from finetune_cli.core.types import TrainingMethod
        >>> 
        >>> # Build config with quantization
        >>> config = ConfigBuilder() \\
        ...     .with_model("meta-llama/Llama-2-7b-hf", load_in_4bit=True) \\
        ...     .with_training(TrainingMethod.QLORA, "./output") \\
        ...     .with_lora(r=16, lora_alpha=64) \\
        ...     .build()
        >>> 
        >>> # Load quantized model
        >>> model, tokenizer = load_model_and_tokenizer(config.model.to_config())
        >>> 
        >>> # Train with QLoRA
        >>> result = train_with_qlora(
        ...     model, tokenizer, dataset,
        ...     config.training.to_config(),
        ...     config.lora.to_config(),
        ...     config.model.to_config()
        ... )
    """
    trainer = QLoRATrainer(
        model, tokenizer, training_config, lora_config, model_config
    )
    return trainer.train(train_dataset, eval_dataset)


# ============================================================================
# BEST PRACTICES GUIDE
# ============================================================================


def get_qlora_best_practices() -> dict:
    """
    Get best practices for QLoRA training.
    
    Returns:
        Dictionary of recommendations
    """
    return {
        "quantization": {
            "recommended": "4-bit NF4 with double quantization",
            "models": {
                "small (<1B params)": "8-bit or no quantization",
                "medium (1-7B params)": "4-bit recommended",
                "large (7B+ params)": "4-bit required for consumer GPUs"
            }
        },
        "lora_config": {
            "r": "Higher rank (16-64) recommended for quantized models",
            "alpha": "2-4x the rank value",
            "target_modules": "All attention layers + MLP for best results"
        },
        "training": {
            "batch_size": "Can use larger batch sizes due to memory savings",
            "gradient_accumulation": "4-8 steps recommended",
            "gradient_checkpointing": "Always enable",
            "optimizer": "paged_adamw_32bit (automatic in QLoRATrainer)"
        },
        "hardware": {
            "min_vram": "~6GB for 7B models",
            "recommended_vram": "12GB+ for comfortable training",
            "notes": "QLoRA makes 7B+ models accessible on consumer GPUs"
        }
    }