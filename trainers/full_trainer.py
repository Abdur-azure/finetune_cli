"""
Full fine-tuning trainer implementation.

Trains all model parameters without parameter-efficient methods.
Provides maximum adaptation capability but requires most memory.
"""

from typing import Optional
from pathlib import Path
import time

import torch
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from ..core.types import TrainingConfig, TrainingResult
from ..core.exceptions import InsufficientVRAMError
from ..utils.logging import get_logger
from .base import BaseTrainer, MetricsTracker


logger = get_logger(__name__)


# ============================================================================
# FULL FINE-TUNER
# ============================================================================


class FullFineTuner(BaseTrainer):
    """
    Trainer for full fine-tuning (all parameters).
    
    Updates all model parameters during training. This provides
    maximum adaptation capability but requires significantly more
    memory than parameter-efficient methods like LoRA.
    
    Best for:
    - Small to medium models (<1B parameters)
    - Tasks requiring substantial model adaptation
    - When GPU memory is not a constraint
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig
    ):
        """
        Initialize full fine-tuner.
        
        Args:
            model: Model to fine-tune
            tokenizer: Tokenizer for the model
            config: Training configuration
        """
        super().__init__(model, tokenizer, config)
        self.metrics_tracker = MetricsTracker()
        
        # Check memory requirements
        self._check_memory_requirements()
    
    def _check_memory_requirements(self) -> None:
        """Check if there's sufficient memory for full fine-tuning."""
        memory = self.estimate_memory_usage()
        estimated_total = memory['total_estimated_gb']
        
        self.logger.info(f"Estimated memory requirement: {estimated_total:.2f} GB")
        
        # Check available VRAM
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"Available VRAM: {total_vram:.2f} GB")
            
            # Warn if close to limit
            if estimated_total > total_vram * 0.9:
                self.logger.warning(
                    f"Estimated memory ({estimated_total:.2f} GB) is close to "
                    f"VRAM limit ({total_vram:.2f} GB). Training may fail with OOM."
                )
                self.logger.warning("Consider using:")
                self.logger.warning("  - Smaller batch size")
                self.logger.warning("  - Gradient checkpointing")
                self.logger.warning("  - LoRA/QLoRA instead")
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model for full fine-tuning.
        
        Returns:
            Model with all parameters trainable
        """
        self.logger.info("Preparing model for full fine-tuning...")
        
        # Ensure all parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Trainable parameters:")
        self.logger.info(f"  Total: {total_params:,}")
        self.logger.info(f"  Trainable: {trainable_params:,}")
        self.logger.info(f"  Trainable %: {trainable_params / total_params * 100:.2f}%")
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("Enabled gradient checkpointing")
            else:
                self.logger.warning("Model does not support gradient checkpointing")
        
        return self.model
    
    def get_training_args(self) -> TrainingArguments:
        """
        Build HuggingFace TrainingArguments.
        
        Returns:
            TrainingArguments configured for full fine-tuning
        """
        # Use conservative defaults for full fine-tuning
        # to avoid OOM and instability
        
        return TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps if self.config.save_strategy == "steps" else None,
            evaluation_strategy=self.config.evaluation_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            report_to="none",
            seed=self.config.seed,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # Full fine-tuning specific
            dataloader_num_workers=4,
            remove_unused_columns=True,
            label_names=["labels"],
            # More aggressive memory optimizations
            dataloader_pin_memory=True,
            ddp_find_unused_parameters=False,
        )
    
    def _execute_training(
        self,
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: TrainingArguments,
        resume_from_checkpoint: Optional[Path]
    ) -> TrainingResult:
        """
        Execute full fine-tuning.
        
        Args:
            model: Model to train
            train_dataset: Training data
            eval_dataset: Optional evaluation data
            training_args: Training arguments
            resume_from_checkpoint: Optional checkpoint to resume from
        
        Returns:
            Training results
        """
        # Log training info
        self.log_training_info()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer with custom callback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[self._create_callback()]
        )
        
        # Train
        self.logger.info("Starting full fine-tuning...")
        try:
            train_result = trainer.train(
                resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                memory = self.estimate_memory_usage()
                from ..core.exceptions import handle_oom_error
                raise handle_oom_error(e, "cuda")
            raise
        
        # Save final model
        self.save()
        
        # Compute training time
        training_time = time.time() - self.start_time
        
        # Build result
        result = TrainingResult(
            method=self.config.method,
            final_loss=train_result.training_loss,
            best_loss=self.state.best_loss,
            num_epochs_completed=self.config.num_epochs,
            total_steps=self.state.total_steps,
            training_time_seconds=training_time,
            metrics=self.metrics_tracker.to_dict()['metrics'],
            output_dir=self.config.output_dir
        )
        
        self.logger.info("Full fine-tuning complete!")
        self.logger.info(f"  Final loss: {result.final_loss:.4f}")
        self.logger.info(f"  Best loss: {result.best_loss:.4f}")
        self.logger.info(f"  Training time: {training_time:.2f}s")
        
        return result
    
    def _create_callback(self):
        """Create custom callback for tracking metrics."""
        from transformers import TrainerCallback
        
        trainer_instance = self
        
        class MetricsCallback(TrainerCallback):
            """Callback to track metrics during training."""
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                """Called when metrics are logged."""
                if logs:
                    # Update state
                    trainer_instance.state.current_step = state.global_step
                    trainer_instance.state.current_epoch = int(state.epoch) if state.epoch else 0
                    trainer_instance.state.total_steps = state.max_steps
                    
                    # Track loss
                    if 'loss' in logs:
                        loss = logs['loss']
                        trainer_instance.state.current_loss = loss
                        trainer_instance.state.loss_history.append(loss)
                        trainer_instance.metrics_tracker.add(state.global_step, 'loss', loss)
                        
                        # Check for NaN
                        if loss != loss:
                            from ..core.exceptions import handle_nan_loss
                            raise handle_nan_loss(state.global_step, trainer_instance.state.loss_history)
                        
                        # Update best loss
                        if loss < trainer_instance.state.best_loss:
                            trainer_instance.state.best_loss = loss
                            trainer_instance.state.best_epoch = trainer_instance.state.current_epoch
                    
                    # Track learning rate
                    if 'learning_rate' in logs:
                        trainer_instance.metrics_tracker.add(
                            state.global_step, 'learning_rate', logs['learning_rate']
                        )
            
            def on_epoch_end(self, args, state, control, **kwargs):
                """Called at the end of each epoch."""
                trainer_instance.logger.info(
                    f"Epoch {trainer_instance.state.current_epoch} complete. "
                    f"Loss: {trainer_instance.state.current_loss:.4f}"
                )
        
        return MetricsCallback()
    
    def _save_model(self, output_dir: Path) -> None:
        """
        Save full model.
        
        Args:
            output_dir: Directory to save to
        """
        self.model.save_pretrained(output_dir)
        self.logger.info(f"Saved model to: {output_dir}")
    
    def cleanup(self) -> None:
        """Cleanup after training."""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Cleared CUDA cache")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def train_full_finetuning(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    training_config: TrainingConfig,
    eval_dataset: Optional[Dataset] = None
) -> TrainingResult:
    """
    Convenience function for full fine-tuning.
    
    Args:
        model: Model to fine-tune
        tokenizer: Tokenizer
        train_dataset: Training data
        training_config: Training configuration
        eval_dataset: Optional evaluation data
    
    Returns:
        Training results
    
    Example:
        >>> from finetune_cli.models.loader import load_model_and_tokenizer
        >>> from finetune_cli.core.config import ConfigBuilder
        >>> from finetune_cli.core.types import TrainingMethod
        >>> 
        >>> config = ConfigBuilder() \\
        ...     .with_model("gpt2") \\
        ...     .with_training(
        ...         TrainingMethod.FULL_FINETUNING,
        ...         "./output",
        ...         num_epochs=3,
        ...         batch_size=2,
        ...         gradient_checkpointing=True
        ...     ) \\
        ...     .build()
        >>> 
        >>> model, tokenizer = load_model_and_tokenizer(config.model.to_config())
        >>> result = train_full_finetuning(
        ...     model, tokenizer, dataset,
        ...     config.training.to_config()
        ... )
    """
    trainer = FullFineTuner(model, tokenizer, training_config)
    return trainer.train(train_dataset, eval_dataset)


# ============================================================================
# COMPARISON GUIDE
# ============================================================================


def get_finetuning_comparison() -> dict:
    """
    Get comparison between full fine-tuning and LoRA/QLoRA.
    
    Returns:
        Dictionary comparing methods
    """
    return {
        "full_finetuning": {
            "trainable_params": "100%",
            "memory_usage": "Very High (baseline)",
            "training_speed": "Baseline",
            "adaptation_quality": "Maximum",
            "best_for": [
                "Small models (<1B params)",
                "Tasks requiring significant adaptation",
                "When memory is not constrained"
            ],
            "gpu_requirements": {
                "gpt2 (124M)": "~4GB VRAM",
                "gpt2-medium (355M)": "~8GB VRAM",
                "gpt2-large (774M)": "~16GB VRAM",
                "gpt2-xl (1.5B)": "~24GB+ VRAM"
            }
        },
        "lora": {
            "trainable_params": "0.1-1%",
            "memory_usage": "Medium (~50% of full)",
            "training_speed": "Faster (fewer params)",
            "adaptation_quality": "High",
            "best_for": [
                "Most use cases",
                "Medium to large models",
                "Multiple task-specific adapters"
            ]
        },
        "qlora": {
            "trainable_params": "0.1-1%",
            "memory_usage": "Low (~12-25% of full)",
            "training_speed": "Slightly slower (quantization overhead)",
            "adaptation_quality": "High",
            "best_for": [
                "Large models (7B+ params)",
                "Consumer GPUs",
                "Memory-constrained environments"
            ]
        },
        "recommendations": {
            "if_memory_abundant": "Full fine-tuning for maximum quality",
            "if_memory_limited": "LoRA for good balance",
            "if_memory_very_limited": "QLoRA for large models on consumer hardware",
            "if_multiple_tasks": "LoRA with separate adapters per task"
        }
    }