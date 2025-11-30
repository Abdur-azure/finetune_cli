"""
LoRA (Low-Rank Adaptation) trainer implementation.

Implements parameter-efficient fine-tuning using LoRA adapters
with support for various model architectures.
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from ..core.types import TrainingConfig, TrainingResult, LoRAConfig as LoRAConfigType
from ..core.exceptions import TrainingError, NaNLossError
from ..utils.logging import get_logger
from ..models.loader import detect_target_modules
from .base import BaseTrainer, MetricsTracker


logger = get_logger(__name__)


# ============================================================================
# LORA TRAINER
# ============================================================================


class LoRATrainer(BaseTrainer):
    """
    Trainer for LoRA (Low-Rank Adaptation) fine-tuning.
    
    Applies LoRA adapters to specific modules in the model,
    training only the adapter parameters while keeping the
    base model frozen.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: LoRAConfigType
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Base model to adapt
            tokenizer: Tokenizer for the model
            training_config: Training configuration
            lora_config: LoRA-specific configuration
        """
        super().__init__(model, tokenizer, training_config)
        self.lora_config = lora_config
        self.peft_model: Optional[PeftModel] = None
        self.metrics_tracker = MetricsTracker()
        
        # Log LoRA config
        self._log_lora_config()
    
    def _log_lora_config(self) -> None:
        """Log LoRA configuration."""
        self.logger.info("LoRA Configuration:")
        self.logger.info(f"  Rank (r): {self.lora_config.r}")
        self.logger.info(f"  Alpha: {self.lora_config.lora_alpha}")
        self.logger.info(f"  Dropout: {self.lora_config.lora_dropout}")
        self.logger.info(f"  Target modules: {self.lora_config.target_modules or 'auto-detect'}")
        self.logger.info(f"  Bias: {self.lora_config.bias}")
        self.logger.info(f"  Initialization: {self.lora_config.init_lora_weights}")
    
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model with LoRA adapters.
        
        Returns:
            Model with LoRA adapters applied
        """
        self.logger.info("Applying LoRA adapters...")
        
        # Detect target modules if not specified
        target_modules = self.lora_config.target_modules
        if target_modules is None:
            self.logger.info("Auto-detecting target modules...")
            target_modules = detect_target_modules(self.model)
            self.logger.info(f"Detected modules: {target_modules}")
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=self.lora_config.fan_in_fan_out,
            init_lora_weights=self.lora_config.init_lora_weights
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # Log trainable parameters
        self.peft_model.print_trainable_parameters()
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.peft_model.enable_input_require_grads()
            self.logger.info("Enabled gradient checkpointing")
        
        return self.peft_model
    
    def get_training_args(self) -> TrainingArguments:
        """
        Build HuggingFace TrainingArguments.
        
        Returns:
            TrainingArguments configured from config
        """
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
            report_to="none",  # Disable wandb/tensorboard for now
            seed=self.config.seed,
            gradient_checkpointing=self.config.gradient_checkpointing,
            # Additional optimizations
            dataloader_num_workers=4,
            remove_unused_columns=True,
            label_names=["labels"],
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
        Execute LoRA training.
        
        Args:
            model: Model with LoRA adapters
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
            mlm=False  # Causal LM (not masked LM)
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[self._create_callback()]
        )
        
        # Train
        self.logger.info("Starting training...")
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
        )
        
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
        
        self.logger.info("Training complete!")
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
                        if loss != loss:  # NaN check
                            raise NaNLossError(state.global_step, trainer_instance.state.best_loss)
                        
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
        Save LoRA adapters.
        
        Args:
            output_dir: Directory to save to
        """
        if self.peft_model is not None:
            self.peft_model.save_pretrained(output_dir)
            self.logger.info(f"Saved LoRA adapters to: {output_dir}")
    
    def cleanup(self) -> None:
        """Cleanup after training."""
        # Optional: merge adapters back into base model
        # (Usually not done for LoRA to keep adapters separate)
        pass
    
    def merge_and_save(self, output_dir: Optional[Path] = None) -> None:
        """
        Merge LoRA adapters into base model and save.
        
        This creates a standalone model without needing PEFT.
        
        Args:
            output_dir: Directory to save merged model
        """
        if self.peft_model is None:
            raise TrainingError("Model not trained yet")
        
        output_dir = output_dir or (self.config.output_dir / "merged")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Merging LoRA adapters into base model...")
        
        # Merge adapters
        merged_model = self.peft_model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Saved merged model to: {output_dir}")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def train_with_lora(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    training_config: TrainingConfig,
    lora_config: LoRAConfigType,
    eval_dataset: Optional[Dataset] = None
) -> TrainingResult:
    """
    Convenience function to train with LoRA.
    
    Args:
        model: Base model
        tokenizer: Tokenizer
        train_dataset: Training data
        training_config: Training configuration
        lora_config: LoRA configuration
        eval_dataset: Optional evaluation data
    
    Returns:
        Training results
    
    Example:
        >>> from finetune_cli.models.loader import load_model_and_tokenizer
        >>> from finetune_cli.data import prepare_dataset
        >>> from finetune_cli.core.config import ConfigBuilder
        >>> from finetune_cli.core.types import TrainingMethod
        >>> 
        >>> # Build config
        >>> config = ConfigBuilder() \\
        ...     .with_model("gpt2") \\
        ...     .with_training(TrainingMethod.LORA, "./output") \\
        ...     .with_lora(r=8, lora_alpha=32) \\
        ...     .build()
        >>> 
        >>> # Load model
        >>> model, tokenizer = load_model_and_tokenizer(config.model.to_config())
        >>> 
        >>> # Prepare data
        >>> dataset = prepare_dataset(...)
        >>> 
        >>> # Train
        >>> result = train_with_lora(
        ...     model, tokenizer, dataset,
        ...     config.training.to_config(),
        ...     config.lora.to_config()
        ... )
    """
    trainer = LoRATrainer(model, tokenizer, training_config, lora_config)
    return trainer.train(train_dataset, eval_dataset)