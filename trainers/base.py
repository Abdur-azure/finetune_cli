"""
Abstract base classes and interfaces for trainers.

Defines the contract that all trainer implementations must follow,
along with shared utilities and base functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import time

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments

from ..core.types import TrainingConfig, TrainingResult, TrainingMethod
from ..core.exceptions import TrainingError, TrainingFailedError
from ..utils.logging import get_logger, log_model_info, LogProgress


logger = get_logger(__name__)


# ============================================================================
# TRAINING STATE
# ============================================================================


@dataclass
class TrainingState:
    """
    Tracks the state of training process.
    
    Used for checkpointing, resuming, and monitoring.
    """
    
    # Identifiers
    run_id: str
    method: TrainingMethod
    start_time: datetime
    
    # Progress
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    
    # Metrics
    current_loss: float = float('inf')
    best_loss: float = float('inf')
    best_epoch: int = 0
    loss_history: list = field(default_factory=list)
    
    # Status
    is_training: bool = False
    is_completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'method': self.method.value,
            'start_time': self.start_time.isoformat(),
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'current_loss': self.current_loss,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'loss_history': self.loss_history,
            'is_training': self.is_training,
            'is_completed': self.is_completed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        data['method'] = TrainingMethod(data['method'])
        return cls(**data)


# ============================================================================
# ABSTRACT TRAINER
# ============================================================================


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    
    Provides common functionality and defines the interface that
    all trainers must implement.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # State
        self.state: Optional[TrainingState] = None
        self.start_time: Optional[float] = None
        
        # Callbacks
        self._callbacks: list = []
        
        # Validate setup
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Validate trainer setup."""
        # Check model is in correct mode
        if not self.model.training:
            self.model.train()
            self.logger.debug("Set model to training mode")
        
        # Check output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Output directory: {self.config.output_dir}")
        
        # Log model info
        log_model_info(self.logger, self.model)
    
    # ========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ========================================================================
    
    @abstractmethod
    def prepare_model(self) -> PreTrainedModel:
        """
        Prepare model for training (apply LoRA, setup optimizer, etc.).
        
        Returns:
            Prepared model ready for training
        """
        pass
    
    @abstractmethod
    def get_training_args(self) -> TrainingArguments:
        """
        Build HuggingFace TrainingArguments from config.
        
        Returns:
            TrainingArguments object
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup after training (merge adapters, clear cache, etc.).
        """
        pass
    
    # ========================================================================
    # TRAINING WORKFLOW
    # ========================================================================
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[Path] = None
    ) -> TrainingResult:
        """
        Main training method.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Optional checkpoint to resume from
        
        Returns:
            Training results
        
        Raises:
            TrainingError: If training fails
        """
        with LogProgress(self.logger, f"Training with {self.config.method.value}"):
            try:
                # Initialize state
                self._initialize_state()
                
                # Prepare model
                self.logger.info("Preparing model for training...")
                prepared_model = self.prepare_model()
                
                # Build training arguments
                training_args = self.get_training_args()
                
                # Execute training
                result = self._execute_training(
                    prepared_model,
                    train_dataset,
                    eval_dataset,
                    training_args,
                    resume_from_checkpoint
                )
                
                # Cleanup
                self.cleanup()
                
                # Mark complete
                self.state.is_completed = True
                self.state.is_training = False
                
                return result
                
            except KeyboardInterrupt:
                self.logger.warning("Training interrupted by user")
                self.state.is_training = False
                raise
            
            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                self.state.is_training = False
                raise TrainingFailedError(str(e), self.state.current_epoch)
    
    def _initialize_state(self) -> None:
        """Initialize training state."""
        import uuid
        
        self.state = TrainingState(
            run_id=str(uuid.uuid4())[:8],
            method=self.config.method,
            start_time=datetime.now()
        )
        self.state.is_training = True
        self.start_time = time.time()
        
        self.logger.info(f"Training run ID: {self.state.run_id}")
    
    @abstractmethod
    def _execute_training(
        self,
        model: PreTrainedModel,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: TrainingArguments,
        resume_from_checkpoint: Optional[Path]
    ) -> TrainingResult:
        """
        Execute the actual training loop.
        
        Args:
            model: Prepared model
            train_dataset: Training data
            eval_dataset: Optional evaluation data
            training_args: Training arguments
            resume_from_checkpoint: Optional checkpoint path
        
        Returns:
            Training results
        """
        pass
    
    # ========================================================================
    # SAVING & LOADING
    # ========================================================================
    
    def save(self, output_dir: Optional[Path] = None) -> None:
        """
        Save trained model and training state.
        
        Args:
            output_dir: Directory to save to (uses config.output_dir if None)
        """
        output_dir = output_dir or self.config.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving model to: {output_dir}")
        
        # Save model (implemented by subclass)
        self._save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        if self.state:
            import json
            state_file = output_dir / "training_state.json"
            with open(state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        
        self.logger.info("Model saved successfully")
    
    @abstractmethod
    def _save_model(self, output_dir: Path) -> None:
        """
        Save model-specific artifacts.
        
        Args:
            output_dir: Directory to save to
        """
        pass
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load training state
        import json
        state_file = checkpoint_path / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            self.state = TrainingState.from_dict(state_dict)
            self.logger.info(f"Resumed from epoch {self.state.current_epoch}")
    
    # ========================================================================
    # CALLBACKS
    # ========================================================================
    
    def add_callback(self, callback: Callable) -> None:
        """
        Add training callback.
        
        Args:
            callback: Callback function
        """
        self._callbacks.append(callback)
    
    def _trigger_callbacks(self, event: str, **kwargs) -> None:
        """
        Trigger all callbacks for an event.
        
        Args:
            event: Event name
            **kwargs: Event data
        """
        for callback in self._callbacks:
            try:
                callback(event, self.state, **kwargs)
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_device(self) -> torch.device:
        """Get training device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage for training.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Rough estimates (assuming float32)
        param_memory = param_count * 4 / 1e9  # Parameters
        grad_memory = trainable_params * 4 / 1e9  # Gradients
        optimizer_memory = trainable_params * 8 / 1e9  # Adam state
        activation_memory = self.config.batch_size * self.config.gradient_accumulation_steps * 0.5  # Rough estimate
        
        return {
            'parameters_gb': param_memory,
            'gradients_gb': grad_memory,
            'optimizer_gb': optimizer_memory,
            'activations_gb': activation_memory,
            'total_estimated_gb': param_memory + grad_memory + optimizer_memory + activation_memory
        }
    
    def log_training_info(self) -> None:
        """Log detailed training information."""
        self.logger.info("Training Configuration:")
        self.logger.info(f"  Method: {self.config.method.value}")
        self.logger.info(f"  Epochs: {self.config.num_epochs}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  Weight decay: {self.config.weight_decay}")
        self.logger.info(f"  Warmup ratio: {self.config.warmup_ratio}")
        self.logger.info(f"  LR scheduler: {self.config.lr_scheduler_type}")
        self.logger.info(f"  Max grad norm: {self.config.max_grad_norm}")
        self.logger.info(f"  FP16: {self.config.fp16}")
        self.logger.info(f"  BF16: {self.config.bf16}")
        self.logger.info(f"  Gradient checkpointing: {self.config.gradient_checkpointing}")
        
        # Memory estimation
        memory = self.estimate_memory_usage()
        self.logger.info("Estimated Memory Usage:")
        for key, value in memory.items():
            self.logger.info(f"  {key}: {value:.2f} GB")


# ============================================================================
# TRAINING METRICS TRACKER
# ============================================================================


class MetricsTracker:
    """
    Tracks and aggregates training metrics.
    """
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.step_metrics: Dict[int, Dict[str, float]] = {}
    
    def add(self, step: int, metric_name: str, value: float) -> None:
        """
        Add a metric value.
        
        Args:
            step: Training step
            metric_name: Name of metric
            value: Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        if step not in self.step_metrics:
            self.step_metrics[step] = {}
        self.step_metrics[step][metric_name] = value
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Optional[float]:
        """
        Get best value for metric.
        
        Args:
            metric_name: Metric name
            mode: 'min' or 'max'
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        if mode == 'min':
            return min(self.metrics[metric_name])
        else:
            return max(self.metrics[metric_name])
    
    def get_average(self, metric_name: str) -> Optional[float]:
        """Get average value for metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
    
    def get_history(self, metric_name: str) -> list:
        """Get full history for metric."""
        return self.metrics.get(metric_name, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metrics': self.metrics,
            'step_metrics': self.step_metrics
        }