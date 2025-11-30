"""
Trainer factory for automatic trainer selection and instantiation.

Provides registry pattern for extensibility and automatic trainer
selection based on training method.
"""

from typing import Dict, Type, Optional, Any
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset

from ..core.types import (
    TrainingMethod, TrainingConfig, TrainingResult,
    LoRAConfig, ModelConfig
)
from ..core.exceptions import (
    MethodNotImplementedError,
    UnsupportedMethodError,
    MissingConfigError
)
from ..utils.logging import get_logger
from .base import BaseTrainer
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoRATrainer
from .full_finetuner import FullFineTuner


logger = get_logger(__name__)


# ============================================================================
# TRAINER REGISTRY
# ============================================================================


class TrainerRegistry:
    """
    Registry for trainer implementations.
    
    Enables dynamic trainer selection based on training method
    and allows registration of custom trainers.
    """
    
    def __init__(self):
        self._trainers: Dict[TrainingMethod, Type[BaseTrainer]] = {}
        self._register_default_trainers()
    
    def _register_default_trainers(self) -> None:
        """Register built-in trainers."""
        self.register(TrainingMethod.LORA, LoRATrainer)
        self.register(TrainingMethod.QLORA, QLoRATrainer)
        self.register(TrainingMethod.FULL_FINETUNING, FullFineTuner)
        
        logger.debug("Registered default trainers")
    
    def register(self, method: TrainingMethod, trainer_class: Type[BaseTrainer]) -> None:
        """
        Register a trainer for a specific method.
        
        Args:
            method: Training method
            trainer_class: Trainer class to register
        """
        self._trainers[method] = trainer_class
        logger.debug(f"Registered {trainer_class.__name__} for {method.value}")
    
    def get_trainer_class(self, method: TrainingMethod) -> Type[BaseTrainer]:
        """
        Get trainer class for method.
        
        Args:
            method: Training method
        
        Returns:
            Trainer class
        
        Raises:
            MethodNotImplementedError: If method not registered
        """
        if method not in self._trainers:
            raise MethodNotImplementedError(method.value)
        
        return self._trainers[method]
    
    def is_implemented(self, method: TrainingMethod) -> bool:
        """
        Check if method is implemented.
        
        Args:
            method: Training method to check
        
        Returns:
            True if implemented
        """
        return method in self._trainers
    
    def list_implemented_methods(self) -> list[TrainingMethod]:
        """
        Get list of implemented methods.
        
        Returns:
            List of implemented training methods
        """
        return list(self._trainers.keys())


# Global registry instance
_registry = TrainerRegistry()


# ============================================================================
# TRAINER FACTORY
# ============================================================================


class TrainerFactory:
    """
    Factory for creating trainers with proper configuration.
    
    Handles method-specific requirements and validates configuration.
    """
    
    @staticmethod
    def create_trainer(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None
    ) -> BaseTrainer:
        """
        Create appropriate trainer based on configuration.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            training_config: Training configuration
            lora_config: LoRA config (required for LoRA/QLoRA)
            model_config: Model config (required for QLoRA)
        
        Returns:
            Configured trainer instance
        
        Raises:
            MethodNotImplementedError: If method not implemented
            MissingConfigError: If required config is missing
        """
        method = training_config.method
        
        # Get trainer class
        trainer_class = _registry.get_trainer_class(method)
        
        # Create trainer with method-specific requirements
        if method == TrainingMethod.LORA:
            if lora_config is None:
                raise MissingConfigError("lora_config", "LoRA training")
            return LoRATrainer(model, tokenizer, training_config, lora_config)
        
        elif method == TrainingMethod.QLORA:
            if lora_config is None:
                raise MissingConfigError("lora_config", "QLoRA training")
            if model_config is None:
                raise MissingConfigError("model_config", "QLoRA training")
            return QLoRATrainer(model, tokenizer, training_config, lora_config, model_config)
        
        elif method == TrainingMethod.FULL_FINETUNING:
            return FullFineTuner(model, tokenizer, training_config)
        
        else:
            # For extensibility: call constructor with available args
            try:
                return trainer_class(model, tokenizer, training_config)
            except TypeError:
                raise MethodNotImplementedError(
                    f"Trainer {trainer_class.__name__} has incompatible constructor"
                )
    
    @staticmethod
    def train(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        training_config: TrainingConfig,
        lora_config: Optional[LoRAConfig] = None,
        model_config: Optional[ModelConfig] = None,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[Path] = None
    ) -> TrainingResult:
        """
        Create trainer and execute training in one call.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training data
            training_config: Training configuration
            lora_config: Optional LoRA configuration
            model_config: Optional model configuration
            eval_dataset: Optional evaluation data
            resume_from_checkpoint: Optional checkpoint to resume from
        
        Returns:
            Training results
        """
        # Create trainer
        trainer = TrainerFactory.create_trainer(
            model, tokenizer, training_config, lora_config, model_config
        )
        
        # Execute training
        return trainer.train(train_dataset, eval_dataset, resume_from_checkpoint)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def register_trainer(method: TrainingMethod, trainer_class: Type[BaseTrainer]) -> None:
    """
    Register a custom trainer.
    
    Args:
        method: Training method
        trainer_class: Trainer class
    
    Example:
        >>> class CustomTrainer(BaseTrainer):
        ...     def prepare_model(self): ...
        ...     def get_training_args(self): ...
        ...     def _execute_training(self, ...): ...
        ...     def _save_model(self, ...): ...
        ...     def cleanup(self): ...
        >>> 
        >>> register_trainer(TrainingMethod.CUSTOM, CustomTrainer)
    """
    _registry.register(method, trainer_class)


def get_available_methods() -> list[TrainingMethod]:
    """
    Get list of available training methods.
    
    Returns:
        List of implemented methods
    """
    return _registry.list_implemented_methods()


def is_method_available(method: TrainingMethod) -> bool:
    """
    Check if training method is available.
    
    Args:
        method: Training method to check
    
    Returns:
        True if method is implemented
    """
    return _registry.is_implemented(method)


# ============================================================================
# METHOD RECOMMENDER
# ============================================================================


class MethodRecommender:
    """
    Recommends training method based on constraints and requirements.
    """
    
    @staticmethod
    def recommend(
        model_size_params: int,
        available_vram_gb: float,
        task_complexity: str = "medium",
        needs_multiple_adapters: bool = False
    ) -> Dict[str, Any]:
        """
        Recommend training method based on constraints.
        
        Args:
            model_size_params: Model size in parameters
            available_vram_gb: Available VRAM in GB
            task_complexity: "simple", "medium", "complex"
            needs_multiple_adapters: Whether multiple task adapters needed
        
        Returns:
            Dictionary with recommendation and reasoning
        """
        model_size_gb = model_size_params * 4 / 1e9  # Rough FP32 estimate
        
        # Calculate memory requirements
        full_ft_memory = model_size_gb * 4  # Model + grads + optimizer + activations
        lora_memory = model_size_gb * 2     # Model + LoRA overhead
        qlora_memory = model_size_gb * 0.5  # Quantized model + LoRA
        
        recommendations = []
        
        # Check what fits
        if available_vram_gb >= full_ft_memory:
            if task_complexity == "complex" or model_size_params < 500e6:
                recommendations.append({
                    "method": TrainingMethod.FULL_FINETUNING,
                    "priority": 1,
                    "reason": "Sufficient memory and task benefits from full adaptation"
                })
        
        if available_vram_gb >= lora_memory:
            recommendations.append({
                "method": TrainingMethod.LORA,
                "priority": 2 if not needs_multiple_adapters else 1,
                "reason": "Good balance of quality and efficiency" + 
                         (" + supports multiple adapters" if needs_multiple_adapters else "")
            })
        
        if available_vram_gb >= qlora_memory:
            recommendations.append({
                "method": TrainingMethod.QLORA,
                "priority": 3 if recommendations else 1,
                "reason": "Most memory efficient, enables large models on consumer hardware"
            })
        
        if not recommendations:
            return {
                "recommendation": None,
                "reason": f"Insufficient VRAM. Need at least {qlora_memory:.1f}GB",
                "suggestions": [
                    "Use a smaller model",
                    "Reduce batch size",
                    "Use cloud GPU with more VRAM"
                ]
            }
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        best = recommendations[0]
        
        return {
            "recommendation": best["method"],
            "reason": best["reason"],
            "alternatives": recommendations[1:],
            "memory_estimates": {
                "full_finetuning": f"{full_ft_memory:.1f}GB",
                "lora": f"{lora_memory:.1f}GB",
                "qlora": f"{qlora_memory:.1f}GB",
                "available": f"{available_vram_gb:.1f}GB"
            }
        }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    training_config: TrainingConfig,
    lora_config: Optional[LoRAConfig] = None,
    model_config: Optional[ModelConfig] = None,
    eval_dataset: Optional[Dataset] = None
) -> TrainingResult:
    """
    High-level training function that handles everything.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training data
        training_config: Training configuration
        lora_config: Optional LoRA configuration
        model_config: Optional model configuration
        eval_dataset: Optional evaluation data
    
    Returns:
        Training results
    
    Example:
        >>> from finetune_cli.core.config import ConfigBuilder
        >>> from finetune_cli.core.types import TrainingMethod
        >>> from finetune_cli.models.loader import load_model_and_tokenizer
        >>> from finetune_cli.data import prepare_dataset
        >>> from finetune_cli.trainers import train_model
        >>> 
        >>> # Build configuration
        >>> config = ConfigBuilder() \\
        ...     .with_model("gpt2") \\
        ...     .with_dataset("./data.jsonl") \\
        ...     .with_training(TrainingMethod.LORA, "./output") \\
        ...     .with_lora(r=8, lora_alpha=32) \\
        ...     .build()
        >>> 
        >>> # Load model and data
        >>> model, tokenizer = load_model_and_tokenizer(config.model.to_config())
        >>> dataset = prepare_dataset(
        ...     config.dataset.to_config(),
        ...     config.tokenization.to_config(),
        ...     tokenizer
        ... )
        >>> 
        >>> # Train
        >>> result = train_model(
        ...     model, tokenizer, dataset,
        ...     config.training.to_config(),
        ...     config.lora.to_config()
        ... )
        >>> 
        >>> print(f"Final loss: {result.final_loss:.4f}")
    """
    return TrainerFactory.train(
        model, tokenizer, train_dataset,
        training_config, lora_config, model_config,
        eval_dataset
    )