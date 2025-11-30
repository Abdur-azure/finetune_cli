"""
Trainers package for LLM fine-tuning.

Provides implementations for various training methods:
- LoRA: Parameter-efficient fine-tuning
- QLoRA: Quantized LoRA for memory efficiency
- Full Fine-tuning: Train all parameters

High-level interface:
- train_model: One-stop training function
- TrainerFactory: Dynamic trainer creation
- register_trainer: Add custom trainers
"""

from .base import (
    BaseTrainer,
    TrainingState,
    MetricsTracker
)

from .lora_trainer import (
    LoRATrainer,
    train_with_lora
)

from .qlora_trainer import (
    QLoRATrainer,
    train_with_qlora,
    get_qlora_best_practices
)

from .full_finetuner import (
    FullFineTuner,
    train_full_finetuning,
    get_finetuning_comparison
)

from .factory import (
    TrainerFactory,
    TrainerRegistry,
    MethodRecommender,
    train_model,
    register_trainer,
    get_available_methods,
    is_method_available
)


__all__ = [
    # High-level functions
    'train_model',
    'train_with_lora',
    'train_with_qlora',
    'train_full_finetuning',
    
    # Factory & Registry
    'TrainerFactory',
    'TrainerRegistry',
    'MethodRecommender',
    'register_trainer',
    'get_available_methods',
    'is_method_available',
    
    # Trainer classes
    'BaseTrainer',
    'LoRATrainer',
    'QLoRATrainer',
    'FullFineTuner',
    
    # Utilities
    'TrainingState',
    'MetricsTracker',
    'get_qlora_best_practices',
    'get_finetuning_comparison',
]