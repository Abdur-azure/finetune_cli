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

from .base import BaseTrainer, TrainingResult
from .lora_trainer import LoRATrainer
from .qlora_trainer import QLoRATrainer
from .factory import TrainerFactory

__all__ = [
    "BaseTrainer", "TrainingResult",
    "LoRATrainer", "QLoRATrainer", "TrainerFactory",
]