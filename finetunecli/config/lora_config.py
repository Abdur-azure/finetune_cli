from dataclasses import dataclass
from .base_config import BaseConfig

@dataclass
class LoraConfig(BaseConfig):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
