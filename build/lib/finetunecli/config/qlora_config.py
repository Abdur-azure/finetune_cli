from dataclasses import dataclass
from finetunecli.config.base_config import BaseConfig

@dataclass
class QLoraConfig(BaseConfig):
    """Configuration for QLoRA (Quantized LoRA) fine-tuning"""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    
    # QLoRA-specific parameters
    bits: int = 4  # 4-bit quantization
    quant_type: str = "nf4"  # Normal Float 4
    use_double_quant: bool = True  # Double quantization
    compute_dtype: str = "float16"  # Compute dtype for training
