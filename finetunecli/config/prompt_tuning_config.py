from dataclasses import dataclass
from finetunecli.config.base_config import BaseConfig

@dataclass
class PromptTuningConfig(BaseConfig):
    """Configuration for Prompt Tuning fine-tuning"""
    
    # Prompt tuning specific parameters
    num_virtual_tokens: int = 20  # Number of soft prompt tokens
    prompt_tuning_init: str = "TEXT"  # Initialization method: "TEXT" or "RANDOM"
    prompt_tuning_init_text: str = "Classify if the text is positive or negative:"  # Text for initialization
    tokenizer_name_or_path: str = None  # Tokenizer for text initialization
    
    # Training parameters
    lr: float = 3e-2  # Higher learning rate for prompt tuning
    epochs: int = 5
    batch_size: int = 8
    
    # Task type
    task_type: str = "CAUSAL_LM"  # or "SEQ_2_SEQ_LM"
