"""
QLoRA Model Builder
Implements 4-bit quantization with LoRA adapters for memory-efficient fine-tuning
"""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def build_qlora_model(model_name, r, alpha, dropout, bits=4, quant_type="nf4", use_double_quant=True):
    """
    Build a QLoRA model with 4-bit quantization
    
    Args:
        model_name: HuggingFace model name or path
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout
        bits: Quantization bits (4 or 8)
        quant_type: Quantization type ("nf4" or "fp4")
        use_double_quant: Whether to use double quantization
    
    Returns:
        PEFT model with QLoRA adapters
    """
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # More comprehensive than basic LoRA
        bias="none"
    )
    
    # Apply LoRA to quantized model
    model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model
