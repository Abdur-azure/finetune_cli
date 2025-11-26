"""
QLoRA Utility Functions
Helper functions for QLoRA fine-tuning
"""
import torch
from transformers import BitsAndBytesConfig

def get_bnb_config(bits=4, quant_type="nf4", use_double_quant=True, compute_dtype=torch.float16):
    """
    Get BitsAndBytes quantization configuration
    
    Args:
        bits: 4 or 8 bit quantization
        quant_type: "nf4" (Normal Float 4) or "fp4" (Float Point 4)
        use_double_quant: Whether to use nested quantization
        compute_dtype: Compute dtype for training
    
    Returns:
        BitsAndBytesConfig object
    """
    return BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype
    )

def print_model_size(model):
    """Print model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model size: {size_all_mb:.2f} MB')
    return size_all_mb

def estimate_memory_savings(base_bits=16, quant_bits=4, lora_r=8, model_params=125_000_000):
    """
    Estimate memory savings from QLoRA
    
    Args:
        base_bits: Base model precision (16 or 32)
        quant_bits: Quantization bits (4 or 8)
        lora_r: LoRA rank
        model_params: Number of model parameters
    
    Returns:
        Dictionary with memory estimates
    """
    # Base model memory
    base_memory_mb = (model_params * base_bits) / (8 * 1024 * 1024)
    
    # Quantized model memory
    quant_memory_mb = (model_params * quant_bits) / (8 * 1024 * 1024)
    
    # LoRA adapter memory (approximate)
    lora_params = model_params * 0.01 * (lora_r / 8)  # Rough estimate
    lora_memory_mb = (lora_params * base_bits) / (8 * 1024 * 1024)
    
    # Total QLoRA memory
    total_qlora_mb = quant_memory_mb + lora_memory_mb
    
    savings_mb = base_memory_mb - total_qlora_mb
    savings_percent = (savings_mb / base_memory_mb) * 100
    
    return {
        "base_model_mb": base_memory_mb,
        "quantized_model_mb": quant_memory_mb,
        "lora_adapters_mb": lora_memory_mb,
        "total_qlora_mb": total_qlora_mb,
        "savings_mb": savings_mb,
        "savings_percent": savings_percent
    }
