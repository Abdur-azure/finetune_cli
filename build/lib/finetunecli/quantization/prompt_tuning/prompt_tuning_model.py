"""
Prompt Tuning Model Builder
Implements soft prompt tuning where only virtual token embeddings are trained
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, get_peft_model, TaskType

def build_prompt_tuning_model(
    model_name,
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify if the text is positive or negative:",
    tokenizer_name_or_path=None
):
    """
    Build a model with Prompt Tuning
    
    Prompt Tuning adds trainable "soft prompts" (virtual tokens) to the input.
    Only these prompt embeddings are trained, the entire model stays frozen.
    
    Args:
        model_name: HuggingFace model name or path
        num_virtual_tokens: Number of soft prompt tokens to prepend
        prompt_tuning_init: Initialization method ("TEXT" or "RANDOM")
        prompt_tuning_init_text: Text to initialize prompts (if init="TEXT")
        tokenizer_name_or_path: Tokenizer for text initialization
    
    Returns:
        PEFT model with prompt tuning
    """
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Load tokenizer if needed for TEXT initialization
    tokenizer = None
    if prompt_tuning_init == "TEXT":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name
        )
    
    # Configure Prompt Tuning
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=prompt_tuning_init,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=prompt_tuning_init_text if prompt_tuning_init == "TEXT" else None,
        tokenizer_name_or_path=tokenizer_name_or_path or model_name if prompt_tuning_init == "TEXT" else None,
    )
    
    # Apply prompt tuning to model
    model = get_peft_model(base_model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def get_prompt_tuning_init_methods():
    """Get available initialization methods for prompt tuning"""
    return {
        "TEXT": "Initialize from text string (recommended)",
        "RANDOM": "Random initialization"
    }


def estimate_prompt_tuning_params(num_virtual_tokens, embedding_dim=768):
    """
    Estimate number of trainable parameters for prompt tuning
    
    Args:
        num_virtual_tokens: Number of soft prompt tokens
        embedding_dim: Model embedding dimension
    
    Returns:
        Number of trainable parameters
    """
    return num_virtual_tokens * embedding_dim
