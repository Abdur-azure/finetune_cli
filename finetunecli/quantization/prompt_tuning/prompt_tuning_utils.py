"""
Prompt Tuning Utility Functions
Helper functions for prompt tuning
"""

def get_recommended_num_tokens(model_size_b):
    """
    Get recommended number of virtual tokens based on model size
    
    Args:
        model_size_b: Model size in billions of parameters
    
    Returns:
        Recommended number of virtual tokens
    """
    if model_size_b < 1:
        return 8  # Small models (< 1B)
    elif model_size_b < 3:
        return 20  # Medium models (1-3B)
    elif model_size_b < 10:
        return 50  # Large models (3-10B)
    else:
        return 100  # Very large models (> 10B)


def get_recommended_learning_rate(num_virtual_tokens):
    """
    Get recommended learning rate based on number of virtual tokens
    
    Args:
        num_virtual_tokens: Number of soft prompt tokens
    
    Returns:
        Recommended learning rate
    """
    if num_virtual_tokens < 10:
        return 5e-2  # Fewer tokens, higher LR
    elif num_virtual_tokens < 50:
        return 3e-2  # Standard
    else:
        return 1e-2  # More tokens, lower LR


def generate_init_text_examples():
    """Generate example initialization texts for different tasks"""
    return {
        "classification": "Classify if the text is positive or negative:",
        "summarization": "Summarize the following text:",
        "question_answering": "Answer the question based on the context:",
        "translation": "Translate the following text to English:",
        "generation": "Generate a creative response:",
        "sentiment": "Determine the sentiment of this text:",
        "ner": "Extract named entities from the text:",
        "paraphrase": "Paraphrase the following sentence:",
    }


def estimate_trainable_params(num_virtual_tokens, embedding_dim=768):
    """
    Estimate trainable parameters for prompt tuning
    
    Args:
        num_virtual_tokens: Number of soft prompt tokens
        embedding_dim: Model embedding dimension
    
    Returns:
        Dictionary with parameter estimates
    """
    trainable_params = num_virtual_tokens * embedding_dim
    
    # Typical model sizes for comparison
    model_sizes = {
        "GPT-2 Small": 124_000_000,
        "GPT-2 Medium": 355_000_000,
        "GPT-2 Large": 774_000_000,
        "Llama-7B": 7_000_000_000,
    }
    
    comparisons = {}
    for model_name, total_params in model_sizes.items():
        percentage = (trainable_params / total_params) * 100
        comparisons[model_name] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "percentage": percentage
        }
    
    return {
        "trainable_params": trainable_params,
        "comparisons": comparisons
    }


def print_prompt_tuning_info(num_virtual_tokens, model_name="GPT-2"):
    """Print information about prompt tuning configuration"""
    print(f"\n{'='*60}")
    print(f"  Prompt Tuning Configuration")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Virtual Tokens: {num_virtual_tokens}")
    print(f"\n💡 How it works:")
    print(f"   • {num_virtual_tokens} 'soft prompts' are prepended to your input")
    print(f"   • Only these prompt embeddings are trained")
    print(f"   • The entire model stays frozen")
    print(f"   • Extremely parameter-efficient!")
    
    estimates = estimate_trainable_params(num_virtual_tokens)
    print(f"\n📊 Trainable Parameters: {estimates['trainable_params']:,}")
    print(f"   That's only ~0.001% of a typical LLM!")
    print(f"{'='*60}\n")
