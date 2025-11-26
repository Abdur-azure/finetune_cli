# User Guide Overview

Finetune CLI provides a unified, interactive workflow for fine-tuning LLMs.

## The 12-Step Workflow

The CLI guides you through a structured process to ensure successful fine-tuning:

1.  **Model Selection**: Choose your base model.
2.  **Dataset Selection**: Load your training data.
3.  **Technique Selection**: Choose the best fine-tuning method (LoRA, QLoRA, etc.).
4.  **Benchmark Selection**: Decide how to evaluate success.
5.  **Output Configuration**: Set where to save results.
6.  **Base Model Benchmarking**: Establish a baseline performance.
7.  **Parameter Configuration**: Fine-tune hyperparameters.
8.  **Training**: Execute the fine-tuning process.
9.  **Fine-tuned Benchmarking**: Evaluate the new model.
10. **Comparison**: Compare Before vs. After results.
11. **Saving**: Persist model artifacts.
12. **Upload**: Publish to HuggingFace Hub.

## Supported Techniques

*   **Quantization**:
    *   **LoRA**: Standard Low-Rank Adaptation.
    *   **QLoRA**: 4-bit Quantized LoRA for low memory.
    *   **Prompt Tuning**: Soft prompt optimization.
    *   **Prefix Tuning**: (Coming soon)
*   **Distillation**: (Coming soon)
*   **Pruning**: (Coming soon)

## Supported Benchmarks

*   **ROUGE**: For text generation quality.
*   **BLEU**: (Coming soon)
*   **BERTScore**: (Coming soon)
