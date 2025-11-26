# LoRA & QLoRA Configuration

Understanding the hyperparameters is key to successful fine-tuning.

## LoRA Parameters

*   **Rank (r)**: The dimension of the low-rank matrices.
    *   *Typical values*: 8, 16, 32, 64.
    *   *Effect*: Higher rank = more trainable parameters = potentially better performance but higher memory usage.
*   **Alpha**: Scaling factor for LoRA weights.
    *   *Rule of thumb*: Set alpha to 2x the rank (e.g., if r=16, alpha=32).
*   **Dropout**: Probability of dropping out neurons during training.
    *   *Typical values*: 0.05 - 0.1.
    *   *Effect*: Prevents overfitting.

## QLoRA Specifics

QLoRA adds quantization parameters:

*   **Bits**: 4 or 8.
    *   *Recommendation*: 4-bit for maximum memory savings.
*   **Quant Type**: `nf4` (Normal Float 4) or `fp4`.
    *   *Recommendation*: `nf4` is information-theoretically optimal for normal distributions.
*   **Double Quant**: Quantize the quantization constants.
    *   *Recommendation*: Enable for extra memory savings.

## Prompt Tuning Parameters

*   **Virtual Tokens**: Number of soft prompt tokens prepended to input.
    *   *Typical values*: 8 - 100.
*   **Initialization**:
    *   `TEXT`: Initialize from a natural language description (Recommended).
    *   `RANDOM`: Random initialization (Harder to train).
