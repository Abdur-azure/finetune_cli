# Use Cases

Choosing the right technique for your needs.

## When to use LoRA?

*   **Scenario**: You have a decent GPU (16GB+) and want a balance of speed and quality.
*   **Pros**: Standard, well-supported, fast training.
*   **Cons**: Higher memory usage than QLoRA.

## When to use QLoRA?

*   **Scenario**: You have limited VRAM (e.g., 8GB-12GB) but want to fine-tune large models (7B+).
*   **Pros**: Massive memory savings (up to 60%), allows training 7B models on consumer cards.
*   **Cons**: Slightly slower training due to quantization overhead.

## When to use Prompt Tuning?

*   **Scenario**: You need to serve many different tasks from one model, or storage is a major concern.
*   **Pros**: Tiny artifacts (KB), base model stays frozen, easy to switch tasks.
*   **Cons**: May not reach the same peak performance as LoRA for complex tasks.
