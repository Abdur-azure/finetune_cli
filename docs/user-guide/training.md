# Training Process

## What Happens During Training?

1.  **Model Loading**: The base model is loaded (potentially quantized for QLoRA).
2.  **Adapter Attachment**: PEFT adapters (LoRA/Prompt Tuning) are attached to the model.
3.  **Data Preparation**: Dataset is tokenized and batched.
4.  **Optimization Loop**:
    *   Forward pass: Calculate loss.
    *   Backward pass: Calculate gradients (only for adapter parameters).
    *   Optimizer step: Update weights.
5.  **Logging**: Loss and other metrics are logged every 10 steps.
6.  **Checkpointing**: Model is saved at the end of each epoch.

## Monitoring

The CLI displays a progress bar with:
*   Current step / Total steps
*   Training Loss
*   Estimated time remaining

## Hardware Requirements

*   **LoRA**: Requires ~14GB VRAM for a 7B model (fp16).
*   **QLoRA**: Requires ~6GB VRAM for a 7B model (4-bit).
*   **Prompt Tuning**: Requires very low VRAM, similar to inference.
