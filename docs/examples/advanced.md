# Advanced Examples

## QLoRA on Llama-2-7b

**Goal**: Fine-tune a 7B parameter model on a consumer GPU (e.g., RTX 3060).

**Prerequisites**:
*   `bitsandbytes` installed
*   HuggingFace token for Llama-2 access

**Settings**:
*   Model: `meta-llama/Llama-2-7b-hf`
*   Technique: `QLoRA`
*   Bits: `4`
*   Quant Type: `nf4`
*   Rank: `64` (Higher rank possible due to memory savings)

## Prompt Tuning for Multi-Tasking

**Goal**: Use one frozen model for multiple tasks.

**Task 1: Summarization**
*   Init Text: "Summarize the following article:"
*   Virtual Tokens: `20`
*   Output Dir: `./prompts/summarization`

**Task 2: Translation**
*   Init Text: "Translate to Spanish:"
*   Virtual Tokens: `20`
*   Output Dir: `./prompts/translation`

**Inference**:
Load the base model once, then swap the prompt adapter based on the user's request.
