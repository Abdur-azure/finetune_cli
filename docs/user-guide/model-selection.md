# Model Selection

Finetune CLI supports a wide range of Large Language Models.

## HuggingFace Models

You can use any Causal LM available on the HuggingFace Hub. Simply provide the repository ID:

*   `gpt2`
*   `facebook/opt-125m`
*   `meta-llama/Llama-2-7b-hf`
*   `mistralai/Mistral-7B-v0.1`

## Local Models

You can also provide an absolute path to a local directory containing a HuggingFace-compatible model:

*   `/path/to/my/local/model`

## Considerations

*   **Memory**: Ensure your GPU has enough VRAM for the selected model. Use QLoRA for larger models on consumer hardware.
*   **Access**: Some models (like Llama 2) require accepting license terms on HuggingFace and logging in with a token.
