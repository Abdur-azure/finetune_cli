# Frequently Asked Questions

### Can I fine-tune on CPU?
Technically yes, but it will be extremely slow. A GPU is highly recommended.

### How long does training take?
It depends on the model size, dataset size, and hardware.
*   GPT-2 on 100 examples: ~2 minutes.
*   Llama-2-7b on 1000 examples: ~1-2 hours on a decent GPU.

### Can I use my own base model?
Yes, as long as it's compatible with `transformers.AutoModelForCausalLM`.

### Where are the models saved?
By default, in a folder named `finetuned_<technique>` (e.g., `finetuned_lora`) in your current directory, or whatever path you specified.
