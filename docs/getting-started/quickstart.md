# Quick Start

Get up and running with Finetune CLI in minutes.

## Interactive Mode

The easiest way to use Finetune CLI is through the interactive wizard:

```bash
finetune-cli finetune run
```

This command launches a 12-step interactive workflow that guides you through:

1.  **Model Selection**: Choose any HuggingFace model or local path.
2.  **Dataset Selection**: Use local JSON files or HuggingFace datasets.
3.  **Technique Selection**: Choose from LoRA, QLoRA, or Prompt Tuning.
4.  **Benchmarking**: Select metrics like ROUGE to evaluate performance.
5.  **Training**: Configure parameters and start fine-tuning.
6.  **Evaluation**: Compare base vs. fine-tuned model performance.
7.  **Saving & Uploading**: Save your model and optionally upload to HuggingFace.

## Command Line Arguments

For automation or advanced usage, you can pass arguments directly to the specific subcommands (legacy mode):

```bash
# Train with LoRA
finetune-cli train start --model gpt2 --data ./data/train.json --out ./output

# Benchmark a model
finetune-cli benchmark rouge --model ./output --data ./data/test.json
```

## Next Steps

*   Follow the [First Fine-tune](first-finetune.md) guide for a detailed tutorial.
*   Check out the [User Guide](../user-guide/overview.md) for in-depth documentation.
