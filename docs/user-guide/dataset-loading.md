# Dataset Loading

Finetune CLI supports flexible dataset loading options.

## Local JSON Files

The simplest format is a JSON list of objects with `input` and `output` keys:

```json
[
  {
    "input": "User prompt here",
    "output": "Desired model response"
  },
  ...
]
```

## HuggingFace Datasets

You can load datasets directly from the HuggingFace Hub (implementation pending in unified CLI, currently falls back to local/sample).

## Data Processing

The CLI automatically handles:
*   Tokenization
*   Padding/Truncation (max length 512)
*   Batching

Ensure your data is clean and representative of the task you want the model to perform.
