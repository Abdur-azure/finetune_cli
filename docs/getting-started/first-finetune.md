# Your First Fine-tune

This guide walks you through fine-tuning GPT-2 on a custom dataset using LoRA.

## Prerequisites

*   Finetune CLI installed (`pip install .`)
*   A dataset file (e.g., `data.json`)

## Step 1: Prepare Data

Create a file named `data.json` with your training examples:

```json
[
  {
    "input": "Translate to French: Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  },
  {
    "input": "Translate to French: The weather is nice.",
    "output": "Il fait beau."
  }
]
```

## Step 2: Run the CLI

Start the interactive session:

```bash
finetune-cli finetune run
```

## Step 3: Follow the Wizard

1.  **Model**: Enter `gpt2`.
2.  **Dataset**: Enter `./data.json`.
3.  **Technique**: Select `1` (Quantization) -> `1` (LoRA).
4.  **Benchmark**: Select `1` (ROUGE).
5.  **Output**: Press Enter to use default.
6.  **Base Benchmark**: The CLI will evaluate the base GPT-2 model.
7.  **Parameters**:
    *   Rank (r): `8`
    *   Alpha: `32`
    *   Dropout: `0.1`
    *   Epochs: `3`
8.  **Training**: Watch the progress bar as the model trains.
9.  **Evaluation**: See how much the model improved!

## Step 4: Use Your Model

Your fine-tuned model is saved in the output directory. You can load it using `peft`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(base_model, "./finetuned_lora")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("Translate to French: Good morning", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```
