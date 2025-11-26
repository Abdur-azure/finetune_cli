# LLMFineTuner Class

The core class orchestrating the fine-tuning process.

## `finetunecli.training.trainer.LLMFineTuner`

### `__init__(self, model_name: str, dataset_path: str, output_dir: str)`

Initialize the fine-tuner.

*   `model_name`: HuggingFace model ID or local path.
*   `dataset_path`: Path to training data.
*   `output_dir`: Directory to save results.

### `load_model(self)`

Loads the base model and tokenizer. Handles device mapping (CPU/GPU).

### `prepare_dataset(self)`

Loads and processes the dataset.
*   Tokenizes input text.
*   Handles padding and truncation.

### `setup_lora(self, r=8, alpha=32, dropout=0.1)`

Configures the model for LoRA fine-tuning using `peft`.

### `train(self, epochs=3, batch_size=4, lr=2e-4)`

Executes the training loop.
*   Uses `transformers.Trainer`.
*   Saves checkpoints.

### `benchmark(self, metric="rouge")`

Evaluates the model using the specified metric.
