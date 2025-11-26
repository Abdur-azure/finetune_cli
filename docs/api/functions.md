# Utility Functions

## `finetunecli.utils.dataset_loader`

### `load_json_dataset(path: str)`

Loads a JSON dataset from a local file.
*   **Returns**: `datasets.Dataset` object.

## `finetunecli.benchmarking.rouge_metric`

### `RougeMetric.compute(predictions, references)`

Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.

## `finetunecli.quantization.qlora.qlora_utils`

### `estimate_memory_savings(base_bits, quant_bits, lora_r, model_params)`

Estimates VRAM savings for QLoRA vs full fine-tuning.
