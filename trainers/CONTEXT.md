# trainers/ — Context

All training logic lives here. Each trainer is a concrete subclass of `BaseTrainer`.

## Files

| File | Purpose |
|------|---------|
| `base.py` | `BaseTrainer` ABC + `TrainingResult` frozen dataclass. The `train()` method is final — subclasses only override `_setup_peft()`. |
| `lora_trainer.py` | `LoRATrainer` — attaches LoRA adapters, freezes base weights. |
| `qlora_trainer.py` | `QLoRATrainer` — extends LoRATrainer with 4-bit quantization via BitsAndBytes. |
| `full_trainer.py` | `FullFineTuner` — unfreezes all params, no PEFT. Issues VRAM warning for models >1B params. |
| `instruction_trainer.py` | `InstructionTrainer` — extends LoRATrainer. Reformats alpaca-style `{instruction, input, response}` datasets before training. Skips reformatting if `input_ids` or `text` column already present. |
| `factory.py` | `TrainerFactory` — single entry point. Validates required configs, selects and instantiates the right trainer. |

## Adding a new trainer

1. Extend `BaseTrainer` (or `LoRATrainer` if PEFT-based)
2. Implement `_setup_peft(model) -> model` — this is the only required override
3. Add dispatch in `TrainerFactory.create()`
4. Export from `__init__.py`
5. Write mocked unit tests — no GPU, no HF downloads

## Rules

- `TrainingResult` is **frozen** — never add mutable fields
- `output_dir` must always be populated — downstream CLI depends on it
- VRAM / memory warnings belong in `__init__`, not `_setup_peft`
- Never call `hf_trainer.train()` directly outside `base.py` — use `TrainerFactory.train()`