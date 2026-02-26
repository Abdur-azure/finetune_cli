# tests/ — Context

Unit tests (mocked, no GPU) + integration tests (real GPT-2, CPU ok).

## Files

| File | Covers |
|------|--------|
| `test_config.py` | ConfigBuilder, PipelineConfig validation, YAML round-trip |
| `test_trainers.py` | LoRATrainer, QLoRATrainer, TrainerFactory dispatch |
| `test_full_trainer.py` | FullFineTuner, VRAM warning logic |
| `test_instruction_trainer.py` | format_instruction_dataset, InstructionTrainer |
| `test_recommend.py` | recommend CLI command, config validity |
| `test_evaluation.py` | RougeMetric, BleuMetric, BenchmarkReport |
| `test_integration.py` | End-to-end: real GPT-2, 1 step, asserts adapter saved |
| `conftest.py` | Shared fixtures: mock_model, mock_tokenizer, tiny_dataset, tmp_output_dir |

## Rules

- **All unit tests mock HF Trainer and PEFT** — no real model loads, no GPU needed
- **Never allocate real tensors for size testing** — use `MagicMock` with `numel.return_value = N`
- **Integration tests are guarded** with `pytest.importorskip` at module level
- `conftest.py` at repo root handles `sys.path` — never rely on `PYTHONPATH`
- New trainers always get their own test file, not tacked onto `test_trainers.py`

## Running tests

```bash
# Fast (unit only)
pytest tests/ -v --ignore=tests/test_integration.py

# Full (needs torch + transformers)
pytest tests/test_integration.py -v -s

# All
pytest tests/ -v
```