# Tests

This directory contains test scripts for the finetune-cli project.

## Running Tests

Run all tests:
```bash
python -m unittest discover tests
```

Run a specific test:
```bash
python -m unittest tests.test_benchmark_fix
python -m unittest tests.test_qlora_modules
```

## Test Files

- `test_benchmark_fix.py`: Tests for the benchmark dataset sampling fix
- `test_qlora_modules.py`: Tests for QLoRA target module auto-detection
