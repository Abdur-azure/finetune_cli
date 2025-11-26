# Common Issues

## CUDA Out of Memory (OOM)

**Symptoms**: Training crashes with `RuntimeError: CUDA out of memory`.

**Solutions**:
1.  **Reduce Batch Size**: Try `batch_size=1` or `2`.
2.  **Use Gradient Accumulation**: Simulate larger batches.
3.  **Switch to QLoRA**: Use 4-bit quantization.
4.  **Reduce Context Length**: If your data is very long, truncate it.

## bitsandbytes Not Found

**Symptoms**: `ImportError: No module named 'bitsandbytes'` when using QLoRA.

**Solutions**:
1.  Install it: `pip install bitsandbytes`.
2.  **Windows**: You may need a specific version or build for Windows. Check the `bitsandbytes-windows` package if the official one fails.

## Dataset Errors

**Symptoms**: `KeyError: 'input'`

**Solutions**:
Ensure your JSON file has the correct structure. It must be a list of objects, each having `input` and `output` keys.
<<<<<<< Updated upstream
=======

## ImportError: cannot import name 'app'

**Symptoms**: `ImportError: cannot import name 'app' from 'finetunecli'` when running `finetune-cli`.

**Solutions**:
1. Reinstall the package: `pip install -e .`
2. Ensure you're in the virtual environment if using one.
3. Check that `finetunecli/__init__.py` contains the `app` definition.

## Command Not Found: finetune-cli

**Symptoms**: `finetune-cli: command not found` or similar error.

**Solutions**:
1. **Activate Virtual Environment**: If using a virtual environment, activate it first:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
2. **Run Directly**: Use the full path: `.\venv\Scripts\finetune-cli` (Windows) or `./venv/bin/finetune-cli` (Linux/Mac).
3. **Use Python Module**: Run as a module: `python -m finetunecli`
4. **Reinstall Package**: `pip install -e .` to ensure entry points are registered.

## QLoRA Target Modules Error

**Symptoms**: `ValueError: Target modules {'q_proj', 'k_proj', 'v_proj', 'o_proj'} not found in the base model`.

**Solutions**:
This has been fixed in v0.2.1. The system now auto-detects the correct target modules for your model architecture. Update to the latest version.

## Benchmark AttributeError

**Symptoms**: `AttributeError: 'str' object has no attribute 'get'` during benchmarking.

**Solutions**:
This has been fixed in v0.2.1. The issue was with dataset sampling. Update to the latest version.
>>>>>>> Stashed changes
