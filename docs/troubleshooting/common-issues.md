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
