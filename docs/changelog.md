# Changelog

<<<<<<< Updated upstream
=======
## [0.2.1] - 2025-11-26

### Fixed
- **ImportError Fix**: Fixed `ImportError: cannot import name 'app'` by moving app definition to `finetunecli/__init__.py`.
- **Command Not Found**: Fixed `finetune-cli` command not found issue by reinstalling package with correct entry points.
- **Benchmark AttributeError**: Fixed `AttributeError: 'str' object has no attribute 'get'` in benchmarking by implementing proper dataset sampling for HuggingFace Datasets.
- **QLoRA Target Modules**: Fixed `ValueError` in QLoRA training by implementing auto-detection of target modules for different model architectures (GPT-2, Llama, etc.).

### Improved
- **Learning Rate Input**: Updated CLI to accept scientific notation (e.g., `2e-4`) for learning rate input.
- **Test Organization**: Organized test scripts into `tests/` directory with proper structure.

>>>>>>> Stashed changes
## [0.2.0] - 2025-11-26

### Added
*   **Unified Interactive CLI**: New `finetune run` command with 12-step wizard.
*   **QLoRA Support**: 4-bit quantization for memory-efficient training.
*   **Prompt Tuning Support**: Parameter-efficient soft prompt training.
*   **Hierarchical Menus**: Organized technique selection.
*   **Documentation**: Comprehensive guides and API docs.

### Changed
*   Refactored core logic into modular `finetunecli` package.
*   Updated `requirements.txt` with `bitsandbytes`.

## [0.1.0] - Initial Release
*   Basic LoRA fine-tuning.
*   ROUGE benchmarking.
*   Simple CLI interface.
