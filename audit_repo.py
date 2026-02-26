"""
audit_repo.py — run this from your repo root to find missing files.

Usage:
    python audit_repo.py

Prints every file that should exist but doesn't.
Copy the missing ones from the outputs/ folder provided.
"""

from pathlib import Path

REQUIRED_FILES = [
    # Root
    "conftest.py",
    "pyproject.toml",
    "__init__.py",

    # Core
    "core/__init__.py",
    "core/exceptions.py",
    "core/types.py",
    "core/config.py",

    # Utils
    "utils/__init__.py",
    "utils/logging.py",

    # Models
    "models/__init__.py",
    "models/loader.py",

    # Trainers
    "trainers/__init__.py",
    "trainers/base.py",
    "trainers/lora_trainer.py",
    "trainers/qlora_trainer.py",
    "trainers/factory.py",

    # Evaluation
    "evaluation/__init__.py",
    "evaluation/metrics.py",
    "evaluation/benchmarker.py",

    # CLI
    "cli/__init__.py",
    "cli/main.py",

    # Tests
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_config.py",
    "tests/test_trainers.py",
    "tests/test_evaluation.py",
    "tests/test_integration.py",

    # Tasks
    "tasks/todo.md",
    "tasks/lessons.md",

    # Examples
    "examples/configs/lora_gpt2.yaml",
    "examples/configs/qlora_llama.yaml",

    # CI
    ".github/workflows/ci.yml",
]

root = Path(__file__).parent
missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
present = [f for f in REQUIRED_FILES if (root / f).exists()]

print(f"\n{'='*55}")
print(f"  REPO AUDIT — {root.name}")
print(f"{'='*55}")
print(f"  Present : {len(present)}/{len(REQUIRED_FILES)}")
print(f"  Missing : {len(missing)}/{len(REQUIRED_FILES)}")
print(f"{'='*55}")

if missing:
    print("\n  MISSING FILES (copy these from the output zip):\n")
    for f in missing:
        print(f"    ✗  {f}")
else:
    print("\n  All required files present. Run: pytest tests/ -v\n")

print()