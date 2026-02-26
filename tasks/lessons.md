# Lessons Learned

## Pattern: Read existing code before writing
Always search project knowledge for existing types and interfaces before
implementing new modules. Mis-matching signatures wastes iteration cycles.

## Pattern: Follow existing exception hierarchy
This project has a custom `FineTuneError` hierarchy in `core/exceptions.py`.
New errors must extend it — never raise raw Python builtins from module code.

## Pattern: Frozen dataclasses for results
Existing code uses `@dataclass(frozen=True)` for config objects. Results
(e.g. `TrainingResult`) should also be frozen — callers should not mutate them.

## Pattern: Use `get_logger(__name__)` not `logging.getLogger`
The project wraps loggers in `utils/logging.py`. Always import `get_logger`
from there to get consistent formatting and levels.

## Pattern: Trainer outputs must include model path
Downstream CLI and evaluation code needs to know where the saved model is.
`TrainingResult` must carry `output_dir: Path`.

## Pattern: Sandbox has no network — static verification is the fallback
When pip install fails due to network restrictions, use AST parsing +
cross-reference assertions to verify correctness before handing off.
Always note the exact `pytest` command the user needs to run locally.

## Pattern: Integration tests need `pytest.importorskip`
Heavy deps (torch, transformers, peft) should be guarded with
`pytest.importorskip` at module level so CI can skip gracefully
when packages aren't installed, rather than erroring.

## Pattern: Upload command needs adapter merge option
Users often want to upload the merged (non-adapter) model for portability.
Always expose `--merge-adapter` + `--base-model` alongside the default
adapter-only upload path.

## Pattern: Example configs are integration test fixtures
Write YAML example configs to be runnable — real model names, real dataset
paths (with placeholders clearly marked). They serve as both documentation
and fixture inputs for integration tests.