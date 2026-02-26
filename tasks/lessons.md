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

## Pattern: Root conftest.py is the most reliable pytest path fix
Never rely on PYTHONPATH env vars or `pip install -e .` for test discovery.
A root conftest.py with `sys.path.insert(0, repo_root)` works universally
across Windows/macOS/Linux and all pytest invocation styles.

## Pattern: pyproject.toml over setup.py for Windows compatibility
setup.py calls read_text() without encoding= which crashes on Windows cp1252.
pyproject.toml is declarative — no Python code runs during install, no
encoding bugs possible. Always use pyproject.toml for new projects.

## Pattern: Ship an audit script with generated file sets
When delivering many files across sessions, include audit_repo.py so the user
can instantly see which files are missing from their local repo without
manually comparing directory listings.

## Pattern: data/__init__.py must mirror CLI imports exactly
The CLI does `from ..data import quick_load` — if __init__.py doesn't
re-export quick_load the import fails silently at runtime not test time.
Always trace the full import chain from CLI → package → module when
verifying data pipeline wiring.

## Pattern: Deprecation shims beat deletion
Never delete v1 files users might be running. Replace the body with a
DeprecationWarning + sys.exit() that points to the v2 command.
This gives users a clear migration message instead of a FileNotFoundError.

## Pattern: Verify docs with term-presence checks, not just line counts
When rewriting docs, assert that key v2 identifiers (CLI commands, class names,
config keys) are present. A doc that builds without errors can still describe v1.
Always check content, not just structure.

## Pattern: Example data scripts must be zero-dependency
generate_sample_data.py uses only stdlib (json, random, pathlib).
Never import project modules or third-party libs in example generators —
they must run before pip install -e . succeeds.

## Pattern: Name sprints — makes changelog and reviews easier to navigate
Sprint names (e.g. "First Run", "Expand") give reviewers instant context
without reading every commit. Add to todo.md header and CHANGELOG.

## Pattern: Guard lora_config by method — not unconditionally
cli/main.py was calling .with_lora() for every method including full_finetuning.
Always check the method set before attaching method-specific config.
Pattern: define _LORA_METHODS = {LORA, QLORA, INSTRUCTION_TUNING} and gate on it.

## Pattern: CHANGELOG and audit_repo.py are first-class deliverables
After every sprint, update both. CHANGELOG tells humans what changed.
audit_repo.py tells Claude (and contributors) what files must exist.
Skipping these creates session drift — the next session starts blind.

## Pattern: Write test_cli_train.py for every new method added to CLI
When a new TrainingMethod is wired into the factory, add a CLI-level test
that invokes the train command with --method <new_method> and asserts
exit_code == 0. Mock the training stack — this is a wiring test, not a
training test. Catches missing lora_config guards before they hit production.