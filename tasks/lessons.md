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

## Pattern: Use string concatenation not f-strings for multiline Panel content
When building Rich Panel content across multiple lines, avoid putting \n
inside f-strings — heredocs interpret them as literal newlines which breaks
the string literal. Instead build the string with concatenation:
    panel_text = "[header]\n" + f"field: {value}\n" ...
Or use a separate variable assigned before the Panel() call.

## Pattern: Implement lessons as commands, not just notes
lessons.md had "upload needs merge option" documented since Sprint 1.
It sat unbuilt for 4 sprints. During the drill, review lessons.md for
unactioned items and prioritise them as sprint tasks.

## Pattern: Install pytest-timeout whenever --timeout is used in CI
If ci.yml uses --timeout=N, pytest-timeout must be in the install step.
Missing it causes pytest to silently ignore the flag (no error, no timeout).
Always audit install step against all pytest flags used in the workflow.

## Pattern: tasks/CONTEXT.md sprint table drifts — update it every sprint
The sprint table in tasks/CONTEXT.md was frozen at Sprint 3 through Sprint 7.
Rule: when archiving a sprint in todo.md, also add a row to CONTEXT.md.

## Pattern: docs/index.md is a second source of truth — keep it in sync
Version number, test count, and component table in docs/index.md all drifted.
After each sprint, grep for hardcoded version strings and test counts and update them.

## Pattern: CI test path must match actual repo layout — verify against audit_repo.py
ci.yml had `pytest finetune_cli/tests/` but tests live at `tests/` from repo root.
Rule: after writing ci.yml, cross-check every path against audit_repo.py REQUIRED_FILES.
If audit_repo lists "tests/test_config.py" (no finetune_cli/ prefix), the CI path is `tests/`.

## Pattern: ruff config must use [tool.ruff.lint] not [tool.ruff]
Since ruff 0.1+, `select` and `ignore` belong under `[tool.ruff.lint]` in pyproject.toml.
Putting them under `[tool.ruff]` triggers a deprecation warning AND causes E902 errors
because ruff invalidates its own config. Always use the `lint` subsection.

## Pattern: PowerShell uses backtick for line continuation, not backslash
`pytest .\tests\test_cli_train.py -v \` — the trailing `\` on Windows is passed as a
literal argument (the drive root), causing pytest to collect the entire filesystem.
On Windows use backtick: `pytest .\tests\test_cli_train.py -v `
Or just keep commands on one line in the docs.

## Pattern: Tests in tests/ must use absolute imports, not relative
test_cli_train.py and test_merge.py used `from ..cli.main import app`.
Relative imports only work when the file is part of the package being traversed.
Tests in tests/ (repo root) are NOT inside finetune_cli/, so `..` is invalid.
Always use absolute imports in test files: `from finetune_cli.cli.main import app`.

## Pattern: ruff check path should be . not finetune_cli/
`ruff check finetune_cli/` in ci.yml caused E902 because ruff resolves paths
relative to pyproject.toml location. Use `ruff check .` and let pyproject.toml
[tool.ruff] control which files get linted via `include`/`exclude`.

## Pattern: DPO requires trl — guard with try/except ImportError inside train()
DPOTrainer imports trl inside the train() method, not at module level.
This keeps the trainer importable even without trl installed.
The ImportError is caught and re-raised as TrainingError with an install hint.
Same pattern should be used for any optional heavy dependency.

## Pattern: DPO datasets need column validation before training starts
TRL's DPOTrainer gives an opaque error if columns are missing.
Always call validate_dpo_dataset() immediately after unpacking splits,
before any model setup, so the error message is clear and fast.

## Pattern: Every trainer needs an offline sample dataset + local config
DPO was built in Sprint 8 but left pointing at Anthropic/hh-rlhf (network required).
LoRA and instruction tuning both have local data via generate_sample_data.py.
Rule: when adding a new trainer, also add a generator function to generate_sample_data.py
and update the example config to source: local_file before the sprint closes.

## Pattern: Optional heavy deps belong in [project.optional-dependencies]
trl is only needed for DPO. Adding it to core dependencies would force all users to
install it. Use optional groups: pip install "finetune-cli[dpo]".
Same pattern applies to any future optional-only dependency.

## Pattern: pyproject.toml version must be bumped every sprint
pyproject.toml version was "2.0.0" through 10 sprints. pip show finetune-cli
reported the wrong version the entire time. Rule: version bump is item 1 of
the sprint-end checklist. Format: MAJOR.MINOR.PATCH where MINOR = sprint number.

## Pattern: Checklists beat documentation for recurring tasks
Documenting a pattern in lessons.md is not enough if the pattern requires
action at a specific moment (sprint-end). A checklist in CONTRIBUTING.md
that must be ticked off is the only reliable enforcement mechanism.