# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 4: "Hardened"

- [x] cli/main.py — guard .with_lora() for PEFT methods only
- [x] trainers/CONTEXT.md — document _maybe_format input_ids fix
- [x] audit_repo.py — add Sprint 2+3 files to REQUIRED_FILES
- [x] CHANGELOG.md — add Sprint 2 + Sprint 3 entries
- [x] tests/test_cli_train.py — wiring tests for all methods via flags
- [ ] Run: pytest tests/test_cli_train.py -v

---

## Previously Completed

### Sprint 3: "First Run"
- [x] examples/generate_sample_data.py, 4 example configs
- [x] Integration tests: instruction tuning + recommend
- [x] docs/usage.md quickstart, CLAUDE.md, CONTEXT.md files

### Sprint 2: "Expand"
- [x] FullFineTuner, InstructionTrainer, recommend command
- [x] 60 unit tests passing

### Sprint 1: "Stable Foundation"
- [x] All unit + integration tests green
- [x] CI, pyproject.toml, conftest.py, data pipeline
- [x] CHANGELOG.md, v1 deprecation, docs rewritten

---

## Acceptance Gate
pytest tests/test_cli_train.py -v — all 6 tests green.