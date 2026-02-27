# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 10: "DPO Runnable"

- [x] generate_sample_data.py — generate_dpo_samples() → data/dpo_sample.jsonl (200 rows)
- [x] examples/configs/dpo.yaml — switched to local_file source
- [x] pyproject.toml — trl>=0.7.0 in [project.optional-dependencies] dpo group
- [x] docs/configuration.md — DPO section added
- [x] CHANGELOG.md + lessons.md updated
- [x] Verified: python examples/generate_sample_data.py → 200 rows, correct columns

---

## Previously Completed

### Sprint 9: "Housekeeping"
- [x] CONTEXT.md, CLAUDE.md, docs/index.md, trainers/CONTEXT.md, docs/api.md synced

### Sprint 8: "DPO"
- [x] DPOTrainer, validate_dpo_dataset, factory, tests, example config

### Sprint 7: "CI Tight"
- [x] ci.yml paths, pytest-timeout, ruff lint, absolute imports

### Sprint 6: "Documented"
- [x] README, CONTRIBUTING, docs/api.md, CLAUDE.md, cli/CONTEXT.md

### Sprint 5: "Merge & Release"
- [x] finetune-cli merge command + test_merge.py

### Sprint 4: "Hardened"
- [x] CLI lora guard, test_cli_train.py, CHANGELOG, audit_repo

### Sprint 3: "First Run"
- [x] Examples, sample data, integration tests, CLAUDE.md, CONTEXT.md files

### Sprint 2: "Expand"
- [x] FullFineTuner, InstructionTrainer, recommend, 60 tests

### Sprint 1: "Stable Foundation"
- [x] All tests green, CI, pyproject.toml, docs

---

## Acceptance Gate
python examples/generate_sample_data.py
→ data/dpo_sample.jsonl (200 rows, prompt/chosen/rejected) created with no network.