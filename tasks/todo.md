# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 3: "First Run"

- [x] examples/generate_sample_data.py
- [x] examples/configs/lora_gpt2.yaml — updated, points to generated data
- [x] examples/configs/qlora_llama.yaml — verified v2 format
- [x] examples/configs/instruction_tuning.yaml — new
- [x] examples/configs/full_finetuning.yaml — new
- [x] tests/test_integration.py — instruction tuning + recommend tests added
- [x] docs/usage.md — 5-minute quickstart section added
- [ ] Run new integration tests locally to close sprint

---

## Previously Completed

### Sprint 2: "Expand"
- [x] trainers/full_trainer.py, instruction_trainer.py
- [x] TrainerFactory wired, recommend command
- [x] 60 unit tests passing

### Sprint 1: "Stable Foundation"
- [x] All unit + integration tests green
- [x] CI, pyproject.toml, conftest.py, data pipeline
- [x] CHANGELOG.md, v1 deprecation, docs rewritten

---

## Acceptance Gate
pytest tests/test_integration.py -v -s — all 5 integration tests green.