# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Sprint 13: "Test Coverage Complete"

- [x] tests/test_evaluate.py — 6 tests (missing dataset, metric output, unknown metric, num-samples)
- [x] tests/test_benchmark.py — 6 tests (missing dataset, summary output, run_comparison called)
- [x] tests/test_upload.py — 7 tests (missing path/token, private, HF_TOKEN env, merge-adapter)
- [x] tests/test_cli_train.py — full_finetuning test case added
- [x] audit_repo.py — 3 new files registered
- [x] CHANGELOG.md — Sprint 13 entry
- [ ] Run: pytest tests/test_evaluate.py tests/test_benchmark.py tests/test_upload.py -v

---

## Previously Completed

### Sprint 12: "Usage Guide Current"
- [x] docs/usage.md all 6 commands, CLAUDE.md 11-step checklist

### Sprint 11: "Version Sync"
- [x] pyproject.toml 2.8.0, CONTRIBUTING sprint-end checklist

### Sprint 10: "DPO Runnable"
- [x] dpo_sample.jsonl generator, local config, trl optional dep

### Sprint 9: "Housekeeping"
- [x] CONTEXT.md, CLAUDE.md, docs/index.md, api.md synced

### Sprint 8: "DPO"
- [x] DPOTrainer, validate_dpo_dataset, factory, 10 tests

### Sprints 1–7: Foundation, Expand, First Run, Hardened, Merge, Documented, CI Tight

---

## Acceptance Gate
pytest tests/test_evaluate.py tests/test_benchmark.py tests/test_upload.py -v
→ all 19 tests green.