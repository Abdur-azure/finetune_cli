# Tasks — Next Steps Plan

## Status Legend
- [ ] = Todo
- [x] = Complete
- [~] = Blocked / done via alternative

---

## Current Sprint

- [~] Run test suite — sandbox has no network/packages; static AST verification done. Run locally: `pytest finetune_cli/tests/ -v`
- [~] Fix import mismatches — resolved via cross-reference analysis before writing code
- [x] Add `.github/workflows/ci.yml` — pytest on every push/PR
- [x] Add `upload` CLI subcommand — push fine-tuned model to HuggingFace Hub
- [x] Add example YAML configs — `examples/configs/lora_gpt2.yaml`, `qlora_llama.yaml`
- [x] Write end-to-end integration test — `tests/test_integration.py`

---

## Previously Completed (Phases 3-5)

- [x] trainers/ — base, lora_trainer, qlora_trainer, factory
- [x] evaluation/ — metrics, benchmarker
- [x] cli/main.py — train, evaluate, benchmark subcommands
- [x] tests/ — test_config, test_trainers, test_evaluation, conftest

---

## Review

### What was built this sprint
- CI workflow (.github/workflows/ci.yml)
- upload subcommand in CLI
- Example YAML configs (gpt2 LoRA, llama QLoRA)
- End-to-end integration test covering config > data > train > save

### Lessons captured — see tasks/lessons.md