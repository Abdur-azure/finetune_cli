# tasks/ — Context

Session state. Read both files at the start of every working session.

## Files

| File | Purpose |
|------|---------|
| `todo.md` | Sprint-structured task list. Mark items `[x]` as you go. Includes sprint name, acceptance gate, and history of completed sprints. |
| `lessons.md` | Accumulated patterns from corrections. Read before implementing anything — most bugs we've hit before are documented here. |

## Sprints so far

| Sprint | Name | Outcome |
|--------|------|---------|
| 1 | Stable Foundation | All tests green, CI, pyproject.toml, docs |
| 2 | Expand | FullFineTuner, InstructionTrainer, recommend command |
| 3 | First Run | Runnable examples, sample data generator, integration tests |
| 4 | Hardened | CLI lora guard, test_cli_train.py, CHANGELOG, audit_repo |
| 5 | Merge & Release | finetune-cli merge command, test_merge.py (8 tests) |
| 6 | Documented | README, CONTRIBUTING, api.md, docs/index.md all updated |
| 7 | CI Tight | ci.yml paths, pytest-timeout, ruff lint section, absolute imports |
| 8 | DPO | DPOTrainer, validate_dpo_dataset, factory wired, 10 tests |
| 9 | Housekeeping | CONTEXT.md, CLAUDE.md, docs/index.md, api.md synced |
| 10 | DPO Runnable | dpo_sample.jsonl generator, local config, trl optional dep |

## Workflow

1. Start session → read `todo.md` + `lessons.md`
2. Pick next `[ ]` item
3. Implement → verify → mark `[x]`
4. After any correction → update `lessons.md` with the pattern
5. At sprint end → archive sprint in `todo.md`, propose next sprint