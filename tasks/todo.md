# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete
- [~] = Blocked/alternative used

---

## Current Sprint — Close the Loop

- [ ] Run integration test — pytest tests/test_integration.py -v -s
- [x] Deprecate v1 finetune_cli.py — deprecation warning redirecting to new CLI
- [x] Verify data pipeline wiring — trace config → data → trainer in CLI
- [x] Add CHANGELOG.md — v1→v2 migration, what changed, what's new

---

## Previously Completed

- [x] Phases 3-5 — trainers/, evaluation/, cli/main.py
- [x] tests/ — unit + integration tests (unit: all green)
- [x] CI workflow, upload subcommand, example YAML configs
- [x] Foundation stabilised — pyproject.toml, conftest.py, all __init__.py files

---

## Acceptance Gate
Integration test green → repo is considered end-to-end verified.