# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete
- [~] = Blocked/alternative used

---

## Current Sprint — Stabilise Foundation

- [x] Audit missing files — all sandbox files catalogued
- [x] Add root `conftest.py` — fixes pytest sys.path on Windows permanently
- [x] Replace `setup.py` with `pyproject.toml` — eliminates Windows encoding bug
- [x] Deliver all missing core files for local repo — exceptions.py, types.py,
        config.py, utils/logging.py, models/loader.py, all __init__.py files
- [ ] Verify tests pass locally — `pytest tests/ -v` all green (user action)

---

## Previously Completed

- [x] Phases 3-5 — trainers/, evaluation/, cli/main.py
- [x] tests/ — unit + integration tests
- [x] CI workflow, upload subcommand, example YAML configs

---

## Acceptance Gate
No new features until `pytest tests/ -v` is fully green locally.