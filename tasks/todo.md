# Tasks

## Status Legend
- [ ] = Todo
- [x] = Complete

---

## Current Sprint — Sync Docs to v2

- [x] Rewrite docs/index.md
- [x] Rewrite docs/usage.md
- [x] Rewrite docs/api.md
- [x] Rewrite docs/configuration.md
- [ ] Verify mkdocs build passes locally — `mkdocs build --strict`

---

## Previously Completed

- [x] Phases 3-5 — trainers/, evaluation/, cli/main.py
- [x] tests/ — all 35+ unit tests + integration test green
- [x] CI, upload subcommand, example YAML configs
- [x] Foundation — pyproject.toml, conftest.py, data pipeline
- [x] CHANGELOG.md, v1 deprecation shim

---

## Acceptance Gate
`mkdocs build --strict` passes with zero warnings.