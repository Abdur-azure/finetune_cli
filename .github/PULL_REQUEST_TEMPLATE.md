## Summary

<!-- What does this PR do? One sentence. -->

## Type of change

- [ ] Bug fix
- [ ] New feature / trainer
- [ ] Refactor
- [ ] Documentation
- [ ] Tests only
- [ ] CI / tooling

## Related issue

Closes #

## Changes

<!-- Bullet-point list of what changed and why -->

## Test evidence

```
pytest tests/test_<module>.py -v
# paste output here
```

## Checklist

- [ ] Tests added or updated
- [ ] `ruff check xlmtec/` — 0 errors
- [ ] `mypy xlmtec/` — 0 errors
- [ ] CONTEXT.md updated (if adding/changing a module)
- [ ] CHANGELOG.md entry added
- [ ] Heavy optional deps imported **inside functions**, never at module level
- [ ] No `torch` / `transformers` at module top-level in new files