# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 3.x (latest) | ✅ Active |
| 2.x | ❌ End of life |
| < 2.0 | ❌ End of life |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately via GitHub's
[private vulnerability reporting](https://github.com/Abdur-azure/xlmtec/security/advisories/new)
(Settings → Security → Advisories → New draft advisory).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within **72 hours** acknowledging receipt.
If the issue is confirmed, a patch will be released as soon as possible
and you will be credited in the release notes (unless you prefer anonymity).

## Scope

This policy covers:
- The `xlmtec` Python package and CLI
- The xlmtec documentation site (https://Abdur-azure.github.io/xlmtec)

Out of scope:
- Third-party dependencies (report those to the respective projects)
- Model weights or datasets loaded by users

## Safe use notes

- xlmtec loads YAML config files — never load untrusted YAML from unknown sources
- xlmtec can download models from HuggingFace Hub — verify model sources
- API keys for AI providers (Claude, Gemini, GPT) are read from environment
  variables; never hardcode them in config files committed to source control