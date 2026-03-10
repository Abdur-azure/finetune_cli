# xlmtec/notifications — Context

Training event notifications powering the `--notify` flag on `xlmtec train`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package docstring |
| `base.py` | `Notifier` ABC, `NotifyEvent` enum, `NotifyPayload` dataclass |
| `slack.py` | Slack webhook notifier (stdlib urllib — no extra deps) |
| `email.py` | SMTP email notifier (stdlib smtplib — no extra deps) |
| `desktop.py` | OS desktop notification via `plyer` (graceful fallback to console) |
| `dispatcher.py` | `NotificationDispatcher` — builds notifiers by name, routes events |
| `CONTEXT.md` | This file |

## Channel names

| Channel | Class | Env vars required |
|---------|-------|-------------------|
| `slack` | `SlackNotifier` | `XLMTEC_SLACK_WEBHOOK` |
| `email` | `EmailNotifier` | `XLMTEC_EMAIL_TO`, `XLMTEC_SMTP_HOST`, … |
| `desktop` | `DesktopNotifier` | none (`plyer` optional — falls back to print) |

## Rules

- **`Notifier.send()` never raises** — all backends catch exceptions and return `False`.
- **All notifier constructors read from env vars** — no secrets in config files.
- **`desktop` has zero hard deps** — if `plyer` is not installed it prints to console.
- **`slack` and `email` use stdlib only** — no third-party HTTP clients.
- **Tests mock at the network layer** — never make real network calls.
  Patch `urllib.request.urlopen` for Slack, `smtplib.SMTP` for email.

## Extension pattern

To add a new channel (e.g. Teams, PagerDuty):
1. Create `xlmtec/notifications/teams.py` subclassing `Notifier`
2. Set `name = "teams"` and implement `send()`
3. Register in `dispatcher._NOTIFIERS` inside `_register()`
4. Add tests to `tests/test_notifications.py`
5. Add env vars to CONTEXT.md table above

## CLI wiring

`--notify` accepts a comma-separated list:

```bash
xlmtec train --config config.yaml --notify slack
xlmtec train --config config.yaml --notify slack,desktop
xlmtec train --config config.yaml --notify email
```