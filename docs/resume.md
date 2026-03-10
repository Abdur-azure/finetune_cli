# Checkpoint Resume

Resume a fine-tuning run from a saved checkpoint — no data re-processing, no starting over.

---

## Quick start

```bash
# See what checkpoints exist and what would happen
xlmtec resume output/run1 --dry-run

# Resume from the latest checkpoint
xlmtec resume output/run1

# Resume from a specific checkpoint
xlmtec resume output/run1 --checkpoint checkpoint-500

# Resume and train for more epochs
xlmtec resume output/run1 --epochs 5
```

---

## How it works

`xlmtec resume` reads the `checkpoint-N` directories written by the HuggingFace Trainer inside your output directory. It picks up training state (optimizer, scheduler, step count) from `trainer_state.json` and continues from exactly where training stopped.

No config file is needed — the original config is read from `config.yaml` inside the run directory.

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | latest | Name of checkpoint to resume from (e.g. `checkpoint-500`) |
| `--epochs` | from config | Override total epochs for the continued run |
| `--dry-run` | off | Validate checkpoint and show plan — do not train |

---

## Dry run output

```
┌─ Checkpoint Resume Plan ──────────────────────────┐
│  Run dir     : output/run1                        │
│  Checkpoint  : checkpoint-500  (step 500)         │
│  Epochs done : 2 / 5                              │
│  Epochs left : 3                                  │
│  Config      : lora · gpt2 · lr=2e-4              │
└───────────────────────────────────────────────────┘
Remove --dry-run to start training.
```

---

## Checkpoint discovery

`xlmtec resume` scans for any directory matching `checkpoint-{N}` inside the run dir, sorted by step number. `--checkpoint latest` (the default) picks the highest step.

```
output/run1/
  checkpoint-250/
  checkpoint-500/    ← latest (default)
  config.yaml
  trainer_state.json
```

---

## Common scenarios

**Training crashed mid-run**

```bash
xlmtec resume output/run1 --dry-run   # confirm latest checkpoint found
xlmtec resume output/run1             # resume from it
```

**Want to train for more epochs than originally set**

```bash
xlmtec resume output/run1 --epochs 10
```

**Resume from an earlier checkpoint (e.g. before overfitting started)**

```bash
xlmtec resume output/run1 --checkpoint checkpoint-250
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No checkpoints found` | Ensure `save_steps` was set in training config |
| `checkpoint-N not found` | Run `xlmtec resume output/run1 --dry-run` to list available checkpoints |
| `config.yaml missing` | Resume requires the original run directory to be intact |