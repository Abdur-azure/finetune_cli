# Evaluation Dashboard

Compare training runs side by side and identify the best-performing model.

---

## Quick start

```bash
# Compare two runs
xlmtec dashboard compare output/run1 output/run2

# Compare three runs
xlmtec dashboard compare output/run1 output/run2 output/run3

# Export comparison to JSON
xlmtec dashboard compare output/run1 output/run2 --export results.json

# Inspect a single run
xlmtec dashboard show output/run1
```

---

## Compare output

```
╭─── Run Comparison ──────────────────────────────────────────╮
│ Metric              │ run1          │ run2 ★        │        │
│─────────────────────│───────────────│───────────────│        │
│ Total steps         │ 1000          │ 1500          │        │
│ Total epochs        │ 3.0           │ 5.0           │        │
│ Best metric         │ 0.84          │ 0.91          │        │
│ Best eval loss      │ 0.31          │ 0.24          │        │
│ Final train loss    │ 0.18          │ 0.14          │        │
│ Duration            │ 12m 04s       │ 19m 32s       │        │
╰─────────────────────────────────────────────────────────────╯

  ★ Winner: run2  (best_metric: 0.91)
```

The ★ winner column is selected automatically based on the priority below.

---

## Winner selection priority

1. **`best_metric`** — highest value wins (accuracy, F1, BLEU…)
2. **`best_eval_loss`** — lowest value wins (if no best_metric)
3. **`final_train_loss`** — lowest value wins (if no eval loss)
4. **Most steps completed** — fallback if all metrics are missing

---

## Config diff

When comparing runs the dashboard also prints a config diff — only keys that differ between runs are shown:

```
Config differences:
  learning_rate   run1=0.0002   run2=0.0001
  num_epochs      run1=3        run2=5
```

This makes it easy to see which hyperparameters changed between experiments.

---

## Export flag

```bash
xlmtec dashboard compare output/run1 output/run2 --export results.json
```

Writes a JSON file with all metrics and the winner, suitable for further analysis or CI reporting.

---

## Show command

```bash
xlmtec dashboard show output/run1
```

Prints a summary panel for a single run: model name, method, total steps, all eval metrics, and the config used.

---

## What runs need

Each run directory must contain at least one of:

- `trainer_state.json` — written automatically by HuggingFace Trainer
- `config.yaml` — written by xlmtec at the start of training

Missing files are handled gracefully — the run still appears in the comparison with `None` for unavailable fields.

---

## Generating test runs

```bash
python generate_dummy_runs.py
xlmtec dashboard compare output/dummy_run1 output/dummy_run2 output/dummy_run3
```