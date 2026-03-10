# Batch Inference

Run predictions over a dataset using a fine-tuned model — no custom inference code needed.

---

## Quick start

```bash
# Preview without loading the model
xlmtec predict output/run1 --data data/test.jsonl --dry-run

# Run predictions, save to JSONL
xlmtec predict output/run1 --data data/test.jsonl --output predictions.jsonl

# Read CSV, write CSV
xlmtec predict output/run1 --data data/test.csv --output predictions.csv --format csv

# Tune generation settings
xlmtec predict output/run1 \
  --data data/test.jsonl \
  --batch-size 16 \
  --max-new-tokens 256 \
  --temperature 0.7
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | required | Input file (`.jsonl` or `.csv`) |
| `--output` | `predictions.jsonl` | Output file path |
| `--format` | `jsonl` | Output format: `jsonl` or `csv` |
| `--text-column` | auto-detect | Column containing input text |
| `--batch-size` | `8` | Records processed per batch |
| `--max-new-tokens` | `128` | Max tokens to generate per input |
| `--temperature` | `1.0` | Sampling temperature (`1.0` = greedy) |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--dry-run` | off | Validate input without loading model |

---

## Input formats

**JSONL** — one JSON object per line:

```jsonl
{"text": "The product arrived damaged and the customer service was unhelpful."}
{"text": "Excellent quality, fast shipping, highly recommended!"}
```

**CSV** — standard comma-separated:

```csv
text,label
"Great product, fast delivery",positive
"Broken on arrival",negative
```

---

## Auto-detected text columns

If `--text-column` is not set, xlmtec checks these column names in order:

`text` → `input` → `prompt` → `sentence` → `content` → `question` → `context` → `document` → `instruction`

If none are found, the command exits with an error. Use `--text-column` to specify your column explicitly:

```bash
xlmtec predict output/run1 --data data.jsonl --text-column review_body
```

---

## Output format

Each prediction row adds a `prediction` field to the input record:

**JSONL output:**
```jsonl
{"text": "Great product!", "prediction": "positive"}
{"text": "Broken on arrival.", "prediction": "negative"}
```

**CSV output:**
```csv
text,prediction
"Great product!","positive"
"Broken on arrival.","negative"
```

---

## Dry run

Validates the input file, detects the text column, and prints a summary — without loading the model:

```bash
xlmtec predict output/run1 --data data/test.jsonl --dry-run
```

```
┌─ Batch Inference Plan ────────────────────────────┐
│  Model dir    : output/run1                       │
│  Input        : data/test.jsonl  (450 records)    │
│  Text column  : text  (auto-detected)             │
│  Output       : predictions.jsonl                 │
│  Batch size   : 8  → 57 batches                   │
│  Max tokens   : 128                               │
│  Device       : auto                              │
└───────────────────────────────────────────────────┘
Remove --dry-run to run inference.
```

---

## Generating test data

```bash
python generate_inference_data.py
xlmtec predict output/dummy_model --data data/test.jsonl --dry-run
```