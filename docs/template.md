# Config Templates

Start from a proven configuration instead of writing YAML from scratch.

---

## Quick start

```bash
# See all available templates
xlmtec template list

# Preview a template
xlmtec template show sentiment

# Generate a config file from a template
xlmtec template use sentiment --output config.yaml

# Override the base model
xlmtec template use summarisation --model facebook/bart-large --output config.yaml

# Override model, data path, and epochs in one go
xlmtec template use classification \
  --model distilbert-base-uncased \
  --data data/train.jsonl \
  --epochs 5 \
  --output config.yaml
```

---

## Built-in templates

| Name | Method | Base model | Best for |
|------|--------|-----------|----------|
| `sentiment` | LoRA | distilbert-base-uncased | Binary / multi-class sentiment |
| `classification` | LoRA | bert-base-uncased | General text classification |
| `qa` | LoRA | deepset/roberta-base-squad2 | Extractive question answering |
| `summarisation` | LoRA | facebook/bart-base | Abstractive summarisation |
| `code` | QLoRA | Salesforce/codegen-350M-mono | Code completion / generation |
| `chat` | Instruction | microsoft/DialoGPT-small | Conversational / chat tasks |
| `dpo` | DPO | gpt2 | Preference learning from pairs |

---

## Override flags

All three flags are optional — omit any to keep the template default.

| Flag | What it overrides |
|------|------------------|
| `--model MODEL_ID` | `model.name` in the generated YAML |
| `--data PATH` | `dataset.path` in the generated YAML |
| `--epochs N` | `training.num_epochs` in the generated YAML |
| `--output PATH` | Where to write the YAML file (required for `use`) |

---

## Adding a custom template

Use the plugin system to register your own template YAML:

```bash
xlmtec plugin add-template my_task templates/my_task.yaml
xlmtec template list   # my_task appears alongside built-ins
xlmtec template use my_task --output config.yaml
```

The YAML schema mirrors standard xlmtec config files:

```yaml
method: lora
description: "My custom classification template"
task: text-classification
base_model: bert-base-uncased
model:
  name: bert-base-uncased
dataset:
  source: local_file
  path: data/train.jsonl
  text_column: text
  label_column: label
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: [query, value]
training:
  output_dir: output/my_task
  num_epochs: 3
  batch_size: 16
  learning_rate: 0.0002
```

See [Plugin System](plugin.md) for full details on registering custom templates.

---

## Workflow

```
xlmtec template list              # browse templates
xlmtec template show <name>       # inspect before using
xlmtec template use <name> -o config.yaml   # generate
xlmtec train --config config.yaml --dry-run # preview
xlmtec train --config config.yaml           # train
```