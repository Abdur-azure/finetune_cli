# Plugin System

Extend xlmtec with custom templates and AI providers — no source code changes needed.

---

## Quick start

```bash
# Register a custom config template
xlmtec plugin add-template my_task templates/my_task.yaml

# Register a custom AI provider
xlmtec plugin add-provider my_llm providers/my_llm.py --class MyLLMIntegration

# List all registered plugins
xlmtec plugin list

# Remove a plugin
xlmtec plugin remove my_task
```

---

## Custom templates

A template is a standard xlmtec YAML config file. Once registered, it appears in `xlmtec template list` alongside built-ins.

**1. Create a YAML file:**

```yaml
# templates/legal_ner.yaml
method: lora
description: "NER fine-tuning for legal documents"
task: token-classification
base_model: bert-base-uncased
model:
  name: bert-base-uncased
dataset:
  source: local_file
  path: data/legal_ner.jsonl
  text_column: text
  label_column: labels
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: [query, value]
training:
  output_dir: output/legal_ner
  num_epochs: 5
  batch_size: 16
  learning_rate: 0.0002
```

**2. Register it:**

```bash
xlmtec plugin add-template legal_ner templates/legal_ner.yaml
```

**3. Use it like any built-in:**

```bash
xlmtec template list          # legal_ner appears here
xlmtec template show legal_ner
xlmtec template use legal_ner --output config.yaml
```

---

## Custom AI providers

A provider is a Python class that subclasses `xlmtec.integrations.base.AIIntegration`. Once registered, it can be used with `xlmtec ai-suggest --provider my_llm`.

**1. Create the provider class:**

```python
# providers/my_llm.py
from xlmtec.integrations.base import AIIntegration, SuggestResult

class MyLLMIntegration(AIIntegration):
    PROVIDER_NAME = "my_llm"
    ENV_KEY = "MY_LLM_API_KEY"
    DEFAULT_MODEL = "my-model-v1"

    def suggest(self, task: str) -> SuggestResult:
        self._require_api_key()
        # call your LLM API here …
        return SuggestResult(
            method="lora",
            explanation="LoRA is efficient for this task.",
            yaml_config="model:\n  name: gpt2\n",
            command="xlmtec train --config config.yaml",
        )
```

**2. Register it:**

```bash
xlmtec plugin add-provider my_llm providers/my_llm.py --class MyLLMIntegration
```

**3. Use it:**

```bash
export MY_LLM_API_KEY=...
xlmtec ai-suggest "fine-tune for legal NER" --provider my_llm
```

---

## Plugin storage

Plugins are stored in `~/.xlmtec/plugins.json`. The file is created automatically on first registration.

```json
{
  "templates": {
    "legal_ner": {
      "name": "legal_ner",
      "source": "/abs/path/templates/legal_ner.yaml",
      "registered_at": "2026-03-09T12:00:00+00:00"
    }
  },
  "providers": {}
}
```

!!! note "Source paths are absolute"
    xlmtec stores the absolute path to your YAML/Python file. If you move the file, re-register it with `xlmtec plugin add-template`.

---

## Reserved names

These names are reserved and cannot be used for plugins:

**Templates:** `sentiment`, `classification`, `qa`, `summarisation`, `code`, `chat`, `dpo`

**Providers:** `claude`, `gemini`, `codex`

---

## Listing plugins

```bash
xlmtec plugin list
```

```
╭── Custom Templates ──────────────────────────────────────────╮
│ Name         │ Source                          │ Registered  │
│ legal_ner    │ /home/user/templates/legal_ner… │ 2026-03-09  │
╰──────────────────────────────────────────────────────────────╯

╭── Custom Providers ──────────────────────────────────────────╮
│  no custom providers                                         │
╰──────────────────────────────────────────────────────────────╯
```

---

## Removing a plugin

```bash
xlmtec plugin remove legal_ner
```

This unregisters the plugin. The original YAML/Python file is not deleted.