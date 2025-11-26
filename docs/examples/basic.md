# Basic Examples

## Fine-tuning GPT-2 for Sentiment Analysis

**Goal**: Teach GPT-2 to classify text as Positive or Negative.

**Dataset** (`sentiment.json`):
```json
[
  {"input": "I love this movie!", "output": "Positive"},
  {"input": "This is terrible.", "output": "Negative"}
]
```

**Command**:
```bash
finetune-cli finetune run
```

**Settings**:
*   Model: `gpt2`
*   Technique: `LoRA`
*   Rank: `8`
*   Epochs: `3`

## Fine-tuning OPT-125m for Chat

**Goal**: Create a simple chatbot.

**Dataset** (`chat.json`):
```json
[
  {"input": "User: Hello\nAssistant:", "output": "Hi there! How can I help?"},
  {"input": "User: What is AI?\nAssistant:", "output": "AI stands for Artificial Intelligence."}
]
```

**Settings**:
*   Model: `facebook/opt-125m`
*   Technique: `LoRA`
*   Rank: `16`
