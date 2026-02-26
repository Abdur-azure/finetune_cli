import pytest
import torch
from unittest.mock import MagicMock
from datasets import Dataset
from pathlib import Path

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.parameters.return_value = iter([torch.randn(10, 10)])
    model.named_modules.return_value = []
    return model

@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = "[PAD]"
    tok.eos_token = "</s>"
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    return tok

@pytest.fixture
def tiny_dataset():
    texts = ["The quick brown fox jumps over the lazy dog."] * 10
    return Dataset.from_dict({"text": texts})

@pytest.fixture
def tmp_output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d