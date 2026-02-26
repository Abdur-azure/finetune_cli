"""
End-to-end integration test.

Loads GPT-2 (smallest, ~500MB), trains for 1 step on 10 synthetic samples,
and asserts the output directory contains a saved model.

Run with:
    pytest finetune_cli/tests/test_integration.py -v -s

Requirements: transformers, peft, datasets, torch (CPU is sufficient)
Skipped automatically when FINETUNE_CI=1 and torch is unavailable.
"""

import os
import json
import pytest
from pathlib import Path


# Skip entire module if heavy deps are missing
pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")
pytest.importorskip("peft", reason="peft not installed")
pytest.importorskip("datasets", reason="datasets not installed")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def tiny_jsonl(tmp_path_factory) -> Path:
    """10-sample JSONL file for integration tests."""
    d = tmp_path_factory.mktemp("data")
    f = d / "train.jsonl"
    samples = [{"text": f"Sample training sentence number {i}."} for i in range(10)]
    with open(f, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    return f


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("output")


# ============================================================================
# TESTS
# ============================================================================


class TestEndToEnd:
    """Full pipeline: config → model load → data → train → save."""

    def test_config_builds_from_yaml(self, tiny_jsonl, output_dir, tmp_path_factory):
        """ConfigBuilder produces a valid PipelineConfig from inline values."""
        from finetune_cli.core.config import ConfigBuilder
        from finetune_cli.core.types import TrainingMethod, DatasetSource

        config = (
            ConfigBuilder()
            .with_model("gpt2")
            .with_dataset(str(tiny_jsonl), source=DatasetSource.LOCAL_FILE, max_samples=10)
            .with_tokenization(max_length=64)
            .with_training(
                method=TrainingMethod.LORA,
                output_dir=str(output_dir),
                num_epochs=1,
                batch_size=2,
                gradient_accumulation_steps=1,
                logging_steps=1,
            )
            .with_lora(r=4, lora_alpha=8, target_modules=["c_attn"])
            .build()
        )
        assert config.model.name == "gpt2"
        assert config.lora.r == 4

    def test_full_lora_train_saves_model(self, tiny_jsonl, output_dir):
        """
        Load GPT-2 + LoRA, train 1 step on 10 samples, assert model saved.
        This is the single most important integration check.
        """
        import torch
        from finetune_cli.core.config import ConfigBuilder
        from finetune_cli.core.types import TrainingMethod, DatasetSource
        from finetune_cli.models.loader import load_model_and_tokenizer
        from finetune_cli.trainers import TrainerFactory
        from datasets import Dataset

        save_dir = output_dir / "lora_gpt2"
        save_dir.mkdir(exist_ok=True)

        # Build config
        config = (
            ConfigBuilder()
            .with_model("gpt2")
            .with_dataset(str(tiny_jsonl), source=DatasetSource.LOCAL_FILE, max_samples=10)
            .with_tokenization(max_length=64)
            .with_training(
                method=TrainingMethod.LORA,
                output_dir=str(save_dir),
                num_epochs=1,
                batch_size=2,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_strategy="no",
                evaluation_strategy="no",
            )
            .with_lora(r=4, lora_alpha=8, target_modules=["c_attn"])
            .build()
        )

        # Load model
        model, tokenizer = load_model_and_tokenizer(config.model.to_config())

        # Build tiny tokenized dataset manually (avoids full data pipeline dep)
        enc = tokenizer(
            ["Sample training sentence number 0."] * 10,
            max_length=64,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        dataset = Dataset.from_dict({
            "input_ids": enc["input_ids"].tolist(),
            "attention_mask": enc["attention_mask"].tolist(),
        })

        # Train
        result = TrainerFactory.train(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_config=config.training.to_config(),
            lora_config=config.lora.to_config(),
        )

        # Verify
        assert result.output_dir == save_dir
        assert result.steps_completed >= 1
        assert result.train_loss >= 0.0

        # Saved model must contain config.json (HF standard)
        assert (save_dir / "adapter_config.json").exists(), \
            f"config.json not found in {save_dir}. Contents: {list(save_dir.iterdir())}"

    def test_rouge_metric_runs_on_strings(self):
        """ROUGE metric computes without needing a live model."""
        from finetune_cli.evaluation.metrics import RougeMetric
        from finetune_cli.core.types import EvaluationMetric

        metric = RougeMetric(EvaluationMetric.ROUGE_L)
        score = metric.compute(
            predictions=["the quick brown fox"],
            references=["the quick brown fox"],
        )
        assert score == pytest.approx(1.0)

    def test_benchmark_report_summary_format(self):
        """BenchmarkReport.summary() produces a non-empty string."""
        from finetune_cli.evaluation.benchmarker import BenchmarkReport, EvaluationResult

        report = BenchmarkReport(
            baseline=EvaluationResult("base", {"rougeL": 0.30}, 10, 1.0),
            finetuned=EvaluationResult("ft", {"rougeL": 0.45}, 10, 1.0),
        )
        summary = report.summary()
        assert "rougeL" in summary
        assert "0.3000" in summary
        assert "0.4500" in summary
        assert "▲" in summary  # improvement arrow


    def test_instruction_tuning_saves_adapter(self, tiny_jsonl, output_dir):
        """
        InstructionTrainer formats and trains on alpaca-style data, saves adapter.
        """
        import json
        from pathlib import Path
        from finetune_cli.core.config import ConfigBuilder
        from finetune_cli.core.types import TrainingMethod, DatasetSource
        from finetune_cli.models.loader import load_model_and_tokenizer
        from finetune_cli.trainers import TrainerFactory, InstructionTrainer
        from datasets import Dataset

        # Build alpaca-style instruction dataset
        samples = [
            {
                "instruction": f"Explain concept {i}.",
                "input": "",
                "response": f"Concept {i} is important in machine learning.",
            }
            for i in range(10)
        ]
        instr_file = output_dir / "instructions_test.jsonl"
        with open(instr_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        save_dir = output_dir / "instruction_gpt2"
        save_dir.mkdir(exist_ok=True)

        config = (
            ConfigBuilder()
            .with_model("gpt2")
            .with_dataset(str(instr_file), source=DatasetSource.LOCAL_FILE, max_samples=10)
            .with_tokenization(max_length=64)
            .with_training(
                method=TrainingMethod.INSTRUCTION_TUNING,
                output_dir=str(save_dir),
                num_epochs=1,
                batch_size=2,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_strategy="no",
                evaluation_strategy="no",
            )
            .with_lora(r=4, lora_alpha=8, target_modules=["c_attn"])
            .build()
        )

        model, tokenizer = load_model_and_tokenizer(config.model.to_config())

        # Pre-format to text so InstructionTrainer detects and reformats
        enc = tokenizer(
            [f"### Instruction:\nExplain concept {i}.\n\n### Response:\nConcept {i} is important." for i in range(10)],
            max_length=64, truncation=True, padding="max_length", return_tensors="pt",
        )
        dataset = Dataset.from_dict({
            "input_ids": enc["input_ids"].tolist(),
            "attention_mask": enc["attention_mask"].tolist(),
        })

        trainer = TrainerFactory.create(
            model=model,
            tokenizer=tokenizer,
            training_config=config.training.to_config(),
            lora_config=config.lora.to_config(),
        )
        assert isinstance(trainer, InstructionTrainer)

        result = trainer.train(dataset)
        assert result.steps_completed >= 1
        assert (save_dir / "adapter_config.json").exists(), \
            f"adapter_config.json not found. Contents: {list(save_dir.iterdir())}"

    def test_recommend_produces_runnable_config(self, tmp_path_factory):
        """
        recommend command writes a YAML that loads cleanly as PipelineConfig.
        """
        import yaml
        from unittest.mock import patch, MagicMock
        from typer.testing import CliRunner
        from finetune_cli.cli.main import app

        runner = CliRunner()
        out_file = tmp_path_factory.mktemp("recommend") / "config.yaml"

        mock_cfg = MagicMock()
        mock_cfg.hidden_size = 768
        mock_cfg.num_hidden_layers = 12

        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
            result = runner.invoke(app, ["recommend", "gpt2", "--output", str(out_file)])

        assert result.exit_code == 0, f"recommend failed: {result.output}"
        assert out_file.exists()

        parsed = yaml.safe_load(out_file.read_text())
        assert "model" in parsed
        assert "training" in parsed
        assert "tokenization" in parsed

        # Must load as PipelineConfig with patched path validation
        from finetune_cli.core.config import PipelineConfig
        with patch("pathlib.Path.exists", return_value=True):
            cfg = PipelineConfig.from_yaml(out_file)
        assert cfg.model.name == "gpt2"
        assert cfg.training is not None