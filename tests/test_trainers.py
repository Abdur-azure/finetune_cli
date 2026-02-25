"""
Unit tests for the trainer system.

HuggingFace Trainer and PEFT are mocked so tests run without GPU
or large model downloads.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
from datasets import Dataset, DatasetDict

from ..core.types import TrainingConfig, LoRAConfig, TrainingMethod, ModelConfig
from ..core.exceptions import MissingConfigError
from ..trainers import TrainingResult, TrainerFactory
from ..trainers.base import BaseTrainer
from ..trainers.lora_trainer import LoRATrainer
from ..trainers.factory import TrainerFactory


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def training_config(tmp_output_dir) -> TrainingConfig:
    return TrainingConfig(
        method=TrainingMethod.LORA,
        output_dir=tmp_output_dir,
        num_epochs=1,
        batch_size=2,
        learning_rate=2e-4,
    )


@pytest.fixture
def lora_config() -> LoRAConfig:
    return LoRAConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(name="gpt2")


@pytest.fixture
def small_dataset() -> Dataset:
    return Dataset.from_dict({
        "input_ids": [[1, 2, 3, 4]] * 8,
        "attention_mask": [[1, 1, 1, 1]] * 8,
        "labels": [[1, 2, 3, 4]] * 8,
    })


@pytest.fixture
def dataset_dict(small_dataset) -> DatasetDict:
    return DatasetDict({"train": small_dataset, "validation": small_dataset})


# ============================================================================
# TRAINER FACTORY TESTS
# ============================================================================


class TestTrainerFactory:
    """TrainerFactory picks the right trainer and validates required configs."""

    def test_create_lora_trainer(self, mock_model, mock_tokenizer, training_config, lora_config):
        trainer = TrainerFactory.create(
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_config=training_config,
            lora_config=lora_config,
        )
        assert isinstance(trainer, LoRATrainer)

    def test_lora_without_lora_config_raises(self, mock_model, mock_tokenizer, training_config):
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=training_config,
                lora_config=None,  # missing!
            )

    def test_unsupported_method_raises(self, mock_model, mock_tokenizer, tmp_output_dir):
        cfg = TrainingConfig(
            method=TrainingMethod.DPO,
            output_dir=tmp_output_dir,
        )
        with pytest.raises(NotImplementedError):
            TrainerFactory.create(mock_model, mock_tokenizer, cfg)

    def test_qlora_requires_model_config(self, mock_model, mock_tokenizer, lora_config, tmp_output_dir):
        cfg = TrainingConfig(method=TrainingMethod.QLORA, output_dir=tmp_output_dir)
        with pytest.raises(MissingConfigError):
            TrainerFactory.create(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=cfg,
                lora_config=lora_config,
                model_config=None,  # missing!
            )


# ============================================================================
# LORA TRAINER TESTS (mocked HF Trainer)
# ============================================================================


class TestLoRATrainerUnit:
    """LoRATrainer unit tests — HF Trainer.train() is mocked."""

    def _mock_train_output(self):
        out = MagicMock()
        out.training_loss = 0.42
        out.global_step = 10
        out.metrics = {"epoch": 1}
        return out

    def _mock_hf_trainer(self):
        hf_trainer = MagicMock()
        hf_trainer.train.return_value = self._mock_train_output()
        hf_trainer.state.log_history = [
            {"loss": 0.5, "step": 5},
            {"eval_loss": 0.45, "step": 10},
        ]
        return hf_trainer

    @patch("finetune_cli.trainers.lora_trainer.get_peft_model")
    @patch("finetune_cli.trainers.base.BaseTrainer._build_hf_trainer")
    def test_train_returns_training_result(
        self,
        mock_build_hf_trainer,
        mock_get_peft,
        mock_model,
        mock_tokenizer,
        training_config,
        lora_config,
        small_dataset,
        tmp_output_dir,
    ):
        mock_get_peft.return_value = mock_model
        mock_build_hf_trainer.return_value = self._mock_hf_trainer()

        trainer = LoRATrainer(mock_model, mock_tokenizer, training_config, lora_config)
        result = trainer.train(small_dataset)

        assert isinstance(result, TrainingResult)
        assert result.train_loss == pytest.approx(0.42)
        assert result.steps_completed == 10
        assert result.eval_loss == pytest.approx(0.45)

    @patch("finetune_cli.trainers.lora_trainer.get_peft_model")
    @patch("finetune_cli.trainers.base.BaseTrainer._build_hf_trainer")
    def test_train_with_dataset_dict(
        self,
        mock_build_hf_trainer,
        mock_get_peft,
        mock_model,
        mock_tokenizer,
        training_config,
        lora_config,
        dataset_dict,
    ):
        mock_get_peft.return_value = mock_model
        mock_build_hf_trainer.return_value = self._mock_hf_trainer()

        trainer = LoRATrainer(mock_model, mock_tokenizer, training_config, lora_config)
        result = trainer.train(dataset_dict)

        assert isinstance(result, TrainingResult)

    @patch("finetune_cli.trainers.lora_trainer.get_peft_model")
    @patch("finetune_cli.trainers.base.BaseTrainer._build_hf_trainer")
    def test_model_saved_to_output_dir(
        self,
        mock_build_hf_trainer,
        mock_get_peft,
        mock_model,
        mock_tokenizer,
        training_config,
        lora_config,
        small_dataset,
        tmp_output_dir,
    ):
        mock_get_peft.return_value = mock_model
        hf_trainer = self._mock_hf_trainer()
        mock_build_hf_trainer.return_value = hf_trainer

        trainer = LoRATrainer(mock_model, mock_tokenizer, training_config, lora_config)
        result = trainer.train(small_dataset)

        hf_trainer.save_model.assert_called_once_with(str(tmp_output_dir))
        assert result.output_dir == tmp_output_dir