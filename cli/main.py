"""
CLI entry point for the fine-tuning framework.

Commands:
  train      Fine-tune a model (LoRA or QLoRA)
  evaluate   Evaluate a saved model checkpoint
  benchmark  Run before/after benchmark comparison

Usage::

  python -m finetune_cli.cli train --config config.yaml
  python -m finetune_cli.cli train --model gpt2 --dataset ./data.jsonl
  python -m finetune_cli.cli evaluate --model-path ./output --dataset ./data.jsonl
  python -m finetune_cli.cli benchmark --base gpt2 --finetuned ./output --dataset ./data.jsonl
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from ..core.types import (
    TrainingMethod,
    DatasetSource,
    EvaluationMetric,
)
from ..core.config import ConfigBuilder
from ..core.exceptions import FineTuneError
from ..utils.logging import setup_logger, LogLevel


app = typer.Typer(
    name="finetune-cli",
    help="Production-grade LLM fine-tuning CLI (LoRA / QLoRA)",
    add_completion=False,
)
console = Console()


# ============================================================================
# TRAIN
# ============================================================================


@app.command()
def train(
    # Config file (takes precedence over individual flags)
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML/JSON config file"),
    # Quick flags (used when no config file provided)
    model: str = typer.Option("gpt2", "--model", "-m", help="HuggingFace model id"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Local dataset path"),
    hf_dataset: Optional[str] = typer.Option(None, "--hf-dataset", help="HuggingFace dataset id"),
    output_dir: Path = typer.Option(Path("./output"), "--output", "-o", help="Output directory"),
    method: TrainingMethod = typer.Option(TrainingMethod.LORA, "--method", help="Training method"),
    # LoRA hyper-params
    lora_r: int = typer.Option(8, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(32, "--lora-alpha", help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.1, "--lora-dropout", help="LoRA dropout"),
    # Training hyper-params
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Per-device batch size"),
    lr: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    max_length: int = typer.Option(512, "--max-length", help="Max token length"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Limit dataset size"),
    # Misc
    quantize_4bit: bool = typer.Option(False, "--4bit", help="Load model in 4-bit (QLoRA)"),
    fp16: bool = typer.Option(False, "--fp16", help="Mixed precision FP16"),
    log_level: str = typer.Option("info", "--log-level", help="Logging verbosity"),
):
    """Fine-tune a model using LoRA or QLoRA."""
    logger = setup_logger("cli.train", level=LogLevel(log_level))

    try:
        # --- Build config ---
        if config is not None:
            from ..core.config import PipelineConfig
            pipeline_config = (
                PipelineConfig.from_yaml(config)
                if config.suffix in (".yml", ".yaml")
                else PipelineConfig.from_json(config)
            )
        else:
            # Validate dataset source
            if dataset is None and hf_dataset is None:
                console.print("[red]Error:[/red] Provide --dataset or --hf-dataset")
                raise typer.Exit(code=1)

            ds_path = str(dataset) if dataset else str(hf_dataset)
            ds_source = DatasetSource.LOCAL_FILE if dataset else DatasetSource.HUGGINGFACE_HUB

            builder = (
                ConfigBuilder()
                .with_model(model, load_in_4bit=quantize_4bit)
                .with_dataset(ds_path, source=ds_source, max_samples=max_samples)
                .with_tokenization(max_length=max_length)
                .with_training(
                    method=method,
                    output_dir=str(output_dir),
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    fp16=fp16,
                )
                .with_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            )
            pipeline_config = builder.build()

        console.print(Panel(f"[bold green]Training[/bold green] {pipeline_config.model.name}"))

        # --- Load model ---
        from ..models.loader import load_model_and_tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(pipeline_config.model.to_config())

        # --- Prepare data ---
        from ..data import prepare_dataset
        dataset_obj = prepare_dataset(
            dataset_config=pipeline_config.dataset.to_config(),
            tokenization_config=pipeline_config.tokenization.to_config(),
            tokenizer=tokenizer,
            split_for_validation=True,
        )

        # --- Train ---
        from ..trainers import TrainerFactory
        result = TrainerFactory.train(
            model=model_obj,
            tokenizer=tokenizer,
            dataset=dataset_obj,
            training_config=pipeline_config.training.to_config(),
            lora_config=pipeline_config.lora.to_config() if pipeline_config.lora else None,
            model_config=pipeline_config.model.to_config(),
        )

        console.print(Panel(
            f"[bold green]✓ Training complete[/bold green]\n"
            f"Model saved to: {result.output_dir}\n"
            f"Train loss: {result.train_loss:.4f}\n"
            f"Steps: {result.steps_completed}"
        ))

    except FineTuneError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


# ============================================================================
# EVALUATE
# ============================================================================


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Path to fine-tuned model/adapter"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d"),
    hf_dataset: Optional[str] = typer.Option(None, "--hf-dataset"),
    base_model: str = typer.Option("gpt2", "--base-model", help="Base model id for adapter merging"),
    metrics: str = typer.Option("rougeL,bleu", "--metrics", help="Comma-separated metric names"),
    batch_size: int = typer.Option(4, "--batch-size"),
    num_samples: int = typer.Option(100, "--num-samples"),
    max_gen_length: int = typer.Option(100, "--max-gen-length"),
):
    """Evaluate a saved model checkpoint and print metric scores."""
    from ..models.loader import load_model_and_tokenizer
    from ..core.types import ModelConfig, DeviceType, EvaluationConfig
    from ..evaluation import BenchmarkRunner

    if dataset is None and hf_dataset is None:
        console.print("[red]Error:[/red] Provide --dataset or --hf-dataset")
        raise typer.Exit(code=1)

    # Parse metrics
    metric_enums = []
    for m in metrics.split(","):
        m = m.strip()
        try:
            metric_enums.append(EvaluationMetric(m))
        except ValueError:
            console.print(f"[yellow]Warning:[/yellow] Unknown metric '{m}', skipping")

    if not metric_enums:
        console.print("[red]Error:[/red] No valid metrics specified")
        raise typer.Exit(code=1)

    console.print(Panel(f"[bold blue]Evaluating[/bold blue] {model_path}"))

    # Load fine-tuned model
    model_config = ModelConfig(name=str(model_path), device=DeviceType.AUTO)
    model_obj, tokenizer = load_model_and_tokenizer(model_config)

    # Load evaluation dataset
    from ..data import quick_load
    ds_path = str(dataset) if dataset else str(hf_dataset)
    ds_source = "local" if dataset else "huggingface"
    eval_dataset = quick_load(ds_path, tokenizer, source=ds_source, max_samples=num_samples)

    eval_config = EvaluationConfig(
        metrics=metric_enums,
        batch_size=batch_size,
        num_samples=num_samples,
        generation_max_length=max_gen_length,
        generation_do_sample=True,
    )

    runner = BenchmarkRunner(eval_config, tokenizer)
    result = runner.evaluate(model_obj, eval_dataset, label="fine-tuned")

    console.print("\n[bold]Results:[/bold]")
    for metric, score in result.scores.items():
        console.print(f"  {metric:<20} {score:.4f}")


# ============================================================================
# BENCHMARK
# ============================================================================


@app.command()
def benchmark(
    base: str = typer.Argument(..., help="Base model id (e.g. gpt2)"),
    finetuned: Path = typer.Argument(..., help="Path to fine-tuned model/adapter"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d"),
    hf_dataset: Optional[str] = typer.Option(None, "--hf-dataset"),
    metrics: str = typer.Option("rougeL,bleu", "--metrics"),
    batch_size: int = typer.Option(4, "--batch-size"),
    num_samples: int = typer.Option(100, "--num-samples"),
    max_gen_length: int = typer.Option(100, "--max-gen-length"),
):
    """Compare base model vs fine-tuned model on key metrics."""
    from ..models.loader import load_model_and_tokenizer
    from ..core.types import ModelConfig, DeviceType, EvaluationConfig
    from ..evaluation import BenchmarkRunner, BenchmarkReport

    if dataset is None and hf_dataset is None:
        console.print("[red]Error:[/red] Provide --dataset or --hf-dataset")
        raise typer.Exit(code=1)

    metric_enums = [EvaluationMetric(m.strip()) for m in metrics.split(",")]

    console.print(Panel(
        f"[bold cyan]Benchmark[/bold cyan]\n"
        f"Base: {base}\n"
        f"Fine-tuned: {finetuned}"
    ))

    # Load both models
    base_cfg = ModelConfig(name=base, device=DeviceType.AUTO)
    base_model, tokenizer = load_model_and_tokenizer(base_cfg)

    ft_cfg = ModelConfig(name=str(finetuned), device=DeviceType.AUTO)
    ft_model, _ = load_model_and_tokenizer(ft_cfg)

    # Load dataset
    from ..data import quick_load
    ds_path = str(dataset) if dataset else str(hf_dataset)
    ds_source = "local" if dataset else "huggingface"
    eval_dataset = quick_load(ds_path, tokenizer, source=ds_source, max_samples=num_samples)

    eval_config = EvaluationConfig(
        metrics=metric_enums,
        batch_size=batch_size,
        num_samples=num_samples,
        generation_max_length=max_gen_length,
        generation_do_sample=True,
    )

    runner = BenchmarkRunner(eval_config, tokenizer)
    report = runner.run_comparison(base_model, ft_model, eval_dataset)

    console.print("\n" + report.summary())


# ============================================================================
# ENTRY POINT
# ============================================================================


def main():
    app()


if __name__ == "__main__":
    main()