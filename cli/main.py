"""
Main CLI application for LLM fine-tuning framework.

Provides user-friendly command-line interface for all operations:
- train: Train models with various methods
- evaluate: Evaluate trained models
- benchmark: Compare models
- config: Generate configuration files
- list: List available options
"""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from ..utils.logging import setup_logger, LogLevel


# Create Typer app
app = typer.Typer(
    name="finetune-cli",
    help="🤖 LLM Fine-Tuning Framework - Production-grade fine-tuning made easy",
    add_completion=False,
    rich_markup_mode="rich"
)

# Rich console for pretty output
console = Console()


# ============================================================================
# VERSION COMMAND
# ============================================================================


@app.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold cyan]LLM Fine-Tuning Framework[/bold cyan]\n"
        "[yellow]Version:[/yellow] 2.0.0\n"
        "[yellow]Author:[/yellow] Abdur Rahman\n"
        "[yellow]License:[/yellow] MIT",
        title="ℹ️ Version Info",
        border_style="cyan"
    ))


# ============================================================================
# LIST COMMANDS
# ============================================================================


@app.command()
def list_methods():
    """List available training methods."""
    from ..trainers import get_available_methods
    
    methods = get_available_methods()
    
    table = Table(title="🎯 Available Training Methods", show_header=True)
    table.add_column("Method", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Memory", style="yellow")
    
    method_info = {
        "lora": ("LoRA - Parameter-efficient fine-tuning", "Medium"),
        "qlora": ("QLoRA - Quantized LoRA for large models", "Low"),
        "full_finetuning": ("Full Fine-tuning - Train all parameters", "High")
    }
    
    for method in methods:
        info = method_info.get(method.value, ("", ""))
        table.add_row(method.value, info[0], info[1])
    
    console.print(table)


@app.command()
def list_metrics():
    """List available evaluation metrics."""
    from ..evaluation import list_available_metrics
    
    metrics = list_available_metrics()
    
    table = Table(title="📊 Available Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Description", style="white")
    
    metric_descriptions = {
        "rouge1": "ROUGE-1 - Unigram overlap",
        "rouge2": "ROUGE-2 - Bigram overlap",
        "rougeL": "ROUGE-L - Longest common subsequence",
        "bleu": "BLEU - N-gram precision",
        "f1": "F1 Score - Token-level F1",
        "exact_match": "Exact Match - Percentage of exact matches",
        "accuracy": "Accuracy - Classification accuracy"
    }
    
    for metric in metrics:
        desc = metric_descriptions.get(metric, "")
        table.add_row(metric, desc)
    
    console.print(table)


# ============================================================================
# IMPORT SUBCOMMANDS
# ============================================================================


# Import command groups
from .commands import train, evaluate, config, recommend


# Register command groups
app.add_typer(train.app, name="train", help="🚀 Train models")
app.add_typer(evaluate.app, name="evaluate", help="📊 Evaluate models")
app.add_typer(config.app, name="config", help="⚙️ Manage configurations")
app.add_typer(recommend.app, name="recommend", help="💡 Get recommendations")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()