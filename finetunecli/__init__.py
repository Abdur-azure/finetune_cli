import typer

from finetunecli.cli.benchmark_cli import benchmark_app
from finetunecli.cli.quantize_cli import quantize_app
from finetunecli.cli.train_cli import train_app
from finetunecli.cli.unified_cli import unified_app
#from finetunecli.cli.prune_cli import prune_app
#from finetunecli.cli.distill_cli import distill_app

app = typer.Typer(help="FinetuneCLI — Modular LLM Finetuning Toolkit")

# Register subcommands
app.add_typer(unified_app, name="finetune")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(quantize_app, name="quantize")
app.add_typer(train_app, name="train")
#app.add_typer(prune_app, name="prune")
#app.add_typer(distill_app, name="distill")

def main():
    app()
