import typer
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetunecli.benchmarking.rouge_metric import RougeMetric
from finetunecli.config.benchmark_config import BenchmarkConfig
from finetunecli.utils.dataset_loader import load_json_dataset

benchmark_app = typer.Typer(help="Evaluation & Metrics")

@benchmark_app.command("rouge")
def rouge_eval(
    model: str = typer.Option(...),
    data: str = typer.Option(...),
    samples: int = typer.Option(30)
):
    cfg = BenchmarkConfig(model_name=model, dataset_path=data, max_samples=samples)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    dataset = load_json_dataset(cfg.dataset_path)

    preds, refs = [], []

    for i, row in enumerate(dataset):
        if i >= cfg.max_samples:
            break
        inp = row["input"]
        ref = row["output"]

        out = model.generate(
            tokenizer(inp, return_tensors="pt").input_ids,
            max_new_tokens=128
        )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append(ref)

    scorer = RougeMetric()
    results = scorer.compute(preds, refs)

    typer.echo(results)
