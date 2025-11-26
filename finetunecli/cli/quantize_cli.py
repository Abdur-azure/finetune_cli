import typer
from finetunecli.config.lora_config import LoraConfig
from finetunecli.quantization.lora.lora_trainer import LoraTrainer

quantize_app = typer.Typer(help="Quantization / Finetuning Modules")

@quantize_app.command("train")
def lora_train(
    model: str = typer.Option(...),
    data: str = typer.Option(...),
    out: str = typer.Option("lora_out")
):
    cfg = LoraConfig(model_name=model, dataset_path=data, output_dir=out)
    trainer = LoraTrainer(cfg)
    trainer.train()
