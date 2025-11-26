import typer
from typing import Optional
from finetunecli.training.trainer import LLMFineTuner
import os

train_app = typer.Typer(help="Fine-tune LLMs using LoRA")

@train_app.command("start")
def start_training(
    model_name: str = typer.Option("gpt2", help="Model name (e.g., gpt2, facebook/opt-125m)"),
    output_dir: str = typer.Option("./finetuned_model", help="Output directory for the fine-tuned model"),
    dataset_source: str = typer.Option(..., help="Path to local dataset or HuggingFace dataset name"),
    dataset_config: Optional[str] = typer.Option(None, help="Dataset config/subset name"),
    data_files: Optional[str] = typer.Option(None, help="Specific data files to load"),
    split: str = typer.Option("train", help="Dataset split to use"),
    num_samples: Optional[int] = typer.Option(None, help="Limit number of samples"),
    max_length: int = typer.Option(512, help="Max sequence length"),
    lora_r: int = typer.Option(8, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.1, help="LoRA dropout"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    upload_to_hf: bool = typer.Option(False, help="Upload to HuggingFace after training"),
    hf_repo: Optional[str] = typer.Option(None, help="HuggingFace repository name"),
    hf_token: Optional[str] = typer.Option(None, help="HuggingFace token"),
    private_repo: bool = typer.Option(False, help="Make HuggingFace repository private"),
):
    """
    Start the fine-tuning process.
    """
    print("="*60)
    print("🤖 LLM Fine-Tuning CLI Tool with LoRA")
    print("="*60)
    
    finetuner = LLMFineTuner(model_name, output_dir)
    finetuner.load_model()
    
    # Load Dataset
    dataset = finetuner.load_dataset_from_source(
        dataset_source, 
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        data_files=data_files
    )
    
    # Prepare Dataset
    tokenized_dataset, text_columns = finetuner.prepare_dataset(dataset, max_length=max_length)
    print(f"✅ Using columns for training: {text_columns}")
    
    # Pre-training Benchmark
    print("\n📝 Pre-training Benchmark")
    num_bench_samples = min(10, len(dataset))
    test_prompts = []
    for i in range(num_bench_samples):
        if len(text_columns) == 1:
            prompt = str(dataset[i][text_columns[0]])[:50]
        else:
            prompt = " ".join([str(dataset[i][col])[:30] for col in text_columns])[:50]
        test_prompts.append(prompt)
        
    base_scores = finetuner.benchmark(test_prompts, use_finetuned=False)
    
    # Setup LoRA
    finetuner.setup_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    # Train
    finetuner.train(tokenized_dataset, epochs, batch_size, learning_rate)
    
    # Post-training Benchmark
    print("\n📝 Post-training Benchmark")
    finetuned_scores = finetuner.benchmark(test_prompts, use_finetuned=True)
    
    # Comparison
    print("\n" + "="*60)
    print("📊 PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Metric':<12} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*60)
    for metric in base_scores:
        base = base_scores[metric]
        finetuned = finetuned_scores[metric]
        improvement = ((finetuned - base) / base * 100) if base > 0 else 0
        print(f"{metric.upper():<12} {base:<15.4f} {finetuned:<15.4f} {improvement:+.2f}%")
    print("="*60 + "\n")
    
    # Upload to HF
    if upload_to_hf:
        if not hf_repo:
            print("❌ Error: --hf-repo is required when --upload-to-hf is set.")
            return
            
        finetuner.upload_to_huggingface(
            hf_repo, 
            token=hf_token, 
            create_new=True, 
            private=private_repo
        )
