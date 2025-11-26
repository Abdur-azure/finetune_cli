"""
Unified Interactive CLI for LLM Fine-tuning
Implements the complete 12-step workflow with hierarchical technique selection
"""
import typer
import os
from typing import Optional, Dict, Any
from pathlib import Path

# Import existing trainers
from finetunecli.quantization.lora.lora_trainer import LoraTrainer
from finetunecli.config.lora_config import LoraConfig
from finetunecli.quantization.qlora.qlora_trainer import QLoraTrainer
from finetunecli.config.qlora_config import QLoraConfig
from finetunecli.quantization.prompt_tuning.prompt_tuning_trainer import PromptTuningTrainer
from finetunecli.config.prompt_tuning_config import PromptTuningConfig

# Import benchmarking
from finetunecli.benchmarking.rouge_metric import RougeMetric

# Import utilities
from finetunecli.utils.dataset_loader import load_json_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

unified_app = typer.Typer(help="Unified Interactive Fine-tuning Workflow")


class UnifiedFineTuner:
    """Orchestrates the complete fine-tuning workflow"""
    
    def __init__(self):
        self.model_name = None
        self.dataset_path = None
        self.output_dir = None
        self.technique_category = None
        self.technique_name = None
        self.benchmark_type = None
        self.base_model = None
        self.tokenizer = None
        self.base_scores = None
        self.finetuned_scores = None
        
    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60)
        
    def print_step(self, step_num: int, text: str):
        """Print step header"""
        print(f"\n📝 STEP {step_num}: {text}")
        print("-" * 60)
    
    def select_model(self):
        """Step 1: Model Selection"""
        self.print_step(1, "Model Selection")
        print("\n💡 Enter a model from HuggingFace (e.g., gpt2, facebook/opt-125m)")
        print("   Or provide a local path to your model")
        
        self.model_name = typer.prompt("\nModel name or path", default="gpt2")
        print(f"✅ Selected model: {self.model_name}")
        
    def select_dataset(self):
        """Step 2: Dataset Selection"""
        self.print_step(2, "Dataset Selection")
        print("\n💡 Enter a dataset path:")
        print("   - Local file: ./data/train.json")
        print("   - HuggingFace dataset: wikitext, squad, etc.")
        
        self.dataset_path = typer.prompt("\nDataset path")
        
        # Validate dataset exists if local
        if os.path.exists(self.dataset_path):
            print(f"✅ Found local dataset: {self.dataset_path}")
        else:
            print(f"✅ Will load from HuggingFace: {self.dataset_path}")
    
    def select_technique(self):
        """Step 3: Fine-tuning Technique Selection"""
        self.print_step(3, "Fine-tuning Technique Selection")
        
        print("\n🎯 Select Fine-tuning Category:")
        print("   1. Quantization (LoRA, QLoRA, Prompt Tuning, Prefix Tuning)")
        print("   2. Distillation (KD, RLHF, SFT)")
        print("   3. Pruning (Magnitude, Movement, NVIDIA Ampere)")
        
        category_choice = typer.prompt("\nSelect category (1-3)", type=int)
        
        if category_choice == 1:
            self.technique_category = "quantization"
            print("\n📦 Quantization Techniques:")
            print("   1. LoRA (Low-Rank Adaptation)")
            print("   2. QLoRA (Quantized LoRA)")
            print("   3. Prompt Tuning")
            print("   4. Prefix Tuning")
            
            tech_choice = typer.prompt("\nSelect technique (1-4)", type=int)
            techniques = ["lora", "qlora", "prompt_tuning", "prefix_tuning"]
            self.technique_name = techniques[tech_choice - 1]
            
        elif category_choice == 2:
            self.technique_category = "distillation"
            print("\n🧪 Distillation Techniques:")
            print("   1. Knowledge Distillation (KD)")
            print("   2. RLHF (Reinforcement Learning from Human Feedback)")
            print("   3. SFT (Supervised Fine-Tuning)")
            
            tech_choice = typer.prompt("\nSelect technique (1-3)", type=int)
            techniques = ["kd", "rlhf", "sft"]
            self.technique_name = techniques[tech_choice - 1]
            
        elif category_choice == 3:
            self.technique_category = "pruning"
            print("\n✂️ Pruning Techniques:")
            print("   1. Magnitude Pruning")
            print("   2. Movement Pruning")
            print("   3. NVIDIA Ampere Pruning")
            
            tech_choice = typer.prompt("\nSelect technique (1-3)", type=int)
            techniques = ["magnitude", "movement", "nvidia_ampere"]
            self.technique_name = techniques[tech_choice - 1]
        
        print(f"\n✅ Selected: {self.technique_category.upper()} → {self.technique_name.upper()}")
    
    def select_benchmark(self):
        """Step 4: Benchmark Selection"""
        self.print_step(4, "Benchmark Selection")
        
        print("\n📊 Select Benchmark Metric:")
        print("   1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)")
        print("   2. BLEU (Bilingual Evaluation Understudy)")
        print("   3. BERTScore (Contextual Embeddings)")
        print("   4. Perplexity")
        print("   5. Exact Match")
        print("   6. Cosine Similarity")
        
        bench_choice = typer.prompt("\nSelect benchmark (1-6)", type=int)
        benchmarks = ["rouge", "bleu", "bertscore", "perplexity", "exact_match", "cosine"]
        self.benchmark_type = benchmarks[bench_choice - 1]
        
        print(f"✅ Selected benchmark: {self.benchmark_type.upper()}")
    
    def select_output_dir(self):
        """Step 5: Output Directory Selection"""
        self.print_step(5, "Output Directory")
        
        default_dir = f"./finetuned_{self.technique_name}"
        self.output_dir = typer.prompt("\nOutput directory for fine-tuned model", default=default_dir)
        
        # Create directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {self.output_dir}")
    
    def benchmark_base_model(self):
        """Step 6: Benchmark Base Model"""
        self.print_step(6, "Base Model Benchmarking")
        
        print(f"\n🔄 Loading base model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded")
        
        # Load dataset for benchmarking
        print(f"\n📚 Loading dataset for benchmarking...")
        if os.path.exists(self.dataset_path):
            dataset = load_json_dataset(self.dataset_path)
        else:
            print("⚠️  HuggingFace dataset loading not implemented in benchmark yet")
            print("   Using sample prompts for demonstration")
            dataset = [{"input": "Hello, how are you?", "output": "I'm doing well, thank you!"}]
        
        # Run benchmark
        print(f"\n📊 Running {self.benchmark_type.upper()} benchmark on base model...")
        self.base_scores = self._run_benchmark(self.base_model, dataset[:10])
        
        print(f"\n✅ Base Model Scores:")
        for metric, score in self.base_scores.items():
            print(f"   {metric}: {score:.4f}")
    
    def get_technique_parameters(self) -> Dict[str, Any]:
        """Step 7: Get Parameters for Selected Technique"""
        self.print_step(7, f"Parameters for {self.technique_name.upper()}")
        
        params = {}
        
        if self.technique_name == "lora":
            print("\n💡 LoRA Parameter Guide:")
            print("   • r (rank): 4-64, controls adapter size")
            print("   • alpha: typically 2x the rank")
            print("   • dropout: 0.05-0.2 for regularization")
            
            params['r'] = typer.prompt("LoRA rank (r)", type=int, default=8)
            params['alpha'] = typer.prompt("LoRA alpha", type=int, default=32)
            params['dropout'] = typer.prompt("LoRA dropout", type=float, default=0.1)
            params['epochs'] = typer.prompt("Training epochs", type=int, default=3)
            params['batch_size'] = typer.prompt("Batch size", type=int, default=4)
            params['lr'] = typer.prompt("Learning rate", type=float, default=2e-4)
            
        elif self.technique_name == "qlora":
            print("\n💡 QLoRA Parameter Guide:")
            print("   • r (rank): 4-64, controls adapter size")
            print("   • alpha: typically 2x the rank")
            print("   • dropout: 0.05-0.2 for regularization")
            print("   • bits: 4 or 8 (4-bit recommended for memory efficiency)")
            print("   • quant_type: nf4 (Normal Float 4) or fp4")
            
            params['r'] = typer.prompt("LoRA rank (r)", type=int, default=8)
            params['alpha'] = typer.prompt("LoRA alpha", type=int, default=32)
            params['dropout'] = typer.prompt("LoRA dropout", type=float, default=0.1)
            params['bits'] = typer.prompt("Quantization bits (4 or 8)", type=int, default=4)
            params['quant_type'] = typer.prompt("Quantization type (nf4/fp4)", default="nf4")
            params['epochs'] = typer.prompt("Training epochs", type=int, default=3)
            params['batch_size'] = typer.prompt("Batch size", type=int, default=2)
            params['lr'] = typer.prompt("Learning rate", type=float, default=2e-4)
            
        elif self.technique_name == "prompt_tuning":
            print("\n💡 Prompt Tuning Parameter Guide:")
            print("   • num_virtual_tokens: 8-100, number of soft prompt tokens")
            print("   • init_method: TEXT (from text) or RANDOM")
            print("   • learning_rate: 1e-2 to 5e-2 (higher than LoRA)")
            
            params['num_virtual_tokens'] = typer.prompt("Number of virtual tokens", type=int, default=20)
            params['init_method'] = typer.prompt("Initialization method (TEXT/RANDOM)", default="TEXT")
            
            if params['init_method'].upper() == "TEXT":
                print("\n💡 Example init texts:")
                print("   • Classification: 'Classify if the text is positive or negative:'")
                print("   • Summarization: 'Summarize the following text:'")
                print("   • Q&A: 'Answer the question based on the context:'")
                params['init_text'] = typer.prompt("Initialization text", default="Classify if the text is positive or negative:")
            else:
                params['init_text'] = ""
            
            params['epochs'] = typer.prompt("Training epochs", type=int, default=5)
            params['batch_size'] = typer.prompt("Batch size", type=int, default=8)
            params['lr'] = typer.prompt("Learning rate", type=float, default=3e-2)
            
        elif self.technique_name == "prefix_tuning":
            print(f"\n⚠️  {self.technique_name.upper()} implementation is pending")
            print("   Using default LoRA parameters as fallback")
            params = {'r': 8, 'alpha': 32, 'dropout': 0.1, 'epochs': 3, 'batch_size': 4, 'lr': 2e-4}
            
        elif self.technique_category == "distillation":
            print(f"\n⚠️  {self.technique_name.upper()} implementation is pending")
            params = {}
            
        elif self.technique_category == "pruning":
            print(f"\n⚠️  {self.technique_name.upper()} implementation is pending")
            params = {}
        
        return params
    
    def run_finetuning(self, params: Dict[str, Any]):
        """Step 8: Run Fine-tuning"""
        self.print_step(8, f"Fine-tuning with {self.technique_name.upper()}")
        
        if self.technique_name == "lora":
            # Use existing LoRA trainer
            config = LoraConfig(
                model_name=self.model_name,
                dataset_path=self.dataset_path,
                output_dir=self.output_dir,
                r=params['r'],
                alpha=params['alpha'],
                dropout=params['dropout'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                lr=params['lr']
            )
            
            trainer = LoraTrainer(config)
            trainer.train()
            
        elif self.technique_name == "qlora":
            # Use QLoRA trainer
            config = QLoraConfig(
                model_name=self.model_name,
                dataset_path=self.dataset_path,
                output_dir=self.output_dir,
                r=params['r'],
                alpha=params['alpha'],
                dropout=params['dropout'],
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                lr=params['lr'],
                bits=params.get('bits', 4),
                quant_type=params.get('quant_type', 'nf4'),
                use_double_quant=True
            )
            
            trainer = QLoraTrainer(config)
            trainer.train()
            
        elif self.technique_name == "prompt_tuning":
            # Use Prompt Tuning trainer
            config = PromptTuningConfig(
                model_name=self.model_name,
                dataset_path=self.dataset_path,
                output_dir=self.output_dir,
                num_virtual_tokens=params['num_virtual_tokens'],
                prompt_tuning_init=params['init_method'].upper(),
                prompt_tuning_init_text=params.get('init_text', ""),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                lr=params['lr']
            )
            
            trainer = PromptTuningTrainer(config)
            trainer.train()
            
        else:
            print(f"\n⚠️  {self.technique_name.upper()} trainer not yet implemented")
            print("   Skipping fine-tuning step")
    
    def benchmark_finetuned_model(self):
        """Step 9: Benchmark Fine-tuned Model"""
        self.print_step(9, "Fine-tuned Model Benchmarking")
        
        if not os.path.exists(self.output_dir):
            print("⚠️  Fine-tuned model not found, skipping benchmark")
            return
        
        print(f"\n🔄 Loading fine-tuned model from: {self.output_dir}")
        
        # For LoRA, QLoRA, and Prompt Tuning, we need to load with PEFT
        if self.technique_name in ["lora", "qlora", "prompt_tuning", "prefix_tuning"]:
            from peft import PeftModel
            
            # For QLoRA, need to load base with quantization config
            if self.technique_name == "qlora":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                base = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            finetuned_model = PeftModel.from_pretrained(base, self.output_dir)
        else:
            finetuned_model = AutoModelForCausalLM.from_pretrained(self.output_dir)
        
        # Load dataset
        if os.path.exists(self.dataset_path):
            dataset = load_json_dataset(self.dataset_path)
        else:
            dataset = [{"input": "Hello, how are you?", "output": "I'm doing well, thank you!"}]
        
        print(f"\n📊 Running {self.benchmark_type.upper()} benchmark on fine-tuned model...")
        self.finetuned_scores = self._run_benchmark(finetuned_model, dataset[:10])
        
        print(f"\n✅ Fine-tuned Model Scores:")
        for metric, score in self.finetuned_scores.items():
            print(f"   {metric}: {score:.4f}")
    
    def show_comparison(self):
        """Step 10: Show Before/After Comparison"""
        self.print_step(10, "Performance Comparison")
        
        if not self.base_scores or not self.finetuned_scores:
            print("⚠️  Missing benchmark scores, skipping comparison")
            return
        
        self.print_header("📊 BEFORE vs AFTER COMPARISON")
        print(f"\n{'Metric':<20} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-" * 65)
        
        for metric in self.base_scores:
            base = self.base_scores[metric]
            finetuned = self.finetuned_scores.get(metric, 0)
            improvement = ((finetuned - base) / base * 100) if base > 0 else 0
            
            print(f"{metric:<20} {base:<15.4f} {finetuned:<15.4f} {improvement:+.2f}%")
        
        print("=" * 65)
    
    def save_and_display(self):
        """Step 11: Save Model and Display Location"""
        self.print_step(11, "Model Saved")
        
        abs_path = os.path.abspath(self.output_dir)
        
        self.print_header("✅ FINE-TUNING COMPLETE!")
        print(f"\n📁 Model Location: {abs_path}")
        print(f"🎯 Technique: {self.technique_category.upper()} → {self.technique_name.upper()}")
        print(f"📊 Benchmark: {self.benchmark_type.upper()}")
        
        # List saved files
        if os.path.exists(self.output_dir):
            files = os.listdir(self.output_dir)
            print(f"\n📦 Saved Files ({len(files)}):")
            for f in files[:10]:  # Show first 10 files
                print(f"   • {f}")
            if len(files) > 10:
                print(f"   ... and {len(files) - 10} more files")
    
    def upload_to_huggingface(self):
        """Step 12: Optional HuggingFace Upload"""
        self.print_step(12, "HuggingFace Upload (Optional)")
        
        upload = typer.confirm("\n🤗 Upload model to HuggingFace Hub?", default=False)
        
        if not upload:
            print("⏭️  Skipping HuggingFace upload")
            return
        
        print("\n💡 You'll need a HuggingFace token with write access")
        print("   Get it from: https://huggingface.co/settings/tokens")
        
        repo_name = typer.prompt("\nRepository name (username/repo-name)")
        token = typer.prompt("HuggingFace token", hide_input=True)
        private = typer.confirm("Make repository private?", default=False)
        
        # Upload using HF API
        from huggingface_hub import HfApi, login, create_repo
        
        try:
            login(token=token)
            create_repo(repo_name, private=private, exist_ok=True)
            
            api = HfApi()
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            
            print(f"\n✅ Model uploaded successfully!")
            print(f"🔗 https://huggingface.co/{repo_name}")
            
        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
    
    def _run_benchmark(self, model, dataset) -> Dict[str, float]:
        """Run selected benchmark on model"""
        if self.benchmark_type == "rouge":
            scorer = RougeMetric()
            preds, refs = [], []
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.eval()
            
            for row in dataset:
                inp = row.get("input", "")
                ref = row.get("output", "")
                
                inputs = self.tokenizer(inp, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False
                    )
                
                pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                preds.append(pred)
                refs.append(ref)
            
            results = scorer.compute(preds, refs)
            return results
        
        else:
            print(f"⚠️  {self.benchmark_type.upper()} benchmark not yet implemented")
            return {"score": 0.0}
    
    def run_workflow(self):
        """Execute the complete 12-step workflow"""
        self.print_header("🤖 LLM Fine-Tuning - Unified Interactive CLI")
        
        try:
            self.select_model()              # Step 1
            self.select_dataset()            # Step 2
            self.select_technique()          # Step 3
            self.select_benchmark()          # Step 4
            self.select_output_dir()         # Step 5
            self.benchmark_base_model()      # Step 6
            params = self.get_technique_parameters()  # Step 7
            self.run_finetuning(params)      # Step 8
            self.benchmark_finetuned_model() # Step 9
            self.show_comparison()           # Step 10
            self.save_and_display()          # Step 11
            self.upload_to_huggingface()     # Step 12
            
            self.print_header("🎉 WORKFLOW COMPLETE!")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Workflow interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


@unified_app.command("run")
def interactive_workflow():
    """
    Run the complete interactive fine-tuning workflow.
    
    This command guides you through:
    1. Model selection
    2. Dataset selection
    3. Technique selection (Quantization/Distillation/Pruning)
    4. Benchmark selection
    5. Output directory
    6. Base model benchmarking
    7. Parameter configuration
    8. Fine-tuning
    9. Fine-tuned model benchmarking
    10. Performance comparison
    11. Model saving
    12. Optional HuggingFace upload
    """
    finetuner = UnifiedFineTuner()
    finetuner.run_workflow()


if __name__ == "__main__":
    unified_app()
