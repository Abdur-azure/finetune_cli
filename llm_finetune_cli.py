#!/usr/bin/env python3
"""
LLM Fine-Tuning CLI Tool
A comprehensive tool for fine-tuning LLMs with LoRA, benchmarking, and HuggingFace integration
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from rouge_score import rouge_scorer
from huggingface_hub import HfApi, login, create_repo
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class LLMFineTuner:
    def __init__(self, model_name: str, output_dir: str = "./finetuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_model(self):
        """Load the base model and tokenizer"""
        print(f"\n🔄 Loading model: {self.model_name}")
        print(f"📍 Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully\n")
        
    def load_dataset_from_source(self, dataset_source: str, dataset_config: Optional[str] = None) -> Dataset:
        """Load dataset from local file or HuggingFace hub"""
        print(f"\n📚 Loading dataset from: {dataset_source}")
        
        if os.path.exists(dataset_source):
            # Load from local file
            if dataset_source.endswith('.json'):
                with open(dataset_source, 'r') as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data if isinstance(data, dict) else {"data": data})
            elif dataset_source.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(dataset_source)
                dataset = Dataset.from_pandas(df)
            else:
                raise ValueError("Supported formats: .json, .csv")
        else:
            # Load from HuggingFace hub
            if dataset_config:
                dataset = load_dataset(dataset_source, dataset_config, split="train")
            else:
                dataset = load_dataset(dataset_source, split="train")
        
        print(f"✅ Dataset loaded: {len(dataset)} samples\n")
        return dataset
    
    def prepare_dataset(self, dataset: Dataset, text_column: str, max_length: int = 512):
        """Tokenize and prepare dataset for training"""
        print(f"🔧 Preparing dataset (max_length={max_length})...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print("✅ Dataset prepared\n")
        return tokenized_dataset
    
    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None):
        """Configure and apply LoRA to the model"""
        print(f"\n🎯 Setting up LoRA configuration...")
        print(f"   - r: {r}")
        print(f"   - alpha: {lora_alpha}")
        print(f"   - dropout: {lora_dropout}")
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        print("✅ LoRA applied\n")
        
    def train(self, train_dataset: Dataset, num_epochs: int = 3, 
              batch_size: int = 4, learning_rate: float = 2e-4):
        """Train the model with LoRA"""
        print(f"\n🚀 Starting training...")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}\n")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=self.device == "cuda",
            logging_steps=10,
            save_strategy="epoch",
            report_to=None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        
        # Save the final model
        self.peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ Training complete! Model saved to: {self.output_dir}\n")
    
    def benchmark(self, test_prompts: List[str], use_finetuned: bool = False) -> Dict:
        """Benchmark model performance using ROUGE scores"""
        print(f"\n📊 Running benchmark {'(Fine-tuned)' if use_finetuned else '(Base model)'}...")
        
        model_to_use = self.peft_model if use_finetuned and self.peft_model else self.model
        
        if model_to_use is None:
            if use_finetuned:
                # Load fine-tuned model
                print("Loading fine-tuned model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                self.peft_model = PeftModel.from_pretrained(self.model, self.output_dir)
                model_to_use = self.peft_model
            else:
                raise ValueError("Model not loaded")
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        model_to_use.eval()
        
        for prompt in tqdm(test_prompts, desc="Generating"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Use prompt as reference (in real scenario, you'd have reference texts)
            scores = scorer.score(prompt, generated_text)
            
            for key in all_scores:
                all_scores[key].append(scores[key].fmeasure)
        
        # Calculate averages
        avg_scores = {key: sum(vals) / len(vals) for key, vals in all_scores.items()}
        
        print(f"\n{'='*50}")
        print(f"ROUGE Scores {'(Fine-tuned)' if use_finetuned else '(Base)'):")
        print(f"{'='*50}")
        for metric, score in avg_scores.items():
            print(f"  {metric.upper()}: {score:.4f}")
        print(f"{'='*50}\n")
        
        return avg_scores
    
    def upload_to_huggingface(self, repo_name: str, token: Optional[str] = None, 
                             create_new: bool = False, private: bool = False):
        """Upload fine-tuned model to HuggingFace Hub"""
        print(f"\n🚀 Uploading to HuggingFace Hub...")
        
        if token:
            login(token=token)
        else:
            # Check if already logged in
            try:
                api = HfApi()
                api.whoami()
                print("✅ Already logged in to HuggingFace")
            except:
                print("❌ Not logged in. Please provide a HuggingFace token.")
                return
        
        api = HfApi()
        
        # Create repo if needed
        if create_new:
            try:
                create_repo(repo_name, private=private, exist_ok=True)
                print(f"✅ Repository created/verified: {repo_name}")
            except Exception as e:
                print(f"❌ Error creating repository: {e}")
                return
        
        # Upload model files
        try:
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_name}\n")
        except Exception as e:
            print(f"❌ Error uploading model: {e}\n")


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def main():
    print("="*60)
    print("🤖 LLM Fine-Tuning CLI Tool with LoRA")
    print("="*60)
    
    # Step 1: Model Selection
    print("\n📝 STEP 1: Model Configuration")
    model_name = get_user_input("Enter model name (e.g., gpt2, facebook/opt-125m)", "gpt2")
    output_dir = get_user_input("Enter output directory", "./finetuned_model")
    
    finetuner = LLMFineTuner(model_name, output_dir)
    finetuner.load_model()
    
    # Step 2: Dataset Loading
    print("\n📝 STEP 2: Dataset Configuration")
    dataset_source = get_user_input("Enter dataset path or HuggingFace dataset name")
    dataset_config = get_user_input("Enter dataset config (optional, press Enter to skip)", "")
    text_column = get_user_input("Enter text column name", "text")
    
    dataset = finetuner.load_dataset_from_source(
        dataset_source, 
        dataset_config if dataset_config else None
    )
    
    # Prepare dataset
    max_length = int(get_user_input("Enter max sequence length", "512"))
    tokenized_dataset = finetuner.prepare_dataset(dataset, text_column, max_length)
    
    # Step 3: Benchmark Before Fine-tuning
    print("\n📝 STEP 3: Pre-training Benchmark")
    num_samples = min(10, len(dataset))
    test_prompts = [dataset[i][text_column][:50] for i in range(num_samples)]
    
    print(f"Using {num_samples} samples for benchmarking...")
    base_scores = finetuner.benchmark(test_prompts, use_finetuned=False)
    
    # Step 4: LoRA Configuration
    print("\n📝 STEP 4: LoRA Configuration")
    lora_r = int(get_user_input("Enter LoRA r (rank)", "8"))
    lora_alpha = int(get_user_input("Enter LoRA alpha", "32"))
    lora_dropout = float(get_user_input("Enter LoRA dropout", "0.1"))
    
    finetuner.setup_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    # Step 5: Training
    print("\n📝 STEP 5: Training Configuration")
    num_epochs = int(get_user_input("Enter number of epochs", "3"))
    batch_size = int(get_user_input("Enter batch size", "4"))
    learning_rate = float(get_user_input("Enter learning rate", "2e-4"))
    
    finetuner.train(tokenized_dataset, num_epochs, batch_size, learning_rate)
    
    # Step 6: Benchmark After Fine-tuning
    print("\n📝 STEP 6: Post-training Benchmark")
    finetuned_scores = finetuner.benchmark(test_prompts, use_finetuned=True)
    
    # Show comparison
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
    
    # Step 7: Upload to HuggingFace
    print("\n📝 STEP 7: HuggingFace Upload (Optional)")
    upload = get_user_input("Upload to HuggingFace? (yes/no)", "no").lower()
    
    if upload in ['yes', 'y']:
        repo_name = get_user_input("Enter repository name (username/repo-name)")
        create_new = get_user_input("Create new repository? (yes/no)", "yes").lower() in ['yes', 'y']
        private = get_user_input("Make repository private? (yes/no)", "no").lower() in ['yes', 'y']
        
        print("\nℹ️  You'll need a HuggingFace token with write access")
        print("   Get it from: https://huggingface.co/settings/tokens")
        token = get_user_input("Enter HuggingFace token (or press Enter if already logged in)", "")
        
        finetuner.upload_to_huggingface(
            repo_name, 
            token if token else None, 
            create_new, 
            private
        )
    
    print("\n" + "="*60)
    print("🎉 Fine-tuning complete!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)