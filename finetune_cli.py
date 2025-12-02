#!/usr/bin/env python3
"""
LLM Fine-Tuning CLI Tool - Extended Edition
Supports: LoRA, QLoRA, AdaLoRA with user-friendly method selection
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    PeftModel,
    AdaLoraConfig,
    prepare_model_for_kbit_training
)
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
        self.method = None  # Track which method is being used
        
    def load_model(self, method: str = "lora", load_in_4bit: bool = False, load_in_8bit: bool = False):
        """Load the base model and tokenizer with optional quantization"""
        print(f"\n🔄 Loading model: {self.model_name}")
        print(f"📍 Method: {method.upper()}")
        print(f"📍 Device: {self.device}")
        
        self.method = method
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for QLoRA
        model_kwargs = {}
        
        if method == "qlora" or load_in_4bit or load_in_8bit:
            print(f"⚙️  Configuring quantization...")
            
            if load_in_4bit or (method == "qlora" and not load_in_8bit):
                print("   Using 4-bit quantization (NF4)")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs['quantization_config'] = bnb_config
                model_kwargs['device_map'] = "auto"
            elif load_in_8bit:
                print("   Using 8-bit quantization")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                model_kwargs['quantization_config'] = bnb_config
                model_kwargs['device_map'] = "auto"
        else:
            # Standard loading for LoRA and AdaLoRA
            if self.device == "cuda":
                model_kwargs['torch_dtype'] = torch.float16
                model_kwargs['device_map'] = "auto"
        
        model_kwargs['low_cpu_mem_usage'] = True
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Prepare model for k-bit training if using quantization
        if method == "qlora" or load_in_4bit or load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
            print("✅ Model prepared for quantized training")
        
        print("✅ Model loaded successfully\n")
        
    def load_dataset_from_source(self, dataset_source: str, dataset_config: Optional[str] = None, 
                                  split: str = "train", num_samples: Optional[int] = None,
                                  data_files: Optional[str] = None) -> Dataset:
        """Load dataset from local file or HuggingFace hub"""
        print(f"\n📚 Loading dataset from: {dataset_source}")
        
        if os.path.exists(dataset_source):
            # Load from local file
            if dataset_source.endswith('.json'):
                with open(dataset_source, 'r') as f:
                    data = json.load(f)
                dataset = Dataset.from_dict(data if isinstance(data, dict) else {"data": data})
            elif dataset_source.endswith('.jsonl'):
                import pandas as pd
                df = pd.read_json(dataset_source, lines=True)
                dataset = Dataset.from_pandas(df)
            elif dataset_source.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(dataset_source)
                dataset = Dataset.from_pandas(df)
            elif dataset_source.endswith('.txt'):
                with open(dataset_source, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                dataset = Dataset.from_dict({"text": lines})
            else:
                raise ValueError("Supported formats: .json, .jsonl, .csv, .txt")
        else:
            # Load from HuggingFace hub
            load_kwargs = {}
            if dataset_config:
                load_kwargs['name'] = dataset_config
            if data_files:
                load_kwargs['data_files'] = data_files
            
            if data_files:
                print(f"📄 Loading specific file(s): {data_files}")
            
            dataset = load_dataset(dataset_source, split=split, **load_kwargs)
        
        # Limit number of samples if specified
        if num_samples and num_samples < len(dataset):
            print(f"🔪 Selecting {num_samples} samples from {len(dataset)} total samples")
            dataset = dataset.select(range(num_samples))
        
        print(f"✅ Dataset loaded: {len(dataset)} samples\n")
        return dataset
    
    def detect_text_columns(self, dataset: Dataset) -> List[str]:
        """Auto-detect text columns in the dataset"""
        text_columns = []
        
        # Common text column names
        common_names = ['text', 'content', 'input', 'prompt', 'instruction', 'question', 'answer', 'output', 'response']
        
        for col in dataset.column_names:
            # Check if column name matches common patterns
            if col.lower() in common_names:
                text_columns.append(col)
            # Check if column contains string data
            elif len(dataset) > 0 and isinstance(dataset[0][col], str):
                text_columns.append(col)
        
        return text_columns
    
    def prepare_dataset(self, dataset: Dataset, text_columns: Optional[List[str]] = None, max_length: int = 512):
        """Tokenize and prepare dataset for training"""
        print(f"🔧 Preparing dataset (max_length={max_length})...")
        
        # Auto-detect text columns if not provided
        if text_columns is None:
            text_columns = self.detect_text_columns(dataset)
            print(f"📋 Detected text columns: {text_columns}")
        
        if not text_columns:
            raise ValueError("No text columns found in dataset")
        
        def tokenize_function(examples):
            # Combine all text columns into single text
            if len(text_columns) == 1:
                texts = examples[text_columns[0]]
            else:
                # Combine multiple columns with newlines
                texts = []
                num_examples = len(examples[text_columns[0]])
                for i in range(num_examples):
                    combined = " ".join([
                        f"{col}: {examples[col][i]}" 
                        for col in text_columns 
                        if examples[col][i]
                    ])
                    texts.append(combined)
            
            return self.tokenizer(
                texts,
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
        return tokenized_dataset, text_columns
    
    def get_target_modules(self):
        """Automatically detect target modules for LoRA based on model architecture"""
        print("\n🔍 Detecting target modules...")
        
        # Get all module names
        module_names = set()
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_names.add(name.split('.')[-1])
        
        # Common patterns for different architectures
        target_patterns = [
            # Transformer attention projections
            ["q_proj", "v_proj", "k_proj", "o_proj"],
            ["query", "value", "key", "dense"],
            ["q_lin", "v_lin", "k_lin", "out_lin"],
            ["c_attn", "c_proj"],  # GPT-2 style
            ["qkv_proj", "out_proj"],
        ]
        
        # Find matching patterns
        for pattern in target_patterns:
            if all(module in module_names for module in pattern[:2]):  # At least 2 modules match
                matched = [m for m in pattern if m in module_names]
                print(f"✅ Detected target modules: {matched}")
                return matched
        
        # Fallback: find any linear layers
        linear_modules = [name for name in module_names if 'lin' in name.lower() or 'proj' in name.lower() or 'fc' in name.lower()]
        if linear_modules:
            print(f"✅ Using detected linear modules: {linear_modules[:4]}")
            return linear_modules[:4]
        
        print("⚠️  Could not auto-detect. Using 'all-linear' fallback.")
        return "all-linear"
    
    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None):
        """Configure and apply standard LoRA to the model"""
        print(f"\n🎯 Setting up LoRA configuration...")
        print(f"   - r: {r}")
        print(f"   - alpha: {lora_alpha}")
        print(f"   - dropout: {lora_dropout}")
        
        if target_modules is None:
            target_modules = self.get_target_modules()
        
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
    
    def setup_qlora(self, r: int = 16, lora_alpha: int = 64, lora_dropout: float = 0.1,
                    target_modules: Optional[List[str]] = None):
        """Configure and apply QLoRA (LoRA on quantized model)"""
        print(f"\n🎯 Setting up QLoRA configuration...")
        print(f"   - r: {r} (higher rank for quantized models)")
        print(f"   - alpha: {lora_alpha}")
        print(f"   - dropout: {lora_dropout}")
        print(f"   - Model is quantized (4-bit or 8-bit)")
        
        if target_modules is None:
            target_modules = self.get_target_modules()
        
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
        print("✅ QLoRA applied\n")
        
        # Print memory savings estimate
        self._print_qlora_memory_savings()
    
    def _print_qlora_memory_savings(self):
        """Print estimated memory savings from quantization"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate memory usage
        quant_memory = total_params * 0.5 / 1e9  # 4-bit = 0.5 bytes per param
        full_memory = total_params * 4 / 1e9     # FP32 = 4 bytes per param
        savings_ratio = 8.0
        
        print(f"\n💾 QLoRA Memory Savings:")
        print(f"   Quantized model: ~{quant_memory:.2f} GB")
        print(f"   Full precision would be: ~{full_memory:.2f} GB")
        print(f"   Memory savings: ~{savings_ratio:.1f}x\n")
    
    def setup_adalora(self, target_r: int = 8, init_r: int = 12, lora_alpha: int = 32, 
                      lora_dropout: float = 0.1, target_modules: Optional[List[str]] = None,
                      tinit: int = 0, tfinal: int = 0, deltaT: int = 10,
                      beta1: float = 0.85, beta2: float = 0.85, orth_reg_weight: float = 0.5):
        """Configure and apply AdaLoRA with dynamic rank allocation"""
        print(f"\n🎯 Setting up AdaLoRA configuration (Adaptive LoRA)...")
        print(f"   - target_r: {target_r} (target average rank)")
        print(f"   - init_r: {init_r} (initial rank, will be pruned)")
        print(f"   - alpha: {lora_alpha}")
        print(f"   - dropout: {lora_dropout}")
        print(f"   - Rank allocation: DYNAMIC (importance-based)")
        
        if target_modules is None:
            target_modules = self.get_target_modules()
        
        adalora_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=target_r,
            init_r=init_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            # Pruning schedule
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            # Importance scoring parameters
            beta1=beta1,
            beta2=beta2,
            # Regularization
            orth_reg_weight=orth_reg_weight,
        )
        
        self.peft_model = get_peft_model(self.model, adalora_config)
        self.peft_model.print_trainable_parameters()
        print("✅ AdaLoRA applied")
        print("   Ranks will be dynamically adjusted during training\n")
    
    def train(self, train_dataset: Dataset, num_epochs: int = 3, 
              batch_size: int = 4, learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4):
        """Train the model with the configured method"""
        print(f"\n🚀 Starting training with {self.method.upper()}...")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Gradient accumulation: {gradient_accumulation_steps}\n")
        
        # Determine if using FP16 based on device and method
        use_fp16 = self.device == "cuda" and self.method != "qlora"
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="epoch",
            report_to=None,
            # QLoRA specific optimizations
            optim="paged_adamw_32bit" if self.method == "qlora" else "adamw_torch",
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
        model_type = "(Fine-tuned)" if use_finetuned else "(Base)"
        print(f"ROUGE Scores {model_type}:")
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


def print_method_info():
    """Print information about available methods"""
    print("\n" + "="*70)
    print("📚 FINE-TUNING METHODS GUIDE")
    print("="*70)
    
    print("\n1️⃣  LoRA (Low-Rank Adaptation)")
    print("   • Memory: Moderate (~50% of full fine-tuning)")
    print("   • Trainable params: ~0.1-1%")
    print("   • Best for: General purpose, balanced efficiency")
    print("   • GPU requirement: 8GB+ VRAM")
    
    print("\n2️⃣  QLoRA (Quantized LoRA)")
    print("   • Memory: Very Low (~12-25% of full fine-tuning)")
    print("   • Trainable params: ~0.1-1%")
    print("   • Best for: Large models (7B+), limited GPU memory")
    print("   • GPU requirement: 6GB+ VRAM (can run 7B models!)")
    
    print("\n3️⃣  AdaLoRA (Adaptive LoRA)")
    print("   • Memory: Moderate (same as LoRA)")
    print("   • Trainable params: ~0.1-1%")
    print("   • Best for: Optimal performance, automatic rank allocation")
    print("   • GPU requirement: 8GB+ VRAM")
    print("   • Special: Dynamically adjusts ranks during training")
    
    print("\n" + "="*70 + "\n")


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def main():
    print("="*70)
    print("🤖 LLM Fine-Tuning CLI Tool - Extended Edition")
    print("   Supports: LoRA, QLoRA, AdaLoRA")
    print("="*70)
    
    # Step 1: Method Selection
    print("\n📝 STEP 1: Select Fine-Tuning Method")
    print_method_info()
    
    print("Available methods:")
    print("  1. LoRA       - Balanced efficiency and performance")
    print("  2. QLoRA      - Maximum memory efficiency (for large models)")
    print("  3. AdaLoRA    - Adaptive rank allocation (best performance)")
    print("  4. Show detailed info about methods")
    
    method_choice = get_user_input("\nSelect method [1-4]", "1")
    
    if method_choice == "4":
        print_method_info()
        method_choice = get_user_input("Select method [1-3]", "1")
    
    method_map = {
        "1": "lora",
        "2": "qlora",
        "3": "adalora"
    }
    
    selected_method = method_map.get(method_choice, "lora")
    
    print(f"\n✅ Selected method: {selected_method.upper()}")
    
    # Step 2: Model Configuration
    print("\n📝 STEP 2: Model Configuration")
    model_name = get_user_input("Enter model name (e.g., gpt2, facebook/opt-125m)", "gpt2")
    output_dir = get_user_input("Enter output directory", "./finetuned_model")
    
    finetuner = LLMFineTuner(model_name, output_dir)
    
    # Load model with appropriate configuration
    if selected_method == "qlora":
        print("\n💡 QLoRA uses 4-bit quantization by default")
        use_8bit = get_user_input("Use 8-bit instead of 4-bit? (yes/no)", "no").lower() == "yes"
        finetuner.load_model(method=selected_method, load_in_4bit=not use_8bit, load_in_8bit=use_8bit)
    else:
        finetuner.load_model(method=selected_method)
    
    # Step 3: Dataset Loading
    print("\n📝 STEP 3: Dataset Configuration")
    print("\n💡 Dataset Loading Options:")
    print("   1. Local file (JSON, JSONL, CSV, TXT)")
    print("   2. HuggingFace dataset repository")
    print("\n   For local files: Enter the file path (e.g., ./data/train.json)")
    print("   For HuggingFace: Enter dataset name (e.g., wikitext, squad)")
    
    dataset_source = get_user_input("\nEnter dataset path or HuggingFace dataset name")
    
    # Check if it's a local file or HF repo
    is_local = os.path.exists(dataset_source)
    
    dataset_config = None
    split = "train"
    data_files = None
    num_samples = None
    
    if not is_local:
        # HuggingFace dataset options
        dataset_config = get_user_input("Enter dataset config/subset (optional, press Enter to skip)", "")
        dataset_config = dataset_config if dataset_config else None
        
        # Ask about specific files
        use_specific_file = get_user_input("Load specific file from repo? (yes/no)", "no").lower()
        if use_specific_file in ['yes', 'y']:
            print("\n💡 Examples:")
            print("   - data/train-00000-of-00001.parquet")
            print("   - train.json")
            print("   - data/*.parquet (all parquet files in data/)")
            data_files = get_user_input("Enter file path/pattern")
        
        # Ask about split
        split = get_user_input("Enter split (train/test/validation)", "train")
    
    # Ask about limiting samples
    limit_samples = get_user_input("Limit number of samples? (yes/no)", "yes").lower()
    if limit_samples in ['yes', 'y']:
        num_samples = int(get_user_input("Enter number of samples to use", "1000"))
    
    dataset = finetuner.load_dataset_from_source(
        dataset_source, 
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        data_files=data_files
    )
    
    # Show dataset structure
    print(f"\n📊 Dataset Structure:")
    print(f"   Columns: {dataset.column_names}")
    if len(dataset) > 0:
        print(f"   Sample: {list(dataset[0].keys())}")
    
    # Prepare dataset with auto-detection
    max_length = int(get_user_input("Enter max sequence length", "512"))
    tokenized_dataset, text_columns = finetuner.prepare_dataset(dataset, max_length=max_length)
    
    print(f"✅ Using columns for training: {text_columns}")
    
    # Step 4: Benchmark Before Fine-tuning
    print("\n📝 STEP 4: Pre-training Benchmark")
    run_benchmark = get_user_input("Run pre-training benchmark? (yes/no)", "yes").lower()
    
    base_scores = None
    if run_benchmark in ['yes', 'y']:
        num_samples = min(10, len(dataset))
        
        # Create test prompts from detected text columns
        test_prompts = []
        for i in range(num_samples):
            if len(text_columns) == 1:
                prompt = str(dataset[i][text_columns[0]])[:50]
            else:
                prompt = " ".join([str(dataset[i][col])[:30] for col in text_columns])[:50]
            test_prompts.append(prompt)
        
        print(f"Using {num_samples} samples for benchmarking...")
        base_scores = finetuner.benchmark(test_prompts, use_finetuned=False)
    
    # Step 5: Method-Specific Configuration
    print(f"\n📝 STEP 5: {selected_method.upper()} Configuration")
    
    if selected_method == "lora":
        print("\n💡 LoRA Parameter Guide:")
        print("   • r (rank): Controls adapter size (4=light, 8=balanced, 16=strong)")
        print("   • alpha: Scaling factor (typically 2x the rank)")
        print("   • dropout: Regularization (0.05=low, 0.1=balanced, 0.2=high)\n")
        
        lora_r = int(get_user_input("Enter LoRA r (rank)", "8"))
        lora_alpha = int(get_user_input("Enter LoRA alpha", "32"))
        lora_dropout = float(get_user_input("Enter LoRA dropout", "0.1"))
        
        finetuner.setup_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    elif selected_method == "qlora":
        print("\n💡 QLoRA Parameter Guide:")
        print("   • Higher rank recommended for quantized models (8-16)")
        print("   • alpha: 4x the rank works well")
        print("   • dropout: 0.1 is standard\n")
        
        lora_r = int(get_user_input("Enter QLoRA r (rank)", "16"))
        lora_alpha = int(get_user_input("Enter QLoRA alpha", "64"))
        lora_dropout = float(get_user_input("Enter QLoRA dropout", "0.1"))
        
        finetuner.setup_qlora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    
    elif selected_method == "adalora":
        print("\n💡 AdaLoRA Parameter Guide:")
        print("   • target_r: Target average rank after pruning (4-16)")
        print("   • init_r: Initial rank before pruning (1.5-2x target_r)")
        print("   • Ranks will be automatically adjusted during training\n")
        
        target_r = int(get_user_input("Enter target rank", "8"))
        init_r = int(get_user_input("Enter initial rank", "12"))
        lora_alpha = int(get_user_input("Enter LoRA alpha", "32"))
        lora_dropout = float(get_user_input("Enter LoRA dropout", "0.1"))
        
        print("\n💡 Advanced AdaLoRA settings (press Enter to use defaults)")
        tinit = int(get_user_input("Pruning start step (0 = from beginning)", "0") or "0")
        tfinal = int(get_user_input("Pruning end step (0 = auto-calculate)", "0") or "0")
        deltaT = int(get_user_input("Steps between pruning iterations", "10") or "10")
        
        finetuner.setup_adalora(
            target_r=target_r,
            init_r=init_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT
        )
    
    # Step 6: Training Configuration
    print("\n📝 STEP 6: Training Configuration")
    num_epochs = int(get_user_input("Enter number of epochs", "3"))
    batch_size = int(get_user_input("Enter batch size", "4"))
    learning_rate = float(get_user_input("Enter learning rate", "2e-4"))
    gradient_accum = int(get_user_input("Enter gradient accumulation steps", "4"))
    
    # Train
    finetuner.train(
        tokenized_dataset, 
        num_epochs=num_epochs, 
        batch_size=batch_size, 
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accum
    )
    
    # Step 7: Benchmark After Fine-tuning
    if run_benchmark in ['yes', 'y']:
        print("\n📝 STEP 7: Post-training Benchmark")
        finetuned_scores = finetuner.benchmark(test_prompts, use_finetuned=True)
        
        # Show comparison
        print("\n" + "="*70)
        print("📊 PERFORMANCE COMPARISON")
        print("="*70)
        print(f"{'Metric':<12} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-"*70)
        for metric in base_scores:
            base = base_scores[metric]
            finetuned = finetuned_scores[metric]
            improvement = ((finetuned - base) / base * 100) if base > 0 else 0
            print(f"{metric.upper():<12} {base:<15.4f} {finetuned:<15.4f} {improvement:+.2f}%")
        print("="*70 + "\n")
    
    # Step 8: Upload to HuggingFace (Optional)
    print("\n📝 STEP 8: HuggingFace Upload (Optional)")
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
    
    print("\n" + "="*70)
    print(f"🎉 Fine-tuning complete using {selected_method.upper()}!")
    print("="*70)
    print(f"\n✅ Model saved to: {output_dir}")
    print(f"✅ Method used: {selected_method.upper()}")
    print(f"✅ Tokenizer saved")
    
    if selected_method == "adalora":
        print("\n💡 AdaLoRA Info:")
        print("   - Ranks were dynamically adjusted during training")
        print("   - Final rank distribution is optimized for your task")
    elif selected_method == "qlora":
        print("\n💡 QLoRA Info:")
        print("   - Base model was quantized (4-bit or 8-bit)")
        print("   - Only LoRA adapters are saved")
        print("   - Significantly reduced memory footprint")
    
    print("\n📚 Next Steps:")
    print("   1. Test the model with your own prompts")
    print("   2. Evaluate on a held-out test set")
    print("   3. Deploy or share on HuggingFace Hub")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)