#!/usr/bin/env python3
"""
LLM Fine-Tuning & Distillation CLI Tool
Supports: LoRA, QLoRA, AdaLoRA, Knowledge Distillation
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ============================================================================
# DISTILLATION TRAINER
# ============================================================================

class DistillationTrainer(Trainer):
    """Custom Trainer for Knowledge Distillation"""
    
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha
        
        # Move teacher to same device as student
        if torch.cuda.is_available():
            self.teacher = self.teacher.cuda()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute distillation loss"""
        # Student forward pass
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        
        # Get logits
        student_logits = outputs_student.logits
        teacher_logits = outputs_teacher.logits
        
        # Distillation loss (KL divergence with temperature scaling)
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        
        loss_kd = loss_fct(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * student_loss + (1 - self.alpha) * loss_kd
        
        return (loss, outputs_student) if return_outputs else loss


# ============================================================================
# MAIN FINETUNER CLASS
# ============================================================================

class LLMFineTuner:
    def __init__(self, model_name: str, output_dir: str = "./finetuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.method = None
        self.teacher_model = None  # For distillation
        
    def load_model(self, method: str = "lora", load_in_4bit: bool = False, 
                   load_in_8bit: bool = False):
        """Load the base model and tokenizer with optional quantization"""
        print(f"\n🔄 Loading model: {self.model_name}")
        print(f"📍 Method: {method.upper()}")
        print(f"📍 Device: {self.device}")
        
        self.method = method
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {'low_cpu_mem_usage': True}
        
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
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs['quantization_config'] = bnb_config
                model_kwargs['device_map'] = "auto"
        else:
            if self.device == "cuda":
                model_kwargs['torch_dtype'] = torch.float16
                model_kwargs['device_map'] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if method == "qlora" or load_in_4bit or load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
            print("✅ Model prepared for quantized training")
        
        print("✅ Model loaded successfully\n")
    
    def load_teacher_model(self, teacher_name: str):
        """Load teacher model for distillation"""
        print(f"\n👨‍🏫 Loading teacher model: {teacher_name}")
        
        teacher_kwargs = {'low_cpu_mem_usage': True}
        if self.device == "cuda":
            teacher_kwargs['torch_dtype'] = torch.float16
            teacher_kwargs['device_map'] = "auto"
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            **teacher_kwargs
        )
        self.teacher_model.eval()
        
        print("✅ Teacher model loaded successfully\n")
        
    def load_dataset_from_source(self, dataset_source: str, dataset_config: Optional[str] = None, 
                                  split: str = "train", num_samples: Optional[int] = None,
                                  data_files: Optional[str] = None) -> Dataset:
        """Load dataset from local file or HuggingFace hub"""
        print(f"\n📚 Loading dataset from: {dataset_source}")
        
        if os.path.exists(dataset_source):
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
            load_kwargs = {}
            if dataset_config:
                load_kwargs['name'] = dataset_config
            if data_files:
                load_kwargs['data_files'] = data_files
            
            if data_files:
                print(f"📄 Loading specific file(s): {data_files}")
            
            dataset = load_dataset(dataset_source, split=split, **load_kwargs)
        
        if num_samples and num_samples < len(dataset):
            print(f"🔪 Selecting {num_samples} samples from {len(dataset)} total samples")
            dataset = dataset.select(range(num_samples))
        
        print(f"✅ Dataset loaded: {len(dataset)} samples\n")
        return dataset
    
    def detect_text_columns(self, dataset: Dataset) -> List[str]:
        """Auto-detect text columns in the dataset"""
        text_columns = []
        common_names = ['text', 'content', 'input', 'prompt', 'instruction', 
                       'question', 'answer', 'output', 'response']
        
        for col in dataset.column_names:
            if col.lower() in common_names:
                text_columns.append(col)
            elif len(dataset) > 0 and isinstance(dataset[0][col], str):
                text_columns.append(col)
        
        return text_columns
    
    def prepare_dataset(self, dataset: Dataset, text_columns: Optional[List[str]] = None, 
                       max_length: int = 512):
        """Tokenize and prepare dataset for training"""
        print(f"🔧 Preparing dataset (max_length={max_length})...")
        
        if text_columns is None:
            text_columns = self.detect_text_columns(dataset)
            print(f"📋 Detected text columns: {text_columns}")
        
        if not text_columns:
            raise ValueError("No text columns found in dataset")
        
        def tokenize_function(examples):
            if len(text_columns) == 1:
                texts = examples[text_columns[0]]
            else:
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
        """Automatically detect target modules for LoRA"""
        print("\n🔍 Detecting target modules...")
        
        module_names = set()
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                module_names.add(name.split('.')[-1])
        
        target_patterns = [
            ["q_proj", "v_proj", "k_proj", "o_proj"],
            ["query", "value", "key", "dense"],
            ["q_lin", "v_lin", "k_lin", "out_lin"],
            ["c_attn", "c_proj"],
            ["qkv_proj", "out_proj"],
        ]
        
        for pattern in target_patterns:
            if all(module in module_names for module in pattern[:2]):
                matched = [m for m in pattern if m in module_names]
                print(f"✅ Detected target modules: {matched}")
                return matched
        
        linear_modules = [name for name in module_names 
                         if 'lin' in name.lower() or 'proj' in name.lower() or 'fc' in name.lower()]
        if linear_modules:
            print(f"✅ Using detected linear modules: {linear_modules[:4]}")
            return linear_modules[:4]
        
        print("⚠️  Could not auto-detect. Using 'all-linear' fallback.")
        return "all-linear"
    
    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None):
        """Configure and apply standard LoRA"""
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
        """Configure and apply QLoRA"""
        print(f"\n🎯 Setting up QLoRA configuration...")
        print(f"   - r: {r} (higher rank for quantized models)")
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
        print("✅ QLoRA applied\n")
        self._print_qlora_memory_savings()
    
    def _print_qlora_memory_savings(self):
        """Print estimated memory savings"""
        total_params = sum(p.numel() for p in self.model.parameters())
        quant_memory = total_params * 0.5 / 1e9
        full_memory = total_params * 4 / 1e9
        
        print(f"\n💾 QLoRA Memory Savings:")
        print(f"   Quantized model: ~{quant_memory:.2f} GB")
        print(f"   Full precision: ~{full_memory:.2f} GB")
        print(f"   Savings: ~8.0x\n")
    
    def setup_adalora(self, target_r: int = 8, init_r: int = 12, lora_alpha: int = 32, 
                      lora_dropout: float = 0.1, target_modules: Optional[List[str]] = None,
                      tinit: int = 0, tfinal: int = 0, deltaT: int = 10,
                      beta1: float = 0.85, beta2: float = 0.85, orth_reg_weight: float = 0.5):
        """Configure and apply AdaLoRA"""
        print(f"\n🎯 Setting up AdaLoRA configuration...")
        print(f"   - target_r: {target_r}")
        print(f"   - init_r: {init_r}")
        print(f"   - alpha: {lora_alpha}")
        
        if target_modules is None:
            target_modules = self.get_target_modules()
        
        adalora_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=target_r,
            init_r=init_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2,
            orth_reg_weight=orth_reg_weight,
        )
        
        self.peft_model = get_peft_model(self.model, adalora_config)
        self.peft_model.print_trainable_parameters()
        print("✅ AdaLoRA applied\n")
    
    def train(self, train_dataset: Dataset, num_epochs: int = 3, 
              batch_size: int = 4, learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4):
        """Train with fine-tuning methods"""
        print(f"\n🚀 Starting training with {self.method.upper()}...")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}\n")
        
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
        
        self.peft_model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ Training complete! Model saved to: {self.output_dir}\n")
    
    def train_distillation(self, train_dataset: Dataset, num_epochs: int = 3,
                          batch_size: int = 4, learning_rate: float = 2e-4,
                          gradient_accumulation_steps: int = 4,
                          temperature: float = 2.0, alpha: float = 0.5):
        """Train with knowledge distillation"""
        print(f"\n🚀 Starting Knowledge Distillation training...")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Temperature: {temperature}")
        print(f"   - Alpha (CE weight): {alpha}\n")
        
        use_fp16 = self.device == "cuda"
        
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
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = DistillationTrainer(
            teacher_model=self.teacher_model,
            temperature=temperature,
            alpha=alpha,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ Distillation complete! Model saved to: {self.output_dir}\n")
    
    def benchmark(self, test_prompts: List[str], use_finetuned: bool = False) -> Dict:
        """Benchmark model performance"""
        print(f"\n📊 Running benchmark {'(Fine-tuned)' if use_finetuned else '(Base)'}...")
        
        model_to_use = self.peft_model if use_finetuned and self.peft_model else self.model
        
        if model_to_use is None:
            if use_finetuned:
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
            scores = scorer.score(prompt, generated_text)
            
            for key in all_scores:
                all_scores[key].append(scores[key].fmeasure)
        
        avg_scores = {key: sum(vals) / len(vals) for key, vals in all_scores.items()}
        
        print(f"\n{'='*50}")
        print(f"ROUGE Scores:")
        print(f"{'='*50}")
        for metric, score in avg_scores.items():
            print(f"  {metric.upper()}: {score:.4f}")
        print(f"{'='*50}\n")
        
        return avg_scores
    
    def upload_to_huggingface(self, repo_name: str, token: Optional[str] = None, 
                             create_new: bool = False, private: bool = False):
        """Upload model to HuggingFace Hub"""
        print(f"\n🚀 Uploading to HuggingFace Hub...")
        
        if token:
            login(token=token)
        else:
            try:
                api = HfApi()
                api.whoami()
                print("✅ Already logged in")
            except:
                print("❌ Not logged in. Please provide token.")
                return
        
        api = HfApi()
        
        if create_new:
            try:
                create_repo(repo_name, private=private, exist_ok=True)
                print(f"✅ Repository created: {repo_name}")
            except Exception as e:
                print(f"❌ Error: {e}")
                return
        
        try:
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            print(f"✅ Uploaded to: https://huggingface.co/{repo_name}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_approach_info():
    """Print information about approaches"""
    print("\n" + "="*70)
    print("📚 TRAINING APPROACHES")
    print("="*70)
    
    print("\n🎯 FINE-TUNING")
    print("   • Adapts pre-trained model to specific task/domain")
    print("   • Methods: LoRA, QLoRA, AdaLoRA")
    print("   • Best for: Task specialization, domain adaptation")
    print("   • Output: Task-specific model (same size as base)")
    
    print("\n🧠 KNOWLEDGE DISTILLATION")
    print("   • Transfers knowledge from large teacher to small student")
    print("   • Method: Temperature-scaled softmax + KL divergence")
    print("   • Best for: Model compression, faster inference")
    print("   • Output: Smaller, faster model with similar performance")
    
    print("\n" + "="*70 + "\n")

def print_method_info():
    """Print fine-tuning methods info"""
    print("\n" + "="*70)
    print("📚 FINE-TUNING METHODS")
    print("="*70)
    
    print("\n1️⃣  LoRA (Low-Rank Adaptation)")
    print("   • Memory: ~50% of full fine-tuning")
    print("   • Best for: General purpose, balanced")
    print("   • GPU: 8GB+ VRAM")
    
    print("\n2️⃣  QLoRA (Quantized LoRA)")
    print("   • Memory: ~12-25% of full fine-tuning")
    print("   • Best for: Large models (7B+)")
    print("   • GPU: 6GB+ VRAM")
    
    print("\n3️⃣  AdaLoRA (Adaptive LoRA)")
    print("   • Memory: Same as LoRA")
    print("   • Best for: Optimal performance")
    print("   • GPU: 8GB+ VRAM")
    
    print("\n" + "="*70 + "\n")

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🤖 LLM Fine-Tuning & Distillation CLI Tool")
    print("   Supports: LoRA, QLoRA, AdaLoRA, Knowledge Distillation")
    print("="*70)
    
    # Step 0: Choose Approach
    print("\n📝 STEP 0: Select Training Approach")
    print_approach_info()
    
    print("Available approaches:")
    print("  1. Fine-Tuning    - Adapt model to specific task")
    print("  2. Distillation   - Compress model for efficiency")
    print("  3. Show detailed info")
    
    approach_choice = get_user_input("\nSelect approach [1-3]", "1")
    
    if approach_choice == "3":
        print_approach_info()
        approach_choice = get_user_input("Select approach [1-2]", "1")
    
    use_distillation = approach_choice == "2"
    
    if use_distillation:
        print("\n✅ Selected: Knowledge Distillation")
        
        # Distillation workflow
        print("\n📝 STEP 1: Model Configuration")
        teacher_name = get_user_input("Enter teacher model (larger)", "gpt2-medium")
        student_name = get_user_input("Enter student model (smaller)", "gpt2")
        output_dir = get_user_input("Enter output directory", "./distilled_model")
        
        finetuner = LLMFineTuner(student_name, output_dir)
        
        # Load models
        finetuner.load_teacher_model(teacher_name)
        finetuner.load_model(method="distillation")
        
        # Dataset
        print("\n📝 STEP 2: Dataset Configuration")
        dataset_source = get_user_input("Enter dataset path or HF name")
        
        is_local = os.path.exists(dataset_source)
        dataset_config = None
        split = "train"
        num_samples = None
        
        if not is_local:
            dataset_config = get_user_input("Dataset config (optional)", "") or None
            split = get_user_input("Split", "train")
        
        limit_samples = get_user_input("Limit samples? (yes/no)", "yes").lower()
        if limit_samples in ['yes', 'y']:
            num_samples = int(get_user_input("Number of samples", "1000"))
        
        dataset = finetuner.load_dataset_from_source(
            dataset_source, 
            dataset_config=dataset_config,
            split=split,
            num_samples=num_samples
        )
        
        max_length = int(get_user_input("Max sequence length", "512"))
        tokenized_dataset, _ = finetuner.prepare_dataset(dataset, max_length=max_length)
        
        # Distillation parameters
        print("\n📝 STEP 3: Distillation Configuration")
        print("\n💡 Parameter Guide:")
        print("   • Temperature: Controls softness (2.0-5.0 typical)")
        print("   • Alpha: CE loss weight (0.3-0.5 typical)")
        
        temperature = float(get_user_input("Temperature", "2.0"))
        alpha = float(get_user_input("Alpha (CE weight)", "0.5"))
        
        # Training config
        print("\n📝 STEP 4: Training Configuration")
        num_epochs = int(get_user_input("Epochs", "3"))
        batch_size = int(get_user_input("Batch size", "4"))
        learning_rate = float(get_user_input("Learning rate", "2e-4"))
        
        # Train
        finetuner.train_distillation(
            tokenized_dataset,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temperature,
            alpha=alpha
        )
        
        print(f"\n✅ Distillation complete!")
        print(f"   Student model ({student_name}) learned from teacher ({teacher_name})")
        print(f"   Saved to: {output_dir}")
        
    else:
        # Fine-tuning workflow (existing)
        print("\n✅ Selected: Fine-Tuning")
        
        print("\n📝 STEP 1: Select Fine-Tuning Method")
        print_method_info()
        
        print("Available methods:")
        print("  1. LoRA")
        print("  2. QLoRA")
        print("  3. AdaLoRA")
        print("  4. Show info")
        
        method_choice = get_user_input("\nSelect method [1-4]", "1")
        
        if method_choice == "4":
            print_method_info()
            method_choice = get_user_input("Select method [1-3]", "1")
        
        method_map = {"1": "lora", "2": "qlora", "3": "adalora"}
        selected_method = method_map.get(method_choice, "lora")
        
        print(f"\n✅ Selected: {selected_method.upper()}")
        
        # Continue with existing fine-tuning flow...
        # (Rest of the original main() code for fine-tuning)
        print("\n📝 STEP 2: Model Configuration")
        model_name = get_user_input("Model name", "gpt2")
        output_dir = get_user_input("Output directory", "./finetuned_model")
        
        finetuner = LLMFineTuner(model_name, output_dir)
        
        if selected_method == "qlora":
            print("\n💡 QLoRA uses 4-bit quantization")
            use_8bit = get_user_input("Use 8-bit? (yes/no)", "no").lower() == "yes"
            finetuner.load_model(method=selected_method, load_in_4bit=not use_8bit, load_in_8bit=use_8bit)
        else:
            finetuner.load_model(method=selected_method)
        
        # Dataset loading
        print("\n📝 STEP 3: Dataset Configuration")
        dataset_source = get_user_input("Dataset path or HF name")
        
        is_local = os.path.exists(dataset_source)
        dataset_config = None
        split = "train"
        num_samples = None
        
        if not is_local:
            dataset_config = get_user_input("Dataset config", "") or None
            split = get_user_input("Split", "train")
        
        limit_samples = get_user_input("Limit samples? (yes/no)", "yes").lower()
        if limit_samples in ['yes', 'y']:
            num_samples = int(get_user_input("Number of samples", "1000"))
        
        dataset = finetuner.load_dataset_from_source(
            dataset_source,
            dataset_config=dataset_config,
            split=split,
            num_samples=num_samples
        )
        
        max_length = int(get_user_input("Max sequence length", "512"))
        tokenized_dataset, text_columns = finetuner.prepare_dataset(dataset, max_length=max_length)
        
        # Method-specific setup
        print(f"\n📝 STEP 4: {selected_method.upper()} Configuration")
        
        if selected_method == "lora":
            lora_r = int(get_user_input("LoRA rank", "8"))
            lora_alpha = int(get_user_input("LoRA alpha", "32"))
            lora_dropout = float(get_user_input("LoRA dropout", "0.1"))
            finetuner.setup_lora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        elif selected_method == "qlora":
            lora_r = int(get_user_input("QLoRA rank", "16"))
            lora_alpha = int(get_user_input("QLoRA alpha", "64"))
            lora_dropout = float(get_user_input("QLoRA dropout", "0.1"))
            finetuner.setup_qlora(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        elif selected_method == "adalora":
            target_r = int(get_user_input("Target rank", "8"))
            init_r = int(get_user_input("Initial rank", "12"))
            lora_alpha = int(get_user_input("LoRA alpha", "32"))
            lora_dropout = float(get_user_input("LoRA dropout", "0.1"))
            finetuner.setup_adalora(target_r=target_r, init_r=init_r, 
                                   lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        # Training
        print("\n📝 STEP 5: Training Configuration")
        num_epochs = int(get_user_input("Epochs", "3"))
        batch_size = int(get_user_input("Batch size", "4"))
        learning_rate = float(get_user_input("Learning rate", "2e-4"))
        
        finetuner.train(tokenized_dataset, num_epochs=num_epochs, 
                       batch_size=batch_size, learning_rate=learning_rate)
        
        print(f"\n✅ Fine-tuning complete!")
        print(f"   Model saved to: {output_dir}")
    
    # Upload option
    print("\n📝 FINAL STEP: HuggingFace Upload (Optional)")
    upload = get_user_input("Upload? (yes/no)", "no").lower()
    
    if upload in ['yes', 'y']:
        repo_name = get_user_input("Repository name (user/repo)")
        create_new = get_user_input("Create new? (yes/no)", "yes").lower() in ['yes', 'y']
        private = get_user_input("Private? (yes/no)", "no").lower() in ['yes', 'y']
        token = get_user_input("HF token (or press Enter)", "")
        
        finetuner.upload_to_huggingface(repo_name, token if token else None, create_new, private)
    
    print("\n" + "="*70)
    print("🎉 Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)