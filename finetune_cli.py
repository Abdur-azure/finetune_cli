#!/usr/bin/env python3
"""
LLM Fine-Tuning CLI Tool - Extended Edition with Distillation
Supports: LoRA, QLoRA, AdaLoRA, Vanilla Distillation, Feature Distillation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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


class DistillationTrainer(Trainer):
    """Custom Trainer for Knowledge Distillation"""
    
    def __init__(self, teacher_model=None, distillation_type="vanilla", 
                 temperature=2.0, alpha=0.5, feature_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.temperature = temperature
        self.alpha = alpha
        self.feature_layers = feature_layers or []
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute distillation loss"""
        
        # Get student outputs
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Standard CE loss
        loss = outputs.loss
        
        # Add distillation loss if teacher is available
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            if self.distillation_type == "vanilla":
                # Vanilla Distillation: KL divergence on output logits
                distill_loss = self._compute_kl_loss(
                    student_logits, teacher_logits, self.temperature
                )
                
            elif self.distillation_type == "feature":
                # Feature Distillation: MSE on intermediate representations
                distill_loss = self._compute_feature_loss(
                    model, self.teacher_model, inputs
                )
            
            # Combined loss: α * CE + (1-α) * Distillation
            loss = self.alpha * loss + (1 - self.alpha) * distill_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _compute_kl_loss(self, student_logits, teacher_logits, temperature):
        """Compute KL divergence loss for vanilla distillation"""
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return kl_loss
    
    def _compute_feature_loss(self, student_model, teacher_model, inputs):
        """Compute MSE loss on intermediate features"""
        
        # Get hidden states from both models
        student_outputs = student_model(**inputs, output_hidden_states=True)
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=True)
        
        student_hiddens = student_outputs.hidden_states
        teacher_hiddens = teacher_outputs.hidden_states
        
        # Compute MSE on selected layers
        feature_loss = 0.0
        num_layers = len(self.feature_layers) if self.feature_layers else len(student_hiddens)
        
        if not self.feature_layers:
            # Use all layers
            layers_to_match = range(len(student_hiddens))
        else:
            layers_to_match = self.feature_layers
        
        for layer_idx in layers_to_match:
            if layer_idx < len(student_hiddens) and layer_idx < len(teacher_hiddens):
                student_feat = student_hiddens[layer_idx]
                teacher_feat = teacher_hiddens[layer_idx]
                
                # Handle dimension mismatch with projection
                if student_feat.shape[-1] != teacher_feat.shape[-1]:
                    # Simple linear projection (could be learned)
                    teacher_feat = teacher_feat[..., :student_feat.shape[-1]]
                
                feature_loss += F.mse_loss(student_feat, teacher_feat)
        
        return feature_loss / num_layers


class LLMFineTuner:
    def __init__(self, model_name: str, output_dir: str = "./finetuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.teacher_model = None
        self.method = None
        
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
        
        # Configure quantization
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
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs['quantization_config'] = bnb_config
                model_kwargs['device_map'] = "auto"
        else:
            if self.device == "cuda":
                model_kwargs['torch_dtype'] = torch.float16
                model_kwargs['device_map'] = "auto"
        
        model_kwargs['low_cpu_mem_usage'] = True
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if method == "qlora" or load_in_4bit or load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
            print("✅ Model prepared for quantized training")
        
        print("✅ Model loaded successfully\n")
    
    def load_teacher_model(self, teacher_model_name: str):
        """Load teacher model for distillation"""
        print(f"\n🎓 Loading teacher model: {teacher_model_name}")
        
        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs['torch_dtype'] = torch.float16
            model_kwargs['device_map'] = "auto"
        
        model_kwargs['low_cpu_mem_usage'] = True
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            **model_kwargs
        )
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        print("✅ Teacher model loaded and frozen\n")
        
        # Log parameter comparison
        student_params = sum(p.numel() for p in self.model.parameters())
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        compression_ratio = teacher_params / student_params
        
        print(f"📊 Model Comparison:")
        print(f"   Student: {student_params:,} parameters")
        print(f"   Teacher: {teacher_params:,} parameters")
        print(f"   Compression: {compression_ratio:.2f}x\n")
    
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
        common_names = ['text', 'content', 'input', 'prompt', 'instruction', 'question', 'answer', 'output', 'response']
        
        for col in dataset.column_names:
            if col.lower() in common_names:
                text_columns.append(col)
            elif len(dataset) > 0 and isinstance(dataset[0][col], str):
                text_columns.append(col)
        
        return text_columns
    
    def prepare_dataset(self, dataset: Dataset, text_columns: Optional[List[str]] = None, max_length: int = 512):
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
        
        linear_modules = [name for name in module_names if 'lin' in name.lower() or 'proj' in name.lower() or 'fc' in name.lower()]
        if linear_modules:
            print(f"✅ Using detected linear modules: {linear_modules[:4]}")
            return linear_modules[:4]
        
        print("⚠️  Could not auto-detect. Using 'all-linear' fallback.")
        return "all-linear"
    
    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1,
                   target_modules: Optional[List[str]] = None):
        """Configure and apply LoRA"""
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
        """Print memory savings estimate"""
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
    
    def setup_vanilla_distillation(self, temperature: float = 2.0, alpha: float = 0.5):
        """Configure Vanilla (Output) Distillation"""
        print(f"\n🎯 Setting up Vanilla Distillation...")
        print(f"   - Temperature: {temperature}")
        print(f"   - Alpha (CE weight): {alpha}")
        print(f"   - Type: Output logits distillation")
        
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call load_teacher_model() first.")
        
        self.distillation_config = {
            'type': 'vanilla',
            'temperature': temperature,
            'alpha': alpha
        }
        
        print("✅ Vanilla Distillation configured\n")
        print("💡 This method transfers output probability distributions")
    
    def setup_feature_distillation(self, temperature: float = 2.0, alpha: float = 0.3, 
                                   feature_layers: Optional[List[int]] = None):
        """Configure Feature (Intermediate Layer) Distillation"""
        print(f"\n🎯 Setting up Feature Distillation...")
        print(f"   - Temperature: {temperature}")
        print(f"   - Alpha (CE weight): {alpha}")
        print(f"   - Type: Intermediate representations")
        
        if self.teacher_model is None:
            raise ValueError("Teacher model not loaded. Call load_teacher_model() first.")
        
        # Auto-select layers if not specified
        if feature_layers is None:
            num_layers = len(self.model.base_model.transformer.h) if hasattr(self.model, 'base_model') else 12
            # Select evenly spaced layers
            feature_layers = [i for i in range(0, num_layers, max(1, num_layers // 4))]
        
        self.distillation_config = {
            'type': 'feature',
            'temperature': temperature,
            'alpha': alpha,
            'feature_layers': feature_layers
        }
        
        print(f"   - Selected layers: {feature_layers}")
        print("✅ Feature Distillation configured\n")
        print("💡 This method transfers intermediate layer representations")
    
    def train(self, train_dataset: Dataset, num_epochs: int = 3, 
              batch_size: int = 4, learning_rate: float = 2e-4,
              gradient_accumulation_steps: int = 4):
        """Train the model"""
        print(f"\n🚀 Starting training with {self.method.upper()}...")
        print(f"   - Epochs: {num_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Learning rate: {learning_rate}\n")
        
        use_fp16 = self.device == "cuda" and self.method not in ["qlora"]
        
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
        
        # Use distillation trainer if method is distillation
        if self.method in ['vanilla_distillation', 'feature_distillation']:
            trainer = DistillationTrainer(
                teacher_model=self.teacher_model,
                distillation_type=self.distillation_config['type'],
                temperature=self.distillation_config['temperature'],
                alpha=self.distillation_config['alpha'],
                feature_layers=self.distillation_config.get('feature_layers'),
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator
            )
        else:
            trainer = Trainer(
                model=self.peft_model if self.peft_model else self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator
            )
        
        trainer.train()
        
        # Save model
        if self.peft_model:
            self.peft_model.save_pretrained(self.output_dir)
        else:
            self.model.save_pretrained(self.output_dir)
        
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✅ Training complete! Model saved to: {self.output_dir}\n")
    
    def benchmark(self, test_prompts: List[str], use_finetuned: bool = False) -> Dict:
        """Benchmark model performance"""
        print(f"\n📊 Running benchmark {'(Fine-tuned)' if use_finetuned else '(Base model)'}...")
        
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
        print(f"ROUGE Scores {'(Fine-tuned)' if use_finetuned else '(Base)'}:")
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
                print("❌ Not logged in. Provide token.")
                return
        
        api = HfApi()
        
        if create_new:
            try:
                create_repo(repo_name, private=private, exist_ok=True)
                print(f"✅ Repository verified: {repo_name}")
            except Exception as e:
                print(f"❌ Error: {e}")
                return
        
        try:
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            print(f"✅ Uploaded: https://huggingface.co/{repo_name}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def print_method_info():
    """Print method information"""
    print("\n" + "="*70)
    print("📚 AVAILABLE METHODS")
    print("="*70)
    
    print("\n🔧 PARAMETER-EFFICIENT FINE-TUNING:")
    print("1️⃣  LoRA - Low-Rank Adaptation")
    print("2️⃣  QLoRA - Quantized LoRA")
    print("3️⃣  AdaLoRA - Adaptive LoRA")
    
    print("\n🎓 KNOWLEDGE DISTILLATION:")
    print("4️⃣  Vanilla Distillation - Output logits transfer")
    print("5️⃣  Feature Distillation - Intermediate layer transfer")
    
    print("\n" + "="*70 + "\n")


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with default"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()


def main():
    print("="*70)
    print("🤖 LLM Fine-Tuning CLI - Extended Edition")
    print("   Fine-Tuning + Knowledge Distillation")
    print("="*70)
    
    # Method selection
    print("\n📝 STEP 1: Select Method")
    print_method_info()
    
    print("Select:")
    print("  1. LoRA")
    print("  2. QLoRA")
    print("  3. AdaLoRA")
    print("  4. Vanilla Distillation")
    print("  5. Feature Distillation")
    print("  6. Show detailed info")
    
    method_choice = get_user_input("\nSelect [1-6]", "1")
    
    if method_choice == "6":
        print_method_info()
        method_choice = get_user_input("Select [1-5]", "1")
    
    method_map = {
        "1": "lora",
        "2": "qlora",
        "3": "adalora",
        "4": "vanilla_distillation",
        "5": "feature_distillation"
    }
    
    selected_method = method_map.get(method_choice, "lora")
    print(f"\n✅ Selected: {selected_method.upper()}")
    
    # Model configuration
    print("\n📝 STEP 2: Model Configuration")
    model_name = get_user_input("Student model name", "gpt2")
    output_dir = get_user_input("Output directory", "./finetuned_model")
    
    finetuner = LLMFineTuner(model_name, output_dir)
    
    # Load student model
    if selected_method == "qlora":
        use_8bit = get_user_input("Use 8-bit? (yes/no)", "no").lower() == "yes"
        finetuner.load_model(method=selected_method, load_in_4bit=not use_8bit, load_in_8bit=use_8bit)
    else:
        finetuner.load_model(method=selected_method)
    
    # Load teacher for distillation
    if selected_method in ['vanilla_distillation', 'feature_distillation']:
        print("\n💡 Distillation requires a teacher model (larger/better)")
        teacher_name = get_user_input("Teacher model name", "gpt2-medium")
        finetuner.load_teacher_model(teacher_name)
    
    # Dataset loading
    print("\n📝 STEP 3: Dataset")
    dataset_source = get_user_input("Dataset path or HF name")
    
    is_local = os.path.exists(dataset_source)
    dataset_config = None
    split = "train"
    data_files = None
    num_samples = None
    
    if not is_local:
        dataset_config = get_user_input("Config (optional)", "") or None
        use_specific = get_user_input("Load specific file? (yes/no)", "no").lower()
        if use_specific in ['yes', 'y']:
            data_files = get_user_input("File path/pattern")
        split = get_user_input("Split", "train")
    
    limit = get_user_input("Limit samples? (yes/no)", "yes").lower()
    if limit in ['yes', 'y']:
        num_samples = int(get_user_input("Number of samples", "1000"))
    
    dataset = finetuner.load_dataset_from_source(
        dataset_source, dataset_config, split, num_samples, data_files
    )
    
    max_length = int(get_user_input("Max sequence length", "512"))
    tokenized_dataset, text_columns = finetuner.prepare_dataset(dataset, max_length=max_length)
    
    # Pre-training benchmark
    print("\n📝 STEP 4: Pre-training Benchmark")
    run_benchmark = get_user_input("Run benchmark? (yes/no)", "yes").lower()
    
    base_scores = None
    if run_benchmark in ['yes', 'y']:
        num_test = min(10, len(dataset))
        test_prompts = []
        for i in range(num_test):
            if len(text_columns) == 1:
                prompt = str(dataset[i][text_columns[0]])[:50]
            else:
                prompt = " ".join([str(dataset[i][col])[:30] for col in text_columns])[:50]
            test_prompts.append(prompt)
        
        base_scores = finetuner.benchmark(test_prompts, use_finetuned=False)
    
    # Method configuration
    print(f"\n📝 STEP 5: {selected_method.upper()} Configuration")
    
    if selected_method == "lora":
        r = int(get_user_input("LoRA r", "8"))
        alpha = int(get_user_input("LoRA alpha", "32"))
        dropout = float(get_user_input("LoRA dropout", "0.1"))
        finetuner.setup_lora(r, alpha, dropout)
    
    elif selected_method == "qlora":
        r = int(get_user_input("QLoRA r", "16"))
        alpha = int(get_user_input("QLoRA alpha", "64"))
        dropout = float(get_user_input("QLoRA dropout", "0.1"))
        finetuner.setup_qlora(r, alpha, dropout)
    
    elif selected_method == "adalora":
        target_r = int(get_user_input("Target rank", "8"))
        init_r = int(get_user_input("Initial rank", "12"))
        alpha = int(get_user_input("Alpha", "32"))
        dropout = float(get_user_input("Dropout", "0.1"))
        finetuner.setup_adalora(target_r, init_r, alpha, dropout)
    
    elif selected_method == "vanilla_distillation":
        temp = float(get_user_input("Temperature", "2.0"))
        alpha = float(get_user_input("Alpha (CE weight)", "0.5"))
        finetuner.setup_vanilla_distillation(temp, alpha)
    
    elif selected_method == "feature_distillation":
        temp = float(get_user_input("Temperature", "2.0"))
        alpha = float(get_user_input("Alpha (CE weight)", "0.3"))
        finetuner.setup_feature_distillation(temp, alpha)
    
    # Training
    print("\n📝 STEP 6: Training")
    epochs = int(get_user_input("Epochs", "3"))
    batch_size = int(get_user_input("Batch size", "4"))
    lr = float(get_user_input("Learning rate", "2e-4"))
    grad_accum = int(get_user_input("Gradient accumulation", "4"))
    
    finetuner.train(tokenized_dataset, epochs, batch_size, lr, grad_accum)
    
    # Post-training benchmark
    if run_benchmark in ['yes', 'y']:
        print("\n📝 STEP 7: Post-training Benchmark")
        finetuned_scores = finetuner.benchmark(test_prompts, use_finetuned=True)
        
        print("\n" + "="*70)
        print("📊 PERFORMANCE COMPARISON")
        print("="*70)
        print(f"{'Metric':<12} {'Base':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-"*70)
        for metric in base_scores:
            base = base_scores[metric]
            ft = finetuned_scores[metric]
            imp = ((ft - base) / base * 100) if base > 0 else 0
            print(f"{metric.upper():<12} {base:<15.4f} {ft:<15.4f} {imp:+.2f}%")
        print("="*70 + "\n")
    
    # Upload
    print("\n📝 STEP 8: Upload (Optional)")
    upload = get_user_input("Upload to HuggingFace? (yes/no)", "no").lower()
    
    if upload in ['yes', 'y']:
        repo_name = get_user_input("Repo name (user/repo)")
        create_new = get_user_input("Create new? (yes/no)", "yes").lower() in ['yes', 'y']
        private = get_user_input("Private? (yes/no)", "no").lower() in ['yes', 'y']
        token = get_user_input("Token (or Enter if logged in)", "") or None
        
        finetuner.upload_to_huggingface(repo_name, token, create_new, private)
    
    print("\n" + "="*70)
    print(f"🎉 Complete! Method: {selected_method.upper()}")
    print("="*70)
    print(f"✅ Model saved: {output_dir}\n")


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