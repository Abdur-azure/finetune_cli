import os
import json
from typing import Optional, Dict, List, Union
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
            # MLP layers (optional, for more comprehensive tuning)
            ["fc1", "fc2"],
            ["up_proj", "down_proj", "gate_proj"],
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
        """Configure and apply LoRA to the model"""
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
