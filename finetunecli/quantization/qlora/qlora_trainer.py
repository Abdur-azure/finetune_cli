"""
QLoRA Trainer
Handles training with 4-bit quantized models and LoRA adapters
"""
from transformers import Trainer, TrainingArguments, AutoTokenizer
from finetunecli.utils.dataset_loader import load_json_dataset
from finetunecli.utils.logging import get_logger
from .qlora_model import build_qlora_model

log = get_logger("QLoRA")

class QLoraTrainer:
    """Trainer for QLoRA (Quantized LoRA) fine-tuning"""
    
    def __init__(self, cfg):
        """
        Initialize QLoRA trainer
        
        Args:
            cfg: QLoraConfig object with training parameters
        """
        self.cfg = cfg
        log.info(f"Initializing QLoRA trainer for model: {cfg.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Build QLoRA model with quantization
        log.info(f"Building QLoRA model with {cfg.bits}-bit quantization ({cfg.quant_type})")
        self.model = build_qlora_model(
            cfg.model_name,
            cfg.r,
            cfg.alpha,
            cfg.dropout,
            bits=cfg.bits,
            quant_type=cfg.quant_type,
            use_double_quant=cfg.use_double_quant
        )
        
    def train(self):
        """Execute the training loop"""
        log.info(f"Loading dataset from: {self.cfg.dataset_path}")
        dataset = load_json_dataset(self.cfg.dataset_path)
        
        def tokenize(batch):
            """Tokenize batch of data"""
            return self.tokenizer(
                batch["input"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        log.info("Tokenizing dataset...")
        tokenized = dataset.map(tokenize, batched=True)
        
        # Training arguments optimized for QLoRA
        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
            learning_rate=self.cfg.lr,
            num_train_epochs=self.cfg.epochs,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,  # Use mixed precision
            optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
            gradient_checkpointing=True,  # Save memory
            max_grad_norm=0.3,  # Gradient clipping
            warmup_ratio=0.03,  # Warmup steps
            lr_scheduler_type="cosine"  # Cosine learning rate schedule
        )
        
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=tokenized,
            args=args,
        )
        
        log.info("Starting QLoRA fine-tuning...")
        log.info(f"  Epochs: {self.cfg.epochs}")
        log.info(f"  Batch size: {self.cfg.batch_size}")
        log.info(f"  Learning rate: {self.cfg.lr}")
        log.info(f"  Quantization: {self.cfg.bits}-bit {self.cfg.quant_type}")
        
        trainer.train()
        
        # Save the model
        trainer.save_model(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        
        log.info(f"✅ QLoRA model saved to {self.cfg.output_dir}")
