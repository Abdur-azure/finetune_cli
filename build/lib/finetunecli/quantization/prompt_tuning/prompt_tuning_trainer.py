"""
Prompt Tuning Trainer
Handles training with soft prompt embeddings
"""
from transformers import Trainer, TrainingArguments, AutoTokenizer
from finetunecli.utils.dataset_loader import load_json_dataset
from finetunecli.utils.logging import get_logger
from .prompt_tuning_model import build_prompt_tuning_model

log = get_logger("PromptTuning")

class PromptTuningTrainer:
    """Trainer for Prompt Tuning fine-tuning"""
    
    def __init__(self, cfg):
        """
        Initialize Prompt Tuning trainer
        
        Args:
            cfg: PromptTuningConfig object with training parameters
        """
        self.cfg = cfg
        log.info(f"Initializing Prompt Tuning trainer for model: {cfg.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Build Prompt Tuning model
        log.info(f"Building Prompt Tuning model with {cfg.num_virtual_tokens} virtual tokens")
        log.info(f"Initialization method: {cfg.prompt_tuning_init}")
        
        self.model = build_prompt_tuning_model(
            cfg.model_name,
            num_virtual_tokens=cfg.num_virtual_tokens,
            prompt_tuning_init=cfg.prompt_tuning_init,
            prompt_tuning_init_text=cfg.prompt_tuning_init_text,
            tokenizer_name_or_path=cfg.tokenizer_name_or_path or cfg.model_name
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
        
        # Training arguments optimized for Prompt Tuning
        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.lr,  # Higher LR for prompt tuning (3e-2 vs 2e-4)
            num_train_epochs=self.cfg.epochs,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False,  # Usually not needed for prompt tuning
            warmup_steps=100,  # Warmup helps with prompt tuning
            weight_decay=0.01,  # Small weight decay
            logging_dir=f"{self.cfg.output_dir}/logs",
        )
        
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=tokenized,
            args=args,
        )
        
        log.info("Starting Prompt Tuning fine-tuning...")
        log.info(f"  Virtual tokens: {self.cfg.num_virtual_tokens}")
        log.info(f"  Epochs: {self.cfg.epochs}")
        log.info(f"  Batch size: {self.cfg.batch_size}")
        log.info(f"  Learning rate: {self.cfg.lr}")
        log.info(f"  Initialization: {self.cfg.prompt_tuning_init}")
        
        trainer.train()
        
        # Save the model
        trainer.save_model(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        
        log.info(f"✅ Prompt Tuning model saved to {self.cfg.output_dir}")
        log.info(f"   Only {self.cfg.num_virtual_tokens} virtual token embeddings were trained!")
