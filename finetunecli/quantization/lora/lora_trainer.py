from transformers import Trainer, TrainingArguments, AutoTokenizer
from finetunecli.utils.dataset_loader import load_json_dataset
from finetunecli.utils.logging import get_logger
from .lora_model import build_lora_model

log = get_logger("LoRA")

class LoraTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = build_lora_model(cfg.model_name, cfg.r, cfg.alpha, cfg.dropout)

    def train(self):
        dataset = load_json_dataset(self.cfg.dataset_path)

        def tokenize(batch):
            return self.tokenizer(batch["input"], truncation=True)

        tokenized = dataset.map(tokenize)

        args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.lr,
            num_train_epochs=self.cfg.epochs,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=tokenized,
            args=args,
        )

        log.info("Starting LoRA fine-tuning...")
        trainer.train()
        trainer.save_model(self.cfg.output_dir)
        log.info(f"Model saved to {self.cfg.output_dir}")
