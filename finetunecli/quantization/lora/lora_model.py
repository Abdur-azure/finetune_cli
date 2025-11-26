from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def build_lora_model(model_name, r, alpha, dropout):
    base = AutoModelForCausalLM.from_pretrained(model_name)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    return get_peft_model(base, cfg)
