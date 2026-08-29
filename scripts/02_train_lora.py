import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# ================= Config =================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "./data/gsm8k_processed/train"
OUTPUT_DIR = "./checkpoints/qwen-1.5b-gsm8k-lora"
MAX_SEQ_LENGTH = 1024

# ================= Load Model =================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,               # bf16 এর বদলে float16
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ================= Dataset =================
dataset = load_from_disk(DATA_PATH)

def formatting_func(example):
    text = example["prompt"] + example["completion"] + tokenizer.eos_token
    return text

# ================= Training Arguments =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,                         # এখন True
    bf16=False,                        # এখন False
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=50,
    report_to="none",
    remove_unused_columns=False,
)

# ================= Trainer =================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    formatting_func=formatting_func,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training complete! Model saved to:", OUTPUT_DIR)