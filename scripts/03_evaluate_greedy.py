import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./checkpoints/qwen-1.5b-gsm8k-lora"
TEST_PATH = "./data/gsm8k_processed/test"

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

test_data = load_from_disk(TEST_PATH)

def extract_answer(text):
    match = re.search(r"####\s*(-?\d+)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else None

correct = 0
total = 0

for example in tqdm(test_data):
    prompt = example["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,          # Greedy
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = extract_answer(generated)
    gold = extract_answer(example["answer"])
    
    if pred == gold:
        correct += 1
    total += 1

accuracy = correct / total * 100
print(f"\nGreedy Accuracy: {accuracy:.2f}% ({correct}/{total})")