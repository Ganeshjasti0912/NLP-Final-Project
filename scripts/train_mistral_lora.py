import warnings
warnings.filterwarnings("ignore", message="Detected kernel version")

import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

print(" Loading model and tokenizer")
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

# Loading TravelPlanner dataset
train = load_dataset("osunlp/TravelPlanner", "train")["train"]
val   = load_dataset("osunlp/TravelPlanner", "validation")["validation"]

def build_prompt(ex):
    return f"<s>[SYSTEM] You are a travel planner. Generate a structured, day-by-day itinerary.\n[USER] {ex['query']}\n[ASSISTANT] {ex['reference_information']}</s>"

train = train.map(lambda e: {"text": build_prompt(e)})
val   = val.map(lambda e: {"text": build_prompt(e)})

# Tokenization
def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=1024, padding="max_length")

train_tok = train.map(tokenize, batched=True, remove_columns=train.column_names)
val_tok   = val.map(tokenize, batched=True, remove_columns=val.column_names)
train_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_cfg)

args = TrainingArguments(
    output_dir="models/mistral7b_lora_itinerary",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1.5e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=25,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
)

print(" Starting LoRA fine-tuning on Mistral-7B")
trainer.train()

model.save_pretrained("models/mistral7b_lora_itinerary")
tok.save_pretrained("models/mistral7b_lora_itinerary")

print("Mistral-7B LoRA fine-tuning complete and saved successfully.")
