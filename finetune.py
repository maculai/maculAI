"""
macul.ai — LLM Fine-tuning Script
Fine-tunes LLaMA 3 8B on ophthalmology clinical notes using LoRA.

Requirements:
- RTX 4070 (12GB VRAM)
- 64GB RAM
- CUDA 12.1+
- pip install transformers peft datasets accelerate bitsandbytes

Usage:
  python finetune.py --data ./data/ophtho_notes.jsonl --output ./weights/llama3-ophtho
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)

TRAINING_ARGS = TrainingArguments(
    output_dir="./weights/llama3-ophtho",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=100,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none"
)


def format_prompt(example: dict) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert ophthalmology AI assistant analyzing clinical chart notes.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{example['input']}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example['output']}
<|eot_id|>"""


def main(args):
    print(f"[macul.ai] Loading base model: {BASE_MODEL}")
    print(f"[macul.ai] CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.data)

    def tokenize(example):
        text = format_prompt(example)
        return tokenizer(text, truncation=True, max_length=2048, padding="max_length")

    tokenized = dataset.map(tokenize, batched=False)

    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("[macul.ai] Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[macul.ai] Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data/ophtho_notes.jsonl")
    parser.add_argument("--output", default="./weights/llama3-ophtho")
    args = parser.parse_args()
    main(args)
