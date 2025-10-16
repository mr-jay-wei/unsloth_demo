"""
Fine-tuning Gemma-3 using Unsloth from a local directory.
Ensure the model folder (e.g. gemma-3-270m-it/) is in the same directory as this script.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 并行
os.environ["HF_DATASETS_PARALLELISM"] = "false" # 禁用 datasets 多进程

import os
import torch
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig

# ---------------------------
# 基础参数配置
# ---------------------------
LOCAL_MODEL_PATH = "./gemma-3-270m-it"   # 本地模型目录
OUTPUT_DIR = "outputs"
MAX_SEQ_LENGTH = 2048
RANK = 128
SEED = 3407
TRAIN_STEPS = 100
LEARNING_RATE = 5e-5

# ---------------------------
# 加载本地模型与 Tokenizer
# ---------------------------
print(f">>> Loading model from local path: {LOCAL_MODEL_PATH}")
model, tokenizer = FastModel.from_pretrained(
    model_name=LOCAL_MODEL_PATH,     # 这里改为本地路径
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

# 配置 LoRA 微调
print(">>> Applying LoRA configuration...")
model = FastModel.get_peft_model(
    model,
    r=RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

# 应用 Chat 模板
print(">>> Applying chat template...")
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# ---------------------------
# 加载数据集
# ---------------------------
print(">>> Loading dataset...")
dataset = load_dataset("Thytu/ChessInstruct", split="train[:10000]")

def convert_to_chatml(example):
    return {
        "conversations": [
            {"role": "system", "content": example["task"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["expected_output"]},
        ]
    }

dataset = dataset.map(convert_to_chatml)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)

# ---------------------------
# 配置 Trainer
# ---------------------------
print(">>> Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# ---------------------------
# 开始训练
# ---------------------------
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu.name}, {round(gpu.total_memory/1024**3,2)} GB total")
else:
    print("⚠️ No GPU detected, training will be slow")

print(">>> Start training...")
trainer_stats = trainer.train()

# ---------------------------
# 保存微调结果
# ---------------------------
print(">>> Saving fine-tuned model...")
SAVE_PATH = "gemma-3-finetune"
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ Fine-tuning complete. Model saved to: {SAVE_PATH}")
