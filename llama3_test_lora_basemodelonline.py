# test.py

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

# 定义超参数 (保持一致)
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 1. 一步到位加载模型！
#    直接将包含LoRA适配器的文件夹路径作为 model_name。
#    Unsloth会自动加载基础模型，然后应用LoRA权重。
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",  # <--- 你的微调产出文件夹
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. 立即为推理进行优化
FastLanguageModel.for_inference(model)

# 3. 准备提示词 (和训练时保持一致)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 4. 准备输入
inputs = tokenizer(
[
    alpaca_prompt.format(
        "介绍文艺复兴",  # instruction
        "",              # input
        "",              # output - 留空让模型生成!
    )
], return_tensors = "pt").to("cuda")

# 5. 设置流式输出并开始生成
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 5000)