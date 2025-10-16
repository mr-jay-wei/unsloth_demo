from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# -----------------------------
# 1. 加载基础模型（本地 Gemma3）
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./gemma-3-270m-it",   # ✅ 本地基础模型路径
    max_seq_length = 2048,
    load_in_4bit = False,               # 如果内存吃紧可以设为 True
)

# -----------------------------
# 2. 加载 LoRA 适配器
# -----------------------------
# ✅ 注意：不要放在 from_pretrained() 里！
model.load_adapter("./gemma-3-finetune")  # 本地微调后的 LoRA 文件夹

# -----------------------------
# 3. 定义输入（Gemma-3 使用 messages 格式）
# -----------------------------
messages = [
    {"role": "user", "content": "请给我讲一个关于国际象棋的有趣事实。"}
]

# 使用 chat_template 构造模型输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
).removeprefix("<bos>")

# -----------------------------
# 4. 生成回复
# -----------------------------
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# TextStreamer 实时输出
streamer = TextStreamer(tokenizer, skip_prompt=True)

with torch.no_grad():
    _ = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.9,
        top_p=0.95,
        top_k=64,
        streamer=streamer,
    )
