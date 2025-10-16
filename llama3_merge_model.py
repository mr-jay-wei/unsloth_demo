# merge_model.py
#
# 目的: 将Unsloth LoRA适配器与基础模型合并，
#       并进行清理和配置，生成一个可以直接被vLLM部署的、
#       兼容Chat API的最终模型文件夹。
#
# 运行方式: python merge_model.py

import json
import os
import shutil
from unsloth import FastLanguageModel

# ==========================================================
# 1. 配置参数 (与训练时保持一致)
# ==========================================================
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 这是你训练好的LoRA适配器所在的文件夹
lora_model_dir = "lora_model"

# 这是我们最终要生成的、用于部署的完整模型文件夹
output_dir = "merged_model"


# ==========================================================
# 2. 加载LoRA模型
# ==========================================================
print(f"Loading LoRA model from: {lora_model_dir}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=lora_model_dir,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("LoRA model loaded successfully.")

# ==========================================================
# 3. 定义并应用自定义的Alpaca聊天模板
# ==========================================================
# 这个Jinja2模板能够精确地将OpenAI格式的`messages`列表，
# 转换成我们微调时使用的Alpaca格式的单个字符串。
alpaca_chat_template = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\\n\\n' }}"
            "{{ '### Instruction:\\n' + message['content'] + '\\n\\n' }}"
            "{{ '### Input:\\n' + '\\n\\n' }}"  # 假设input为空，如果你的应用需要处理input，这里需要修改
        "{% elif message['role'] == 'assistant' %}"
            "{{ '### Response:\\n' + message['content'] + eos_token + '\\n\\n' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '### Response:\\n' }}"
    "{% endif %}"
)

print("Applying custom Alpaca chat template...")
tokenizer.chat_template = alpaca_chat_template
print("Chat template applied.")


# ==========================================================
# 4. 合并LoRA权重
# ==========================================================
print(f"Merging LoRA weights and saving to: {output_dir}")
# 使用 `save_pretrained_merged` 将模型和tokenizer保存到输出目录
# 这会生成一个16位的高精度模型
model.save_pretrained_merged(output_dir, tokenizer, save_method="hf")
print("Model and tokenizer merged and saved.")


# ==========================================================
# 5. 修复配置文件以兼容vLLM
# ==========================================================
# 这一步至关重要，它解决了我们之前遇到的两个部署错误

# 5.1 复制基础模型的config.json，因为它包含了正确的模型架构信息
base_model_config_path = model.config.name_or_path
if os.path.isdir(base_model_config_path):
    # 通常Unsloth加载后，这个路径就是缓存中的路径
    config_file_path = os.path.join(base_model_config_path, "config.json")
    if os.path.exists(config_file_path):
        print(f"Copying config.json from base model at: {base_model_config_path}")
        shutil.copy(config_file_path, output_dir)

# 5.2 读取刚刚复制或保存的config.json，并移除不兼容的quantization_config
config_path = os.path.join(output_dir, "config.json")
if os.path.exists(config_path):
    print(f"Cleaning config.json at: {config_path}")
    with open(config_path, "r") as f:
        config_data = json.load(f)

    if "quantization_config" in config_data:
        del config_data["quantization_config"]
        print("Removed 'quantization_config' section.")

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print("Config file cleaned and saved.")
else:
    print(f"WARNING: config.json not found at {config_path}. vLLM deployment might fail.")

print("\n🚀 All steps completed successfully!")
print(f"Your final model for vLLM deployment is ready at: {output_dir}")