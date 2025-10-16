from transformers import AutoModelForCausalLM
import torch
import re

MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"   # 改成你的模型名称

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

# === 第一步：列出所有模块名称 ===
module_names = [name for name, _ in model.named_modules()]

print(f"\n共发现 {len(module_names)} 个模块")

# === 第二步：自动识别模式 ===
# 思路：如果多个层的命名模式重复出现，说明那是“层结构”中的通用组件
# 比如 model.layers.0.self_attn.q_proj, model.layers.1.self_attn.q_proj ...
# 我们找出这些“末尾字段在不同层都出现过”的模块
suffix_count = {}
for name in module_names:
    suffix = name.split(".")[-1]
    suffix_count[suffix] = suffix_count.get(suffix, 0) + 1

# === 第三步：筛选出重复出现的子模块名（可能是每层都有的组件）===
repeated_suffixes = [k for k, v in suffix_count.items() if v > 5]  # 出现5次以上
print("\n📊 在多个层中重复出现的模块：")
print(repeated_suffixes)

# === 第四步：在这些候选层里进一步筛选出常见的线性层 ===
candidate_modules = []
for name, module in model.named_modules():
    if hasattr(module, "weight"):  # 有权重
        if any(name.endswith(suffix) for suffix in repeated_suffixes):
            if isinstance(module, torch.nn.Linear):  # 一般 LoRA 只作用在线性层上
                candidate_modules.append(name)

print("\n🎯 自动检测到的可微调模块候选：")
for name in candidate_modules[:50]:  # 打印前50个
    print(name)

# === 第五步：生成 target_modules 列表 ===
unique_targets = sorted(set(
    name.split(".")[-1] for name in candidate_modules
))
print("\n✅ 推荐的 target_modules =")
print(unique_targets)
