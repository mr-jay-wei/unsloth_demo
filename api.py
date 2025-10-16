
# api.py
#
# 目的: 作为vLLM API服务的客户端，使用标准的OpenAI `chat`接口
#       来调用我们微调并部署好的大模型。
#
# 运行方式: python api.py (请确保vLLM服务器正在运行)

from openai import OpenAI
import os

# ==========================================================
# 1. 配置API客户端
# ==========================================================
# vLLM服务器的地址。如果是远程服务器，请替换'localhost'为服务器IP
VLLM_BASE_URL = "http://localhost:8000/v1" # "http://10.233.92.173:8000/v1" #
# vLLM不需要真实的API Key，但字段需要存在
VLLM_API_KEY = "not-used"
# 这个需要和你启动vLLM时 --model 参数指向的文件夹路径完全一致
# 使用绝对路径以避免混淆
MODEL_PATH = "/root/llama3-8b-lora-demo/merged_model" # <--- 注意更新为新文件夹名

try:
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
    )
    print("Successfully connected to vLLM server.")
except Exception as e:
    print(f"Error connecting to vLLM server: {e}")
    exit()

# ==========================================================
# 2. 准备用户输入 (标准的Chat API格式)
# ==========================================================
# 用户输入就像和ChatGPT聊天一样，是一个简单的字符串
user_input = "请介绍一下什么是文艺复兴，以及它的三位代表人物。"

# 在应用层，我们将用户输入封装成`messages`列表
messages = [
    {"role": "user", "content": user_input}
]
print(f"\nUser >>> {user_input}")


# ==========================================================
# 3. 调用API并流式处理输出
# ==========================================================
print("\nAssistant >>> ", end="")
try:
    # 使用行业标准的 `client.chat.completions.create`
    stream = client.chat.completions.create(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=5000,
        stream=True, # 开启流式输出
    )

    # 逐块打印，实现打字机效果
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

except Exception as e:
    print(f"\n\nAn error occurred during API call: {e}")

# 换行结束
print()
