import openai
import json
import sys
import io

# 设置系统编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 配置客户端
client = openai.OpenAI(
    base_url="http://192.168.5.65:30100/v1",  # 使用NodePort
    api_key="sk-vllm-api-key-123456"
)

# 测试模型列表
models = client.models.list()
print("可用模型:", [model.id for model in models.data])

# 测试聊天完成
response = client.chat.completions.create(
    model="qwen3-1.7b",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手"},
        {"role": "user", "content": "请用中文写一个简短的自我介绍"}
    ],
    max_tokens=1024,
    temperature=0.7
)

print("回复:", response.choices[0].message.content)