# MOPAR slice-runner — 通用切片执行镜像（所有切片共用一个镜像）
# 构建（Mac ARM → linux/amd64）:
#   docker buildx build --platform linux/amd64 \
#     -t gpu-harbor.act.buaa.edu.cn/user-xuedh25/mopar-slice-runner:v0.1 --push .
FROM python:3.10-slim

WORKDIR /app

# CPU torch：直装官方 wheel URL（绕开 pytorch index 的元数据解析问题）
RUN pip install --no-cache-dir \
    https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp310-cp310-linux_x86_64.whl

# 其余依赖走默认 PyPI（torchvision 与 +cpu torch 版本兼容；剔除 torch/triton 行）
COPY requirements.txt .
RUN grep -vE "^(torch|torchvision|triton)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir torchvision==0.20.1 -r /tmp/req.txt

COPY . .

EXPOSE 8080
CMD ["python", "-m", "capsule.slice_server", "--help"]
