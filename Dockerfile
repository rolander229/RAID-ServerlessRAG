# MOPAR slice-runner — 通用切片执行镜像（所有切片共用一个镜像）
# 构建（Mac ARM → linux/amd64）:
#   docker buildx build --platform linux/amd64 \
#     -t gpu-harbor.act.buaa.edu.cn/user-xuedh25/mopar-slice-runner:v0.1 --push .
FROM python:3.10-slim

WORKDIR /app

# CPU 精简版 torch（避免拉入 CUDA 全家桶；源不可达时可注释改用默认 PyPI）
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 其余依赖（剔除 torch/torchvision/triton，避免覆盖 CPU 版）
COPY requirements.txt .
RUN grep -vE "^(torch|torchvision|triton)" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

# 仓库代码（.dockerignore 已排除大文件与敏感内容）
COPY . .

EXPOSE 8080
# 实际参数由 K8s Deployment args 传入（--manifest/--slice/--weights/--next-peer）
CMD ["python", "-m", "capsule.slice_server", "--help"]
