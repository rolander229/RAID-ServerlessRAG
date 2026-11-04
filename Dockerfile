# MOPAR slice-runner — 通用切片执行镜像（单镜像复用全部切片）
# 构建（零出网，依赖来自本地 wheelhouse）:
#   docker buildx build --platform linux/amd64 -t gpu-harbor.act.buaa.edu.cn/user-xuedh25/mopar-slice-runner:v0.3 --push .
FROM docker.m.daocloud.io/library/python:3.10-slim
WORKDIR /app
COPY wheelhouse /wheelhouse
RUN pip install --no-cache-dir --no-index --find-links /wheelhouse \
      torch==2.5.1 torchvision==0.20.1 numpy==2.2.6 PyYAML==6.0.2 tqdm==4.67.1
COPY . .
EXPOSE 8080
CMD ["python", "-m", "capsule.slice_server", "--help"]
