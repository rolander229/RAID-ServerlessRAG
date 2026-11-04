#!/usr/bin/env bash
# 生成本地 wheelhouse（离线安装依赖，供 Docker 构建使用）
# 用法：bash scripts/download_wheels.sh   （在 Mac / 任意联网机器上运行）
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
WH="$DIR/wheelhouse"
PY="${PYTHON:-python3}"
mkdir -p "$WH"
echo "[1/2] torch/torchvision CPU 版（pytorch 官方 cpu 源）"
$PY -m pip download --only-binary=:all: --no-deps \
  --platform linux_x86_64 --python-version 3.10 --implementation cp --abi cp310 \
  -d "$WH" torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
echo "[2/2] 其余依赖（阿里云镜像）"
$PY -m pip download --only-binary=:all: \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform linux_x86_64 \
  --python-version 3.10 --implementation cp --abi cp310 \
  -d "$WH" filelock typing-extensions sympy==1.13.1 mpmath networkx jinja2 markupsafe \
  fsspec pillow numpy==2.2.6 PyYAML==6.0.2 tqdm==4.67.1 \
  --index-url https://mirrors.aliyun.com/pypi/simple/
echo "wheelhouse ready: $(ls "$WH" | wc -l) wheels at $WH"
