#!/usr/bin/env bash
# ==============================================================================
# 将已下载的模型权重同步/导出到公共共享目录: /storage/xuedh_work2/models
# 方便团队其他用户 (如 duxc, wangyh 等) 共享使用，避免重复占用磁盘和带宽下载。
# ==============================================================================

set -e

SHARED_DIR="/storage/xuedh_work2/models"
LOCAL_HF_HUB="/storage/xuedh_workspace/.cache/huggingface/hub"

echo "=== MOPAR 多用户共享模型同步工具 ==="
echo "目标共享目录: ${SHARED_DIR}"

# 1. 检查目标共享目录是否存在与写权限
if [ ! -d "${SHARED_DIR}" ]; then
    echo "错误: 目录 ${SHARED_DIR} 不存在，请先执行: sudo mkdir -p ${SHARED_DIR}"
    exit 1
fi

if [ ! -w "${SHARED_DIR}" ]; then
    echo "警告: 当前用户对 ${SHARED_DIR} 无写入权限！"
    echo "请在终端中执行以下提权命令以开放写权限:"
    echo "    sudo chmod -R 777 ${SHARED_DIR}"
    echo "或者:"
    echo "    sudo chown -R $(whoami):$(whoami) ${SHARED_DIR}"
    exit 1
fi

# 2. 同步 ChatGLM3-6B
CHATGLM_SRC="${LOCAL_HF_HUB}/models--THUDM--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac"
CHATGLM_DST="${SHARED_DIR}/chatglm3-6b"

if [ -d "${CHATGLM_SRC}" ]; then
    echo "--> 正在同步 ChatGLM3-6B 真实权重到 ${CHATGLM_DST} ..."
    mkdir -p "${CHATGLM_DST}"
    cp -r -L "${CHATGLM_SRC}"/* "${CHATGLM_DST}/"
    echo "    ChatGLM3-6B 同步完成 ($(du -sh "${CHATGLM_DST}" | cut -f1))"
else
    echo "跳过 ChatGLM3-6B: 本地缓存源不存在 (${CHATGLM_SRC})"
fi

# 3. 同步 Qwen3-1.7B
QWEN_SRC="${LOCAL_HF_HUB}/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
QWEN_DST="${SHARED_DIR}/qwen3-1.7b"

if [ -d "${QWEN_SRC}" ]; then
    echo "--> 正在同步 Qwen3-1.7B 真实权重到 ${QWEN_DST} ..."
    mkdir -p "${QWEN_DST}"
    cp -r -L "${QWEN_SRC}"/* "${QWEN_DST}/"
    echo "    Qwen3-1.7B 同步完成 ($(du -sh "${QWEN_DST}" | cut -f1))"
else
    echo "跳过 Qwen3-1.7B: 本地缓存源不存在 (${QWEN_SRC})"
fi

# 4. 同步常用视觉模型 (VGG-16, ResNet-50, Inception-V3, ConvNeXt)
VISION_SRC="/storage/xuedh_workspace/mopar-impl/weights/hub"
VISION_DST="${SHARED_DIR}/hub"

if [ -d "${VISION_SRC}" ]; then
    echo "--> 正在同步 TorchVision 小模型权重到 ${VISION_DST} ..."
    mkdir -p "${VISION_DST}"
    cp -r "${VISION_SRC}"/* "${VISION_DST}/"
    echo "    TorchVision 视觉模型同步完成 ($(du -sh "${VISION_DST}" | cut -f1))"
fi

# 5. 确保所有其他用户具备只读访问权限
echo "--> 配置全局共享可读权限 (chmod -R a+rX) ..."
chmod -R a+rX "${SHARED_DIR}"/*

echo ""
echo "=== 全部模型已成功同步至公共共享目录！==="
echo "现在任何用户（无论是您自己还是团队其他成员）都可以通过以下方式直接加载："
echo "  1. Python Transformers 直接传本地路径:"
echo "     AutoModel.from_pretrained(\"${SHARED_DIR}/chatglm3-6b\", trust_remote_code=True)"
echo "     AutoModelForCausalLM.from_pretrained(\"${SHARED_DIR}/qwen3-1.7b\")"
echo "  2. 或在环境中设置公共缓存目录:"
echo "     export HF_HOME=${SHARED_DIR}"
