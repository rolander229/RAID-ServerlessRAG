"""Generate a clean system architecture diagram for the MOPAR + Capsule pipeline.

Output: docs/images/system_architecture.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for fp in ("/home/xuedh/.local/share/fonts/wqy-microhei.ttc",
           "/home/xuedh/.local/share/fonts/wqy-zenhei.ttc"):
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        break
plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images" / "system_architecture.png"

STAGES = [
    {
        "title": "输入：单体智能算法模型",
        "color": "#F0F4F8",
        "edge": "#4A5568",
        "lines": ["ResNet-50 / VGG-16 / Inception-V3 / ConvNeXt-T / Qwen3-1.7B (LLM)"],
    },
    {
        "title": "MOPAR 自动微切分",
        "color": "#E6FFFA",
        "edge": "#2C7A7B",
        "lines": [
            "运行时记录 → 计算图构建 → HyPAD 动态规划垂直切分",
            "输出：切片方案（函数片段 + 显存阶梯配额）",
            "数值等价验证：切分链 vs 单体，max_tensor_diff = 0",
        ],
    },
    {
        "title": "Capsule 云原生函数封装",
        "color": "#EBF8FF",
        "edge": "#2B6CB0",
        "lines": [
            "标准化 FaaS Handler 自动合成（事件解析 / 结构化响应 / 异常捕获）",
            "同步 / 异步触发解耦  ·  SAM IaC 模板  ·  差分测试载荷",
        ],
    },
    {
        "title": "云原生部署形态",
        "color": "#F7FAFC",
        "edge": "#718096",
        "lines": [
            "API Gateway (POST /predict) → Slice_0 主函数 → 异步调用 → Slice_1…N Worker",
            "S3 激活存储（跨函数张量传递）",
        ],
    },
    {
        "title": "六维封装准确率评测",
        "color": "#FFFAF0",
        "edge": "#C05621",
        "lines": [
            "接口契约 · 划分边界 · 模板规范 · 触发解耦 · IaC 基础设施 · 运行时差分",
            "独立专家基线 + 真实差分执行 → 5 模型实测平均 88.76%（全部 ≥70% 达标）",
        ],
    },
]

fig, ax = plt.subplots(figsize=(11.5, 9.2), dpi=160)
box_w, box_h, gap = 8.8, 1.62, 0.52
ax.set_xlim(0, 10)
ax.set_ylim(0, len(STAGES) * (box_h + gap) + 0.4)
ax.axis("off")

y = len(STAGES) * (box_h + gap) - box_h
centers = []
for stage in STAGES:
    x0 = (10 - box_w) / 2
    patch = FancyBboxPatch((x0, y), box_w, box_h,
                           boxstyle="round,pad=0.06,rounding_size=0.12",
                           facecolor=stage["color"], edgecolor=stage["edge"], linewidth=1.6)
    ax.add_patch(patch)
    ax.text(5, y + box_h - 0.32, stage["title"], ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1A202C")
    n = len(stage["lines"])
    line_h = 0.30
    start_y = y + box_h - 0.74
    for i, line in enumerate(stage["lines"]):
        ax.text(5, start_y - i * line_h, line, ha="center", va="center",
                fontsize=9.6, color="#2D3748")
    centers.append((y, y + box_h))
    y -= box_h + gap

for (top_box, bottom_box) in zip(centers[:-1], centers[1:]):
    ax.add_patch(FancyArrowPatch((5, top_box[0] - 0.04), (5, bottom_box[1] + 0.04),
                                 arrowstyle="-|>", mutation_scale=22,
                                 linewidth=2.2, color="#4A5568"))

ax.text(5, 0.16, "MOPAR（模型微切分）+ Capsule（云原生服务化）端到端流水线",
        ha="center", va="center", fontsize=11, color="#718096")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"written: {OUT}")
