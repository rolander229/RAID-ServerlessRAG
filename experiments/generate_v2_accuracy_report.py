"""Generate the v2 experimental evaluation report from real evaluation outputs.

Reads the per-model pipeline reports written by capsule.bridge
(build/benchmarks/slicer_<model>/slicer_capsule_evaluation_report.json) plus the
runtime evidence files, and renders docs/experimental_evaluation_report_v2.md.

No accuracy numbers are hardcoded in this script — everything comes from the
evaluation JSON produced by the fixed (uncalibrated) evaluator.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "build" / "benchmarks"
OUT_MD = REPO / "docs" / "experimental_evaluation_report_v2.md"

MODELS = [
    ("resnet50", "ResNet-50", "CNN", 3),
    ("vgg16", "VGG-16", "CNN", 2),
    ("inception_v3", "Inception-V3", "CNN", 3),
    ("convnext_tiny", "ConvNeXt-T", "CNN", 3),
    ("qwen3_1_7b", "Qwen3-1.7B", "LLM", 4),
]

DIM_ORDER = [
    ("API Interface Coverage & Mapping (S_api)", "S_api"),
    ("Function Decomposition Granularity (S_granularity)", "S_granularity"),
    ("Function Template & Signature Compliance (S_template)", "S_template"),
    ("Trigger & Async Decoupling Mechanism (S_trigger)", "S_trigger"),
    ("Cloud-Native Infrastructure & Layer Decoupling (S_infra)", "S_infra"),
    ("Runtime Execution & Semantic Output Equivalence (S_runtime)", "S_runtime"),
]


def main():
    rows = []
    for key, label, family, k in MODELS:
        report_file = BENCH / f"slicer_{key}" / "slicer_capsule_evaluation_report.json"
        if not report_file.exists():
            rows.append(None)
            continue
        data = json.loads(report_file.read_text(encoding="utf-8"))
        evidence_file = BENCH / f"slicer_{key}" / "tool_migrated" / "runtime_evidence.json"
        evidence = json.loads(evidence_file.read_text(encoding="utf-8")) if evidence_file.exists() else {}
        rows.append({"key": key, "label": label, "family": family, "k": k,
                     "data": data, "evidence": evidence})

    lines = []
    lines.append("# 实验评测报告 v2：修复评测器后的真实六维封装准确率")
    lines.append("")
    lines.append("**生成方式**：`experiments/generate_v2_accuracy_report.py` 直接读取修复后评测器输出的 JSON（`build/benchmarks/slicer_<model>/slicer_capsule_evaluation_report.json`），不包含任何硬编码分数。")
    lines.append("")
    lines.append("**与 v1 报告的区别**：v1 报告中的六维分数由 `_calibrate_dimension_realism` 以基准名哈希将 100% 压缩进预设区间合成（详见 v1 报告勘误一节）；v2 删除了该机制，所有分数为评测器的原始实测输出。")
    lines.append("")

    # ---- methodology ----
    lines.append("## 一、评测方法修复说明")
    lines.append("")
    lines.append("| # | 修复项 | 修复前 | 修复后 |")
    lines.append("|---|---|---|---|")
    lines.append("| 1 | 真实性校准 | 完美分数被哈希映射进预设区间（S_api 0.925–0.965 等） | **已删除**，报告原始分数 |")
    lines.append("| 2 | Ground truth | `ground_truth_expert/specification.json` 与工具输出为**同一文件**（自比较，循环论证） | 独立编写的专家基线（`evaluation/expert_baseline.py`，固定阶段规则 + 独立 API/触发/基础设施设计） |")
    lines.append("| 3 | 拆分粒度 | 函数**名**集合 Jaccard（名字对上即满分） | **划分边界**共属对 Jaccard（MoJoFM 风格，IEEE TSE '07）：两层同属一组且同属专家阶段才计一致 |")
    lines.append("| 4 | 触发解耦 | 按函数名逐个比对 trigger 字符串 | **类别级**比较（sync_http vs async），与命名无关 |")
    lines.append("| 5 | 运行时差分 | 单体响应为模拟构造；handler 抛异常时**伪装成功**；任意 2xx 互相匹配 | handler **真实执行**（异常即失败）；状态码精确匹配；叠加 MOPAR 链式切分 vs 单体的**真实数值等价证据**（max_tensor_diff ≤ 1e-4，各占 50%） |")
    lines.append("| 6 | 缺失产物 | handler 缺失/载荷缺失 → 默认满分 | 缺失产物计 0 分并标记 `NOT_EVALUATED`，同时给出全量纲与已评估量纲两个总分 |")
    lines.append("")

    # ---- main accuracy table ----
    lines.append("## 二、六维封装准确率（修复后实测）")
    lines.append("")
    header = "| 模型 | 类型 | 切片数 K | " + " | ".join(sym for _, sym in DIM_ORDER) + " | 总分 | 判定（≥70%） |"
    lines.append(header)
    lines.append("|" + ":---:|" * (len(DIM_ORDER) + 5))
    for row in rows:
        if row is None:
            continue
        dims = row["data"]["dimensions"]
        overall = row["data"]["overall_accuracy"]
        cells = []
        for full, _ in DIM_ORDER:
            d = dims.get(full)
            if d is None:
                cells.append("N/A")
                continue
            if not d.get("evaluated", True):
                cells.append("NOT_EVAL")
            else:
                cells.append(f"{d['raw_score']*100:.2f}%")
        lines.append(f"| {row['label']} | {row['family']} | {row['k']} | " + " | ".join(cells) + f" | **{overall:.2f}%** | {'PASS' if row['data']['passed'] else 'FAIL'} |")
    lines.append("")

    # ---- average ----
    valid = [r for r in rows if r is not None]
    if valid:
        avg = sum(r["data"]["overall_accuracy"] for r in valid) / len(valid)
        lines.append(f"**五模型平均总准确率：{avg:.2f}%**（v1 合成数据为 91.67%，不可比：维度定义与 ground truth 均已改变）")
        lines.append("")

    # ---- qwen3 identity disclosure ----
    lines.append("> **关于 Qwen3-1.7B 的 100%**：该模型的切分方案是 bridge 内的手工固定设计，专家基线对其的定义与该设计同源（见 `evaluation/expert_baseline.py`），因此 S_granularity 等维度实为**设计与专家预期的一致性检查**，而非独立测量；其 100% 应据此解读。四个 CNN 模型的 S_granularity 由 DP 求解器真实产出，与专家阶段设计存在真实差异，是本报告中区分度最强的维度。")
    lines.append("")

    # ---- notes per model ----
    lines.append("## 三、各维度实测细节")
    lines.append("")
    for row in rows:
        if row is None:
            continue
        dims = row["data"]["dimensions"]
        lines.append(f"### {row['label']}（K={row['k']}）")
        lines.append("")
        g = dims.get("Function Decomposition Granularity (S_granularity)", {}).get("details", {})
        if g.get("metric"):
            src = "手工固定切分方案（与专家基线同源，实为一致性检查）" if row["family"] == "LLM" else "DP 求解器按显存/时延代价真实切分"
            lines.append(f"- **S_granularity**：工具划分 {g.get('tool_groups')} 组 vs 专家设计 {g.get('expert_groups')} 组；一致共属对 {g.get('agreeing_pairs')}/{g.get('total_union_pairs')}。划分来源：{src}。")
        r = dims.get("Runtime Execution & Semantic Output Equivalence (S_runtime)", {}).get("details", {})
        if r:
            smoke = r.get("smoke_tests", {})
            num = r.get("numerical_evidence", {})
            lines.append(f"- **S_runtime**：handler 真实执行冒烟 {smoke.get('passed')}/{smoke.get('total')}；数值等价证据：{'max_tensor_diff=' + str(num.get('max_tensor_diff')) + '（实测）' if num.get('available') else '**无实测证据（硬编码切分方案，未执行链式数值验证）**'}")
        lines.append("")

    # ---- caveats ----
    lines.append("## 四、诚实地说明（局限性）")
    lines.append("")
    lines.append("1. **S_api / S_infra 对本类基准是轻量检查**：MOPAR 推理管线只暴露单一同步入口（`POST /predict` 或 `POST /chat`），专家基线与工具在该契约上一致属预期，不构成强区分度；区分度主要来自 S_granularity 与 S_runtime。")
    lines.append("2. **Qwen3-1.7B 的切分是手工设计**（bridge 内固定 4 切分方案），其 max_tensor_diff 为断言值而非实测值，因此其 S_runtime 中的数值证据不计分；真实 GPU 多卡吞吐数据见 `runs/qwen3_real_gpu_partition.json`（独立实验，未计入分数）。")
    lines.append("3. **S_template 的 AST 检查只验证结构规范**（handler 签名、事件解析、结构化响应、异常捕获），不能证明 handler 真正执行了对应切片的模型算子。当前由 bridge 合成的 handler 返回切片元数据而非模型前向输出——**该局限如实呈现，不在评测器中掩盖**。")
    lines.append("4. 专家基线由项目作者编写（`evaluation/expert_baseline.py`），阶段规则固定、独立于工具运行输出，但非第三方盲测。")
    lines.append("")
    lines.append("## 五、v1 报告勘误")
    lines.append("")
    lines.append("v1（`docs/experimental_evaluation_report.md`）中的六维封装准确率（91.33%~92.32%，均值 91.67%）由哈希合成而非实测，该报告仅作历史记录保留，**其中的准确率数据不应被引用**。微切分资源效益部分（MC 节省、Qwen3 多卡吞吐）来自独立实测实验，不受此勘误影响。")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {OUT_MD}")


if __name__ == "__main__":
    main()
