"""D3 的分母：从 runs/ 既有 JSON 回填裸机基线（不新跑任何实验）。

数据源（全部只读）：
- ``runs/e5_end_to_end.json`` —— 裸机闭环并发基线：MOPAR@λ=1792 行的
  per-hop 延迟（mean_span_ms）、moved_mb、mean_link_ms、P95(C=1/2/4/8)、吞吐；
- ``runs/qwen3_real_gpu_partition.json`` / ``runs/chatglm3_real_gpu_partition.json``
  —— 真实多 GPU LLM 基线：char_per_s、comm_ms、mem_gpus_gb（MOPAR 与 NonSplit 行）；
- ``runs/fig10_memory_footprint.json`` —— 内存削减上下文（unsplit 配额与 MC 节省）。

口径与诚实性约束：
- 每个指标项要么带真实值与出处文件，要么 ``"source": "missing"`` 并写明缺什么；
  绝不拿别的口径凑数（任务书红线：不为凑数跑重型 LLM 实验）。
- e5 的 CNN/小模型在 λ=1792 下 DP 解本就是 1 片：per-hop 只有一跳、moved_mb=0
  是**正确数据**而非缺数据，照实回填；多跳基线来自三个大 Transformer 行。

Run: python -m capsule.meters.backfill_baselines   # 写 runs/baselines.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # capsule/meters/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNS = REPO_ROOT / "runs"
E5 = RUNS / "e5_end_to_end.json"
GPU_FILES = {"qwen3_1_7b": RUNS / "qwen3_real_gpu_partition.json",
             "chatglm3_6b": RUNS / "chatglm3_real_gpu_partition.json"}
FIG10 = RUNS / "fig10_memory_footprint.json"
CONCURRENCIES = (1, 2, 4, 8)


def _sourced(value, source: str, **extra) -> dict:
    out = {"value": value, "source": source}
    out.update(extra)
    return out


def _missing(reason: str) -> dict:
    return {"value": None, "source": "missing", "reason": reason}


def _mopar_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not E5.is_file():
        return rows
    for row in json.loads(E5.read_text()):
        if row.get("method") == "MOPAR" and abs(row.get("mb_per_vcpu", 0) - 1792.0) < 1e-9:
            rows[row["key"]] = row  # 每模型 batch=1 一行；同键后写覆盖前者
    return rows


def _from_e5(key: str, row: dict) -> dict:
    source = "runs/e5_end_to_end.json"
    windows = row.get("measured", {}).get("windows", [])
    by_c = {int(w["concurrency"]): w for w in windows}
    entry: dict = {"method": "MOPAR", "mb_per_vcpu": row.get("mb_per_vcpu"),
                   "analytic_slices": row.get("analytic", {}).get("slices"),
                   "analytic_alloc_mb": row.get("analytic", {}).get("alloc_mb")}

    per_hop = {}
    for c in CONCURRENCIES:
        window = by_c.get(c)
        if window and window.get("slices"):
            per_hop[str(c)] = _sourced([s["mean_span_ms"] for s in window["slices"]],
                                       source, unit="ms/hop")
        else:
            per_hop[str(c)] = _missing("no C=%d window in %s" % (c, source))
    entry["per_hop_latency_ms"] = per_hop

    anchor = by_c.get(1, windows[0] if windows else None)
    if anchor and anchor.get("slices"):
        entry["moved_mb_per_hop"] = _sourced([s["moved_mb"] for s in anchor["slices"]],
                                             source, unit="MB/hop", window_c=anchor["concurrency"])
        entry["mean_link_ms_per_hop"] = _sourced([s["mean_link_ms"] for s in anchor["slices"]],
                                                 source, unit="ms/hop",
                                                 window_c=anchor["concurrency"])
    else:
        entry["moved_mb_per_hop"] = _missing("no measured window in %s" % source)
        entry["mean_link_ms_per_hop"] = _missing("no measured window in %s" % source)

    p95 = {}
    for c in CONCURRENCIES:
        window = by_c.get(c)
        p95[str(c)] = (_sourced(window["p95_ms"], source, unit="ms")
                       if window else _missing("no C=%d window in %s" % (c, source)))
    entry["p95_ms"] = p95

    entry["throughput_rps"] = {
        str(c): (_sourced(by_c[c]["throughput_rps"], source, unit="rps")
                 if c in by_c else _missing("no C=%d window in %s" % (c, source)))
        for c in CONCURRENCIES}

    # e5 是 CNN/Transformer 的裸机闭环，char/s（生成口径）与 GPU 显存分卡不在其中。
    entry["char_per_s"] = _missing("e5_end_to_end.json records latency/throughput, "
                                   "not generation char/s; GPU files cover qwen3/chatglm3 only")
    entry["mem_gpus_gb"] = _missing("e5 is bare-metal CPU; per-GPU memory lives in the "
                                    "real-GPU JSONs for qwen3/chatglm3")
    return entry


def _from_gpu_file(key: str, path: Path) -> dict:
    source = "runs/%s" % path.name
    rows = json.loads(path.read_text())
    mopar = [r for r in rows if r.get("strategy") == "MOPAR"]
    non = [r for r in rows if r.get("strategy") == "NonSplit"]
    entry: dict = {"method": "MOPAR (real multi-GPU)", "rows_available": len(rows)}

    if mopar:
        best = max(mopar, key=lambda r: r.get("gpus", 0))
        entry["char_per_s"] = _sourced(
            [round(r["char_per_s"], 2) for r in mopar], source,
            strategy="MOPAR", gpus=[r.get("gpus") for r in mopar], unit="char/s")
        entry["comm_ms"] = _sourced(
            [round(r["comm_ms"], 3) for r in mopar], source,
            strategy="MOPAR", unit="ms/hop")
        entry["mem_gpus_gb"] = _sourced(best["mem_gpus_gb"], source,
                                        gpus=best.get("gpus"), unit="GB/GPU")
        entry["load_time_s"] = _sourced(best.get("load_time_s"), source, unit="s")
        entry["nonsplit_char_per_s"] = (
            _sourced([round(r["char_per_s"], 2) for r in non], source, strategy="NonSplit")
            if non else _missing("no NonSplit row in %s" % source))
    else:
        for field in ("char_per_s", "comm_ms", "mem_gpus_gb"):
            entry[field] = _missing("no MOPAR row in %s" % source)

    entry["per_hop_latency_ms"] = _missing("single-stream GPU runs; no closed-loop C sweep "
                                           "in runs/ (would need an e6/e7-style rerun)")
    entry["moved_mb_per_hop"] = _missing("not recorded by the GPU partition runs")
    entry["mean_link_ms_per_hop"] = _missing("not recorded by the GPU partition runs")
    entry["p95_ms"] = _missing("single-stream GPU runs; no closed-loop C sweep in runs/")
    entry["throughput_rps"] = _missing("GPU runs report char/s, not requests/s")
    return entry


def build() -> dict:
    rows = _mopar_rows()
    models: dict[str, dict] = {}
    for key, row in sorted(rows.items()):
        models[key] = _from_e5(key, row)
    for key, path in GPU_FILES.items():
        if path.is_file():
            models[key] = _from_gpu_file(key, path)
        else:
            models[key] = {"method": "MOPAR (real multi-GPU)",
                           "char_per_s": _missing("%s not found" % path)}

    if FIG10.is_file():
        fig10 = json.loads(FIG10.read_text())
        for key, payload in fig10.items():
            if key in models:
                models[key]["fig10_memory_context"] = {
                    "unsplit_quota_mb": payload.get("unsplit", {}).get("quota_mb"),
                    "mopar_slices_k": payload.get("mopar", {}).get("slices_k"),
                    "mc_saving_percent": payload.get("mopar", {}).get("mc_saving_percent"),
                    "source": "runs/fig10_memory_footprint.json"}

    return {"_meta": {
        "generated_by": "capsule/meters/backfill_baselines.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": ["runs/e5_end_to_end.json", "runs/qwen3_real_gpu_partition.json",
                    "runs/chatglm3_real_gpu_partition.json", "runs/fig10_memory_footprint.json"],
        "convention": "每个指标项 = {value, source} 或 {value: null, source: 'missing', reason}；"
                      "missing 项不补跑实验（任务书红线）",
        "notes": [
            "e5 MOPAR@λ=1792 下 9 个 CNN/小模型的 DP 解就是 1 片：per-hop 单跳、moved_mb=0 是正确数据。",
            "多跳裸机基线来自 bert_1_3b (3片) / bert_3_0b (8片) / inception_transformer (8片)。",
            "qwen3_1_7b / chatglm3_6b 的 char/s、comm_ms、mem_gpus_gb 来自真实 4×V100 分卡实测 JSON。",
        ]},
        "models": models}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill bare-metal baselines from runs/ JSONs.")
    parser.add_argument("--out", default=str(RUNS / "baselines.json"))
    args = parser.parse_args(argv)
    report = build()
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    models = report["models"]
    print("[baselines] models: %s" % ", ".join(sorted(models)))
    filled = sum(1 for m in models.values()
                 for f in ("per_hop_latency_ms", "moved_mb_per_hop", "mean_link_ms_per_hop",
                           "p95_ms", "char_per_s", "mem_gpus_gb")
                 if isinstance(m.get(f), dict) and m[f].get("source") != "missing")
    total = len(models) * 6
    print("[baselines] metric slots filled: %d/%d (rest are honest 'missing')"
          % (filled, total))
    print("[baselines] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
