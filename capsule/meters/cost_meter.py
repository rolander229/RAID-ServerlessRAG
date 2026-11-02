"""计费记账：计量 JSON → 资源量表 + AWS Lambda 锚点影子价列。

计量与定价解耦（lambda-anchor-research.md §2 的 crater 口径）：同一份计量数据
先出**资源量**（内存 GB·s / GB-hour、vCPU-hour、GPU 卡时 / 显存 GB-hour），
再挂**影子价**列。CPU 片锚定 AWS Lambda：

    λ        = 1769 MB/vCPU     （官方 "At 1,769 MB, a function has the
                                  equivalent of one vCPU"；repo 内 1792 是其取整近似）
    price    = $1.667e-5 /GB·s  （x86 列表价）
    vCPU     = ceil(alloc_mb / 1769)   ← K8s requests.cpu 的折算规则
    $ /请求  = (alloc_mb/1024) × active_s × 1.667e-5

GPU 片在锚点确定前（候选阿里云 FC，¥2.31e-4/显存GB·s 未拍板）只出**资源量列**
（卡时、显存 GB-hour），人民币列留空并注明 "GPU 锚点未定"。

输入计量 JSON（B2/A4 产物）：
- A4：runs/slice_server_logs/{model}_slice{i}.jsonl —— 每跳 busy_ms/hop_ms/moved_mb；
- B2：runs/mem_meter_{model}_k{n}.json —— 逐片 alloc_mb / serving_peak_mb。

Run:
  python -m capsule.meters.cost_meter \
      --logs "runs/slice_server_logs/resnet50_slice*.jsonl" \
      --mem-meter runs/mem_meter_resnet50_k3.json \
      --out runs/cost_report_resnet50.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # capsule/meters/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MB = 2 ** 20
# AWS Lambda 锚点（lambda-anchor-research.md §1.1，已核实官方文档）
LAMBDA_MB_PER_VCPU = 1769.0
LAMBDA_PRICE_PER_GB_S = 1.667e-5
LAMBDA_MIN_ALLOC_MB = 128.0
GPU_PRICE_NOTE = "GPU 锚点未定（候选阿里云 FC，¥2.31e-4/显存GB·s 未拍板）——人民币列留空"


def _vcpu_for(alloc_mb: float) -> int:
    """AWS 锚点的 requests.cpu 折算：ceil(alloc_mb / λ)，下限 1（K8s 语义）。"""
    return max(1, math.ceil(alloc_mb / LAMBDA_MB_PER_VCPU))


def read_slice_logs(pattern: str) -> dict[int, dict]:
    """把 A4 的 JSONL 日志聚成逐片活跃度：requests、busy/hop 均值、moved_mb。"""
    rows: dict[int, dict] = {}
    for path in sorted(glob.glob(pattern)):
        entries = [json.loads(line) for line in Path(path).read_text().splitlines()
                   if line.strip() and json.loads(line).get("status") == "ok"]
        if not entries:
            continue
        slice_id = entries[0]["slice_id"]
        rows[slice_id] = {
            "slice_id": slice_id,
            "requests": len(entries),
            "busy_ms_mean": statistics.fmean(e.get("busy_ms", 0.0) for e in entries),
            "hop_ms_mean": statistics.fmean(e.get("hop_ms", 0.0) for e in entries),
            "moved_mb_mean": statistics.fmean(e.get("moved_mb", 0.0) for e in entries),
            "moved_in_mb_mean": statistics.fmean(e.get("moved_in_mb", 0.0) for e in entries),
            "rss_hwm_mb": max(e.get("rss_mb", 0.0) for e in entries),
            "log_source": path,
        }
    return rows


def _row_from_log(log_row: dict, alloc_mb: float | None, peak_mb: float | None,
                  gpu: bool = False) -> dict:
    """计量行 → 资源量 + AWS 影子价（每请求口径；GB·s 与 GB-hour 并列给出）。"""
    alloc = alloc_mb or LAMBDA_MIN_ALLOC_MB
    active_s = log_row["busy_ms_mean"] / 1e3
    gb_s = alloc / 1024.0 * active_s
    vcpu = _vcpu_for(alloc)
    row = {
        "slice_id": log_row["slice_id"],
        "requests_observed": log_row["requests"],
        "alloc_mb": round(alloc, 1),
        "vcpu_requests": vcpu,                      # ceil(alloc/1769)，K8s requests.cpu
        "active_s_per_request": round(active_s, 6),
        "resource_quantities": {
            "memory_gb_s_per_request": round(gb_s, 6),
            "memory_gb_hour_per_request": round(gb_s / 3600.0, 9),
            "vcpu_hour_per_request": round(vcpu * active_s / 3600.0, 9),
        },
        "shadow_price_aws": {
            "anchor": "AWS Lambda x86 (λ=1769 MB/vCPU, $1.667e-5/GB·s)",
            "usd_per_request": round(gb_s * LAMBDA_PRICE_PER_GB_S, 9),
            "usd_per_1000_requests": round(gb_s * LAMBDA_PRICE_PER_GB_S * 1000, 6),
        },
        "moved_mb_per_request": round(log_row["moved_mb_mean"], 3),
        "rss_hwm_mb": log_row.get("rss_hwm_mb"),
    }
    if peak_mb is not None:
        row["serving_peak_mb"] = peak_mb
    if gpu:
        # GPU 片：资源量口径照出，价格列留空（锚点未定，计量与定价解耦）。
        row["resource_quantities"]["gpu_card_hours_per_request"] = None
        row["resource_quantities"]["vram_gb_hour_per_request"] = None
        row["shadow_price_aws"] = None
        row["price_note"] = GPU_PRICE_NOTE
    return row


def build_report(log_pattern: str, mem_meter_path: str | None = None,
                 model: str | None = None) -> dict:
    """A4 日志 (+ B2 配额) → 计费记账报告。"""
    logs = read_slice_logs(log_pattern)
    if not logs:
        raise SystemExit("no slice_server JSONL logs match %s" % log_pattern)
    allocs: dict[int, float] = {}
    peaks: dict[int, float] = {}
    mem_meta = None
    if mem_meter_path:
        mem = json.loads(Path(mem_meter_path).read_text())
        mem_meta = {"source": mem_meter_path, "slices": mem.get("slices")}
        for row in mem.get("per_slice", []):
            allocs[row["slice_id"]] = row.get("alloc_mb")
            peaks[row["slice_id"]] = row.get("serving_peak_mb")

    rows = [_row_from_log(entry, allocs.get(sid), peaks.get(sid))
            for sid, entry in sorted(logs.items())]

    total_busy_s = sum(r["active_s_per_request"] for r in rows)
    total_usd = sum(r["shadow_price_aws"]["usd_per_request"] for r in rows)
    total_gb_s = sum(r["resource_quantities"]["memory_gb_s_per_request"] for r in rows)
    total_vcpu_hour = sum(r["resource_quantities"]["vcpu_hour_per_request"] for r in rows)
    summary = {
        "slices": len(rows),
        "memory_gb_s_per_request_total": round(total_gb_s, 6),
        "memory_gb_hour_per_request_total": round(total_gb_s / 3600.0, 9),
        "vcpu_hour_per_request_total": round(total_vcpu_hour, 9),
        "usd_per_request_total": round(total_usd, 9),
        "usd_per_1000_requests_total": round(total_usd * 1000, 6),
        "serial_active_s_per_request": round(total_busy_s, 6),
    }
    return {
        "_meta": {
            "generated_by": "capsule/meters/cost_meter.py",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": model or "unknown",
            "metering_sources": {"a4_logs": sorted(glob.glob(log_pattern)),
                                 "mem_meter": mem_meta},
            "anchors": {"lambda_mb_per_vcpu": LAMBDA_MB_PER_VCPU,
                        "lambda_price_per_gb_s": LAMBDA_PRICE_PER_GB_S,
                        "note": "repo 内 1792 MB/vCPU 与 $1.667e-5 是同一锚点的取整口径；"
                                "此处用官方精确值 1769。GPU 锚点未定，GPU 行只出资源量。"},
        },
        "per_slice": rows,
        "summary": summary,
    }


def print_report(report: dict) -> None:
    meta = report["_meta"]
    print("[cost_meter] model=%s slices=%d" % (meta["model"], len(report["per_slice"])))
    header = ("slice | alloc_mb | vcpu | busy_s/req | mem_GB·s/req | vCPU·h/req | "
              "moved_MB/req | $/req | $/1k req")
    print("[cost_meter] " + header)
    for row in report["per_slice"]:
        print("[cost_meter] %5d | %8.1f | %4d | %10.6f | %12.6f | %10.3g | %12.3f | %.3g | %.3g"
              % (row["slice_id"], row["alloc_mb"], row["vcpu_requests"],
                 row["active_s_per_request"],
                 row["resource_quantities"]["memory_gb_s_per_request"],
                 row["resource_quantities"]["vcpu_hour_per_request"],
                 row["moved_mb_per_request"],
                 row["shadow_price_aws"]["usd_per_request"],
                 row["shadow_price_aws"]["usd_per_1000_requests"]))
    s = report["summary"]
    print("[cost_meter] TOTAL: mem=%.4f GB·s/req, vCPU·h=%.3g/req, $=%.3g/req, $=%.3g/1k req"
          % (s["memory_gb_s_per_request_total"], s["vcpu_hour_per_request_total"],
             s["usd_per_request_total"], s["usd_per_1000_requests_total"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Metering JSON -> resource table + AWS shadow price.")
    parser.add_argument("--logs", required=True, help="glob of A4 slice_server JSONL logs")
    parser.add_argument("--mem-meter", default=None, help="B2 mem_meter JSON (alloc/peak per slice)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = build_report(args.logs, args.mem_meter, args.model)
    print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print("[cost_meter] report -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
