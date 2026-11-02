"""D2 裸机版计量探针：逐片子进程的 RSS 峰值 vs 配额，及聚合削减率。

论文 Fig.10 的口径要在真执行上复核：每个片在一个**独立子进程**里按
``runtime._run_slice`` 的方式启动（zoo.load → record → 取本片 → release 非本片
权重 → gc+malloc_trim），父进程以 10 Hz 采样 ``/proc/<pid>/status`` 的 VmRSS，
并在子进程进入 serving 阶段时用 ``/proc/<pid>/clear_refs``（写 5）把内核维护的
VmHWM 复位——否则峰值里躺着 record 阶段的整服务激活，量出来的就不是“配额
保护的 serving 水位”。

两种口径都落盘（这是实测出来的结构，不是含糊其辞的地方）：

- **进程口径**（serving_peak_mb）：trim+复位后的进程级峰值 RSS。K8s 的
  ``limits.memory``/OOMKilled 看的就是它——但 torch 运行时 base（~370MB）
  也在里面，小配额片必然“超”。
- **净增量口径**（net_peak_mb = serving_peak - serving 起点 base）：RLIMIT_AS
  的语义（runtime 把配额加在进程 base 之上），也是 D2 判定用的口径。

聚合削减率同时给出两套，并附 MC 口径（Σ alloc×活跃时长 vs 单体 alloc×时长）
——Fig.10 的 66.92% 是 MC 削减；线性链下**瞬时** RSS 聚合（每片各付一份
torch base + 各自权重副本）不减反增是真实测量事实，报告里如实呈现。

Run: python -m capsule.meters.mem_meter --model resnet50 --slices 3
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import torch

from slicer.models import zoo
from slicer.models.graphify import _iter_tensors
from slicer.faas.program import record
from slicer.models.weight_shard import expand_parts
from slicer.partition.cost import PlatformConfig, slice_footprint

REPO_ROOT = Path(__file__).resolve().parents[2]  # capsule/meters/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_HZ = 10.0


def _return_to_os() -> None:
    """与 runtime._return_to_os 同法：gc 之后还要 glibc 归还，RSS 才真降。

    release 置空参数表只丢最后一个引用；从模型加载中部 arena 拿走的块，glibc
    不会主动 unmap。不 trim 的话，serving 起点的 RSS 里仍躺着整网 record 的
    中间激活，峰值就成了冷启动的化石而非 serving 的真实水位。
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _status_fields(pid: int) -> dict[str, float]:
    """VmRSS / VmHWM in MB from /proc/<pid>/status (0.0 when the process is gone)."""
    try:
        fields = {}
        for line in Path("/proc/%d/status" % pid).read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                fields[line.split(":")[0]] = float(line.split()[1]) / 1024.0
        return fields
    except OSError:
        return {}


def _reset_hwm(pid: int) -> None:
    """clear_refs=5: 把 VmHWM 复位为当前 RSS，让下一段峰值只属于 serving。"""
    try:
        Path("/proc/%d/clear_refs" % pid).write_text("5\n")
    except OSError:
        pass


def _pid() -> int:
    import os
    return os.getpid()


def _serve_child(model_key: str, parts: list[list[str]], index: int | None,
                 slice_inputs: dict | None, n_iters: int, seed: int, device: str,
                 channel) -> None:  # noqa: ANN001 (mp.Queue)
    """子进程：按 runtime._run_slice 的方式成为一片（或单体），循环 serving。

    ``index=None`` 表示单体口径：跑整条 program。trim/复位/serving 三段都向
    父进程发阶段标记；done 消息自带最终 VmHWM 与活跃时长——子进程随后即退出，
    父进程再读 /proc 会拿不到，所以峰值必须由子进程自己读出捎带回来。
    """
    model, inputs = zoo.load(model_key, device=device, seed=seed)
    program = record(model, inputs, name=model_key, device=device)
    channel.put(("recorded", {}))
    times: list[float] = []
    if index is None:
        _return_to_os()
        channel.put(("serving", {"freed_mb": 0.0,
                                 "resident_mb": _status_fields(_pid()).get("VmRSS", 0.0)}))
        for _iteration in range(n_iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                program.run(inputs)
            times.append(time.perf_counter() - t0)
            time.sleep(0.05)  # 给 10 Hz 采样留出观测窗
    else:
        case = program.slices(parts)[index]
        program.steps = ()
        freed = program.release(case.keeps)
        _return_to_os()
        channel.put(("serving", {"freed_mb": freed / 2**20,
                                 "resident_mb": _status_fields(_pid()).get("VmRSS", 0.0)}))
        loop_inputs = {key: value.clone() for key, value in slice_inputs.items()}
        for _iteration in range(n_iters):
            env: dict = {}
            for key in case.request:
                env[key] = loop_inputs[key].clone()
            for _peer, keys in case.inbound:
                for key in keys:
                    env[key] = loop_inputs[key].clone()
            t0 = time.perf_counter()
            case.run(env)
            times.append(time.perf_counter() - t0)
            time.sleep(0.05)
    channel.put(("done", {"serving_peak_mb": _status_fields(_pid()).get("VmHWM", 0.0),
                          "active_s_total": sum(times),
                          "active_s_mean": sum(times) / max(1, len(times))}))


def _sample_until_done(pid: int, channel, report: dict) -> None:
    """10 Hz 采样循环；serving 标记到达时复位 VmHWM，done 后收尾。"""
    curve: list[float] = []
    serving = False
    while True:
        fields = _status_fields(pid)
        if fields:
            curve.append(fields.get("VmRSS", 0.0))
            if serving:
                report.setdefault("serving_rss_samples", []).append(fields.get("VmRSS", 0.0))
        try:
            stage, payload = channel.get(timeout=1.0 / SAMPLE_HZ)
        except Exception:  # queue.Empty -- keep sampling
            continue
        if stage == "serving" and not serving:
            serving = True
            report.update(payload)
            report["boot_peak_mb"] = _status_fields(pid).get("VmHWM", 0.0)
            _reset_hwm(pid)
        elif stage == "done":
            # 子进程捎回的 VmHWM 是权威值；/proc 再读一次仅为交叉核对。
            report["serving_peak_mb"] = payload.get("serving_peak_mb", 0.0) or \
                _status_fields(pid).get("VmHWM", 0.0)
            report["active_s_total"] = payload.get("active_s_total")
            report["active_s_mean"] = payload.get("active_s_mean")
            break
    report["sampled_peak_mb"] = max(report.get("serving_rss_samples", [0.0])) \
        if report.get("serving_rss_samples") else 0.0
    report["samples"] = len(curve)


def measure(model_key: str, slices: int = 3, n_iters: int = 6, seed: int = 0,
            device: str = "cpu") -> dict:
    """跑一遍 D2 计量；返回逐片 peak/alloc（双口径）与聚合削减率。"""
    import multiprocessing as mp

    model, inputs = zoo.load(model_key, device=device, seed=seed)
    program = record(model, inputs, name=model_key, device=device)
    parts = expand_parts(program, slices)
    cases = program.slices(parts)
    platform = PlatformConfig()

    # 一遍链内执行，取出每片的“到达值”（request 键 + inbound 键的激活）。
    # 必须在该片 case.run **之前**抓取：SliceProgram.run 会按 drop 语义把
    # “最后一次使用后”的键从 env 弹出，事后取 inbound 会拿到空。
    envs: list[dict] = [{} for _ in cases]
    request = dict(zip(program.inputs, _iter_tensors(inputs)))
    slice_inputs: list[dict] = []
    for case, env in zip(cases, envs):
        for key in case.request:
            env[key] = request[key]
        for peer, keys in case.inbound:
            for key in keys:
                env[key] = envs[peer][key]
        keys = list(case.request) + [k for _, peer_keys in case.inbound for k in peer_keys]
        slice_inputs.append({key: env[key].detach().clone() for key in keys})
        case.run(env)

    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    for case, payload in zip(cases, slice_inputs):
        channel = ctx.Queue()
        child = ctx.Process(target=_serve_child,
                            args=(model_key, parts, case.index, payload,
                                  n_iters, seed, device, channel))
        child.start()
        report: dict = {"slice_id": case.index, "layers": len(case.layers),
                        "steps": len(case.steps)}
        _sample_until_done(child.pid, channel, report)
        child.join()
        alloc_mb = platform.allocate_mb(slice_footprint(program.graph, parts[case.index]))
        base_mb = report.get("resident_mb", 0.0)
        report["alloc_mb"] = alloc_mb
        report["base_mb_after_trim"] = round(base_mb, 1)
        report["serving_peak_mb"] = round(report["serving_peak_mb"], 1)
        report["net_peak_mb"] = round(max(0.0, report["serving_peak_mb"] - base_mb), 1)
        report["ratio_process_peak_over_alloc"] = round(report["serving_peak_mb"] / alloc_mb, 3)
        report["ratio_net_peak_over_alloc"] = round(report["net_peak_mb"] / alloc_mb, 3)
        report["active_s_mean"] = round(report.get("active_s_mean") or 0.0, 6)
        rows.append(report)
        print("[mem_meter] slice %d: serving_peak=%.1fMB net_peak=%.1fMB base=%.1fMB "
              "alloc=%.0fMB ratio_net=%.3f ratio_proc=%.3f (freed=%.1fMB, %d samples)"
              % (case.index, report["serving_peak_mb"], report["net_peak_mb"],
                 base_mb, alloc_mb, report["ratio_net_peak_over_alloc"],
                 report["ratio_process_peak_over_alloc"], report.get("freed_mb", 0.0),
                 report.get("samples", 0)))

    channel = ctx.Queue()
    child = ctx.Process(target=_serve_child,
                        args=(model_key, parts, None, None, n_iters, seed, device, channel))
    child.start()
    mono: dict = {"slice_id": "monolith"}
    _sample_until_done(child.pid, channel, mono)
    child.join()
    mono_base = mono.get("resident_mb", 0.0)
    mono_peak = mono["serving_peak_mb"]
    mono_net = max(0.0, mono_peak - mono_base)
    mono_alloc = platform.allocate_mb(slice_footprint(program.graph, program.graph.order))
    mono_active = mono.get("active_s_mean") or 0.0
    print("[mem_meter] monolith: serving_peak=%.1fMB net_peak=%.1fMB base=%.1fMB alloc=%.0fMB"
          % (mono_peak, mono_net, mono_base, mono_alloc))

    agg_peak = sum(r["serving_peak_mb"] for r in rows)
    agg_net = sum(r["net_peak_mb"] for r in rows)
    # MC 口径（Fig.10 的 66.92% 说的就是这个）：Σ 配额×活跃时长 vs 单体配额×时长。
    mc_sliced = sum(r["alloc_mb"] * r["active_s_mean"] for r in rows)
    mc_mono = mono_alloc * mono_active
    summary = {
        "model_key": model_key, "slices": slices, "n_iters": n_iters, "seed": seed,
        "device": device, "sample_hz": SAMPLE_HZ,
        "per_slice": rows,
        "monolith": {"serving_peak_mb": round(mono_peak, 1),
                     "net_peak_mb": round(mono_net, 1),
                     "base_mb_after_trim": round(mono_base, 1),
                     "alloc_mb": mono_alloc,
                     "active_s_mean": round(mono_active, 6)},
        "aggregate_process_peak_mb": round(agg_peak, 1),
        "aggregate_net_peak_mb": round(agg_net, 1),
        "peak_reduction_ratio_process": round(1.0 - agg_peak / mono_peak, 4) if mono_peak else None,
        "peak_reduction_ratio_net": round(1.0 - agg_net / mono_net, 4) if mono_net else None,
        "mc_sliced_mb_s": round(mc_sliced, 3),
        "mc_monolith_mb_s": round(mc_mono, 3),
        "mc_reduction_percent": round(100.0 * (1.0 - mc_sliced / mc_mono), 2) if mc_mono else None,
        "d2_pass_net": all(r["ratio_net_peak_over_alloc"] <= 1.0 for r in rows),
        "d2_pass_process": all(r["ratio_process_peak_over_alloc"] <= 1.0 for r in rows),
    }
    print("[mem_meter] aggregate process_peak=%.1fMB (reduction %.1f%%), "
          "net_peak=%.1fMB (reduction %.1f%%), MC=%.1f vs %.1f MB·s (%.1f%% saved)"
          % (agg_peak, 100 * (summary["peak_reduction_ratio_process"] or 0),
             agg_net, 100 * (summary["peak_reduction_ratio_net"] or 0),
             mc_sliced, mc_mono, summary["mc_reduction_percent"] or 0))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D2 bare-metal per-slice RSS meter.")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--n-iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    report = measure(args.model, args.slices, args.n_iters, args.seed, args.device)
    out = args.out or ("runs/mem_meter_%s_k%d.json" % (args.model, args.slices))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[mem_meter] report -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
