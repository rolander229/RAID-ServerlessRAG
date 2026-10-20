"""Fig. 10, Fig. 11, Fig. 12 and Table III: the six methods run as services.

Sec. III-A-2 compares MOPAR with AlpaServe, Clockwork++, NonSplit, Uniform and Unsplit
on three quantities: normalised memory utilisation (Fig. 10a), 95th-percentile latency
(Fig. 10b/c) and cost per request (Table III). Two of those three are *measured*, and
measuring them means running each plan as the service the plan describes -- one process
per slice, its own quota, its own thread count, the lanes between them carrying the
activations -- which is what :class:`slicer.faas.runtime.Deployment` does.

What this script decides, and why:

**One deployment per distinct plan, not per method.** At the paper's own
``lambda = 1792`` MB/vCPU, Eq. (6)'s latency constraint admits no cut for nine of the
twelve CPU services (``runs/e3b_cut_price.json``), so all six methods hand back the same
one-slice plan. Running those six separately would measure the same process six times
and call it a comparison, so plans are grouped by their executable content -- groups,
etas, quotas, placement -- and every method that shares a configuration reports
``identical_to`` the one that was run.

**Concurrency is swept inside one start.** A slice loads the whole service before it
frees the layers it does not keep, so a start costs a model load per slice; the windows
are run back to back through one started service by :meth:`Deployment.sweep`, which
splits the reports by position afterwards.

**The load is endogenous, so throughput is printed next to the percentile.** The window
is closed-loop -- the paper's diurnal trace (Fig. 9) is not reproduced here -- which
means the offered load is whatever the service absorbs. A 95th percentile taken at an
unstated load is not comparable with the paper's, so every row carries its requests per
second, and Fig. 10b's bar is read at the widest window.

**Money and memory come from Eq. (6), latency from the wall clock.** Table III's cost is
the bill MPE optimised, and it is reported as such; the memory column is given both
ways, since the gap between ``100 * sum(alloc * t_analytic)`` and the same sum over
measured spans is this reproduction's error bar on the profiling model.

Run on v100:

    python experiments/e5_end_to_end.py --models gcn1 \\
        --concurrencies 1,2,4,8 --requests 20

``--verbose`` names the stage that is running and keeps a clock on it: the SP passes,
each function process as it comes up, each window as it finishes, and the Eq. (6)
plan behind every row. Without it a BERT-1.3B service is silent for the best part of
two minutes, which is indistinguishable from a hang. Nothing it prints feeds the JSON,
so the artifact is the same either way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicer.faas.runtime import Deployment  # noqa: E402
from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import build_structure_graph  # noqa: E402
from slicer.partition.baselines import FIGURE_10, plan_for  # noqa: E402
from slicer.partition.cost import PlatformConfig, TimingModel, slice_footprint  # noqa: E402
from slicer.partition.hypad import PartitionPlan, partition  # noqa: E402
from slicer.profile.service import profile_model  # noqa: E402

MB = 2 ** 20
GB = 2 ** 30

#: Table III prices a request in units of 1e-5 dollars.
TABLE_III_UNIT = 1e-5

#: Set by ``--verbose``. Which stage is running, and how long it has been at it, is
#: otherwise invisible: one SP profile and one three-slice start hold the terminal
#: silent for the best part of two minutes on BERT-1.3B, which looks exactly like a
#: hang. Nothing here feeds the JSON, so the artifact is the same either way.
VERBOSE = False
_START = time.perf_counter()
_PRINT = threading.Lock()


def _log(message: str) -> None:
    if VERBOSE:
        with _PRINT:
            print("[%7.1fs] %s" % (time.perf_counter() - _START, message), flush=True)


class _stage:
    """Name a stage, beat while it lasts, report what it took (``--verbose`` only).

    The beats exist because the long stages here have no sub-progress at all -- a
    ``Deployment.sweep`` posts every window and then blocks on the reports -- so a
    line that repeats is the only way to say "still working, this is the stage".
    """

    def __init__(self, label: str, beat_s: float = 20.0) -> None:
        self.label = label
        self.beat_s = beat_s
        self.started = time.perf_counter()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_stage":
        self.started = time.perf_counter()
        _log("%s ..." % self.label)
        if VERBOSE:
            self._stop = threading.Event()

            def beat() -> None:
                while not self._stop.wait(self.beat_s):
                    _log("    %s ... still running, %.0f s"
                         % (self.label, time.perf_counter() - self.started))

            self._thread = threading.Thread(target=beat, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        _log("%s %s in %.1f s" % (self.label, "FAILED" if exc_type else "done",
                                  time.perf_counter() - self.started))
        return False


def _on_pass(label: str) -> None:
    _log("    SP %s" % label)


def _on_slice_ready(index: int, total: int, row: dict) -> None:
    _log("    slice %d/%d up: %d layers, %d steps, base %.0f MB, quota %.0f MB, "
         "freed %.0f MB" % (index, total, row["layers"], row["steps"],
                            row["base_bytes"] / MB, row["quota_bytes"] / MB,
                            row["freed_bytes"] / MB))


def _placement(servers: list[int]) -> tuple[list[int], int]:
    """Name a server for every slice, including the ones the plan has nowhere to put.

    :func:`slicer.partition.baselines.place` returns ``-1`` once Table II's fourteen
    servers are full. On one box a server is a label that decides which link a hop is
    modelled over -- same label means the shared-memory copy, different means the
    ConnectX -- so an overflowing slice gets a fresh label, which is the pessimistic
    reading, and the count is reported rather than absorbed.
    """
    top = max([server for server in servers if server >= 0], default=-1)
    out: list[int] = []
    overflow = 0
    for server in servers:
        if server >= 0:
            out.append(server)
            continue
        overflow += 1
        top += 1
        out.append(top)
    return out, overflow


def _analytic(plan: PartitionPlan, platform: PlatformConfig) -> dict:
    """What Eq. (6) says this plan costs, before anything is run.

    ``memory_per_100rps_gb`` is Little's law read on the paper's own unit: serving 100
    requests a second means holding ``100 * t`` instances of each slice, each of which
    is allocated ``M_gi``, so the memory a service needs is ``100 * sum(M * t)``.
    """
    mc = sum(slice_plan.memory_consumption_gb_s for slice_plan in plan.slices)
    return {"slices": len(plan.slices),
            "etas": [s.eta for s in plan.slices],
            "alloc_mb": [round(s.alloc_mb, 0) for s in plan.slices],
            "latency_ms": round(plan.latency_s * 1e3, 3),
            "cost_us": round(plan.cost_usd * 1e6, 4),
            "cost_table_iii": round(plan.cost_usd / TABLE_III_UNIT, 6),
            "mc_gb_s": round(mc, 6),
            "memory_per_100rps_gb": round(100.0 * mc, 3),
            "comm_mb": round(plan.comm_bytes / MB, 3)}


def _slice_rows(load) -> list[dict]:
    """Per-slice summaries of one window: where the request's seconds actually went."""
    rows = []
    for row in sorted(load.per_slice, key=lambda r: r["index"]):
        spans, busy, waits = row["span_s"], row["busy_s"], row["wait_s"]
        if not spans:
            continue
        rows.append({"index": row["index"], "eta": row["eta"],
                     "requests": row["requests"],
                     "mean_span_ms": round(1e3 * sum(spans) / len(spans), 3),
                     "max_span_ms": round(1e3 * max(spans), 3),
                     "mean_busy_ms": round(1e3 * sum(busy) / len(busy), 3),
                     "mean_wait_ms": round(1e3 * sum(waits) / len(waits), 3),
                     "mean_transport_ms": round(1e3 * sum(row["transport_s"])
                                                / len(spans), 3),
                     "mean_link_ms": round(1e3 * sum(row["link_s"]) / len(spans), 3),
                     "moved_mb": round(sum(row["moved_bytes"]) / MB, 3),
                     "resident_mb": round(max(row["resident_bytes"]) / MB, 1)})
    return rows


def _measure(key: str, batch: int, groups, etas, allocs, servers, platform,
             concurrencies: list[int], requests: int) -> dict:
    """Start this plan once, push every window through it, and take the reports."""
    service = None
    started = time.time()
    try:
        with _stage("build: cut %s into %d slice(s), size the lanes and the arena"
                    % (key, len(groups))):
            service = Deployment.build(key, groups, etas=etas, allocs=allocs,
                                       servers=servers, platform=platform,
                                       mb_per_vcpu=platform.mb_per_vcpu, batch=batch,
                                       queue_depth=max(concurrencies))
        with _stage("start: spawn %d function process(es), one per slice" % len(groups)):
            service.start(device="cpu", batch=batch, mb_per_vcpu=platform.mb_per_vcpu,
                          on_ready=_on_slice_ready)
        campaigns = [(requests, c) for c in concurrencies]
        with _stage("load: %s" % ", ".join("%d req @ C=%d" % camp for camp in campaigns),
                    beat_s=10.0):
            loads = service.sweep(service.inputs, campaigns)
        windows = []
        for concurrency, load in zip(concurrencies, loads):
            latencies = load.latencies_s
            _log("    C=%d  %d requests in %.1f s  mean %.1f ms  p50 %.1f ms  "
                 "p95 %.1f ms  %.2f rps"
                 % (concurrency, len(latencies), load.seconds,
                    1e3 * sum(latencies) / max(1, len(latencies)),
                    1e3 * load.percentile(50), 1e3 * load.percentile(95),
                    load.throughput))
            windows.append({"concurrency": concurrency,
                            "requests": len(latencies),
                            "wall_s": round(load.seconds, 3),
                            "throughput_rps": round(load.throughput, 3),
                            "mean_ms": round(1e3 * sum(latencies) / max(1, len(latencies)),
                                             3),
                            "p50_ms": round(1e3 * load.percentile(50), 3),
                            "p95_ms": round(1e3 * load.percentile(95), 3),
                            "p99_ms": round(1e3 * load.percentile(99), 3),
                            "wire_mb": round(load.wire_bytes / MB, 3),
                            "latencies_ms": [round(1e3 * value, 2)
                                             for value in load.latencies_s],
                            "slices": _slice_rows(load)})
        return {"ok": True, "startup_s": round(time.time() - started, 1),
                "base_mb": [round(row["base_bytes"] / MB, 1) for row in service.ready],
                "freed_mb": [round(row["freed_bytes"] / MB, 1)
                             for row in service.ready],
                "windows": windows}
    except Exception as error:  # one unrunnable plan must not lose the other eleven
        return {"ok": False, "error": "%s: %s" % (type(error).__name__, error),
                "startup_s": round(time.time() - started, 1)}
    finally:
        if service is not None:
            with _stage("close the service (stop the slices, free the lanes)"):
                service.close()


def run(key: str, batch: int, concurrencies: list[int], requests: int,
        mb_per_vcpu: float, methods: list[str] | None = None) -> list[dict]:
    """Every Fig. 10 method's plan for one service, measured or aliased.

    ``methods`` narrows *which* plans get a service started for them, which is how a
    single failing deployment is reproduced without paying for the other five. The
    analytic columns are still written for every method, because Eq. (6) does not
    care whether the box had time to run it.
    """
    with _stage("%s b=%d: load weights" % (key, batch)):
        model, inputs = zoo.load(key, device="cpu", batch=batch)
    with _stage("%s: structure pass (leaf-module graph)" % key):
        graph = build_structure_graph(model, inputs, name=key, device="cpu")
    with _stage("%s: SP profiling passes (Sec. II-B)" % key):
        profile = profile_model(model, inputs, key=key, device="cpu", graph=graph,
                                on_pass=_on_pass)
    sized = profile.graph
    platform = PlatformConfig(mb_per_vcpu=mb_per_vcpu)
    timing = TimingModel(platform)
    with _stage("%s: MPE decision (Fig. 6 elimination + Algorithm 1, lambda=%d MB/vCPU)"
               % (key, int(mb_per_vcpu))):
        slicer = partition(sized, timing, platform)
    _log("    %d layers, forward %.1f ms, whole-service footprint %.0f MB"
         % (len(sized.order), profile.forward_ns / 1e6,
            slice_footprint(sized, sized.order) / MB))
    _log("    Eq. (6)'s answer for MOPAR:\n    %s"
         % slicer.summary().replace("\n", "\n    "))

    plans: dict[str, tuple[PartitionPlan, list[int], int]] = {}
    for method in FIGURE_10:
        plan, servers = plan_for(method, sized, slicer, timing, platform)
        servers, overflow = _placement(servers)
        plans[method] = (plan, servers, overflow)

    # Methods whose plans are the same *service* share one measurement.  The
    # representative must be one that actually runs: in runs/e5_repeat.json
    # Unsplit pointed at an AlpaServe row that --methods had skipped, and the
    # dangling pointer still read ``ok: True``.
    runnable = {method for method in FIGURE_10 if methods is None or method in methods}
    first: dict[tuple, str] = {}
    for method in FIGURE_10:
        if method not in runnable:
            continue
        plan, servers, _ = plans[method]
        first.setdefault(_shape(plan, servers), method)

    whole = slice_footprint(sized, sized.order)
    rows = []
    for method in FIGURE_10:
        plan, servers, overflow = plans[method]
        row = {"key": key, "paper": profile.paper, "batch": batch, "method": method,
               "mb_per_vcpu": mb_per_vcpu, "request_mb": round(profile.size / MB, 3),
               "footprint_mb": round(whole / MB, 1),
               "forward_ms": round(profile.forward_ns / 1e6, 3),
               "analytic": _analytic(plan, platform),
               "servers": servers, "placement_overflow": overflow}
        target = first.get(_shape(plan, servers))
        if target is not None and target != method:
            _log("%s: this plan is the service %s already ran, so no second start"
                 % (method, target))
            row["measured"] = {"ok": True, "identical_to": target}
        elif method not in runnable:
            _log("%s: skipped by --methods, nothing measured" % method)
            row["measured"] = {"ok": False, "error": "--methods skipped %s" % method}
        else:
            _log("%s: Eq. (6) says %s"
                 % (method, plan.summary().replace("\n", "\n    ")))
            row["measured"] = _measure(
                key, batch, [list(s.members) for s in plan.slices],
                [s.eta for s in plan.slices], [s.alloc_mb for s in plan.slices],
                servers, platform, concurrencies, requests)
        rows.append(row)
        tail = row["measured"].get("identical_to") or (
            "p95=%.1fms @%.2frps" % (row["measured"]["windows"][-1]["p95_ms"],
                                     row["measured"]["windows"][-1]["throughput_rps"])
            if row["measured"]["ok"] else row["measured"]["error"])
        print("%-12s b=%d %-11s slices=%d etas=%-13s %s"
              % (key, batch, method, len(plan.slices),
                 ",".join(str(s.eta) for s in plan.slices), tail), flush=True)
    return rows


def _shape(plan: PartitionPlan, servers: list[int]) -> tuple:
    return ((tuple(tuple(s.members) for s in plan.slices),
             tuple(s.eta for s in plan.slices),
             tuple(round(s.alloc_mb, 3) for s in plan.slices), tuple(servers)))


def _services(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Rows indexed by service and then by method, dropping anything that did not run."""
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        if "method" in row and row["measured"].get("ok"):
            out.setdefault(row["key"], {})[row["method"]] = row
    # A service with a hole in it cannot be read against a baseline, and a table with
    # a missing bar is worse than a table that says the service did not run.
    return {key: methods for key, methods in out.items()
            if len(methods) == len(FIGURE_10)}


def _measured(services: dict, key: str, method: str) -> dict:
    """The measurement behind a method's plan; an alias points at the run it copied."""
    while "identical_to" in (measured := services[key][method]["measured"]):
        method = measured["identical_to"]
    return measured


def _window(services: dict, key: str, method: str) -> dict:
    """The widest window, which is the one Fig. 10b's bar is drawn at."""
    return _measured(services, key, method)["windows"][-1]


def _peaks(measured: dict) -> list[float]:
    """The most each slice was ever seen holding, over every window it served."""
    peaks: dict[int, float] = {}
    for window in measured.get("windows", []):
        for row in window["slices"]:
            peaks[row["index"]] = max(peaks.get(row["index"], 0.0), row["resident_mb"])
    return [peaks[index] for index in sorted(peaks)]


def _at(rows: list, index: int) -> float:
    return rows[index] if index < len(rows) else float("nan")


def _report(rows: list[dict]) -> None:
    """Print Fig. 10a, Fig. 10b, Table III and Fig. 12b from the rows this script wrote.

    Two conventions are worth stating because the paper's own axes are easy to invert:

    * memory is shown as ``MC_MOPAR / MC_method``, i.e. Eq. (6)'s memory-seconds turned
      upside down to match Fig. 10a's *utilisation* axis -- above 1 means the other
      method holds more memory per request than MOPAR does;
    * every "reduction" is ``(baseline - MOPAR) / baseline``, so positive is MOPAR better.

    The latency columns come in pairs: the analytic one is what MPE optimised under and
    the measured one is what the closed loop on this box showed, and the gap between
    them is the profiling model's error bar rather than a scheduling result.
    """
    services = _services(rows)
    keys = [k for k in dict.fromkeys(r["key"] for r in rows) if k in services]
    transformers = [k for k in keys if zoo.SPECS[k].family == "Transformer"]
    groups = [("8 non-Transformer", [k for k in keys if k not in transformers]),
              ("%d Transformer" % len(transformers), transformers)]

    def bill(key: str, method: str, field: str) -> float:
        return services[key][method]["analytic"][field]

    def utilisation(key: str, method: str) -> float:
        """``MC_MOPAR / MC_method``: Eq. (6)'s memory-seconds turned into Fig. 10a's axis."""
        mine, theirs = bill(key, "MOPAR", "mc_gb_s"), bill(key, method, "mc_gb_s")
        # A service too small for the bill to round to a byte is unbilled by every
        # method the same way, so the ratio is one rather than a division by zero.
        return mine / theirs if theirs > 0.0 else 1.0

    print("\n== Fig. 10a: MC_MOPAR / MC_method (memory-seconds inverted to utilisation)")
    print("%-22s %s" % ("service", "".join("%12s" % m for m in FIGURE_10)))
    for key in keys:
        print("%-22s %s" % (key, "".join("%12.3f" % utilisation(key, m)
                                        for m in FIGURE_10)))
    for name, group in groups:
        if group:
            print("%-22s %s" % ("mean " + name, "".join(
                "%12.3f" % (sum(utilisation(k, m) for k in group) / len(group))
                for m in FIGURE_10)))

    print("\n== Fig. 10b: measured P95 ms at the widest window, with the throughput "
          "that window actually held")
    for key in keys:
        depth = _measured(services, key, "MOPAR")["windows"][-1]["concurrency"]
        print("%-22s C=%-3d %s" % (key, depth, " ".join(
            "%s=%.0fms/%.2frps" % (m, _window(services, key, m)["p95_ms"],
                                   _window(services, key, m)["throughput_rps"])
            for m in ("MOPAR", "Uniform", "Unsplit"))))
    for name, group in groups:
        if not group:
            continue
        print("%-22s        %s" % ("reduction " + name, " ".join(
            "%s=%+.2f%%" % (m, 100.0 * sum(_window(services, k, m)["p95_ms"]
                                           - _window(services, k, "MOPAR")["p95_ms"]
                                           for k in group)
                / sum(_window(services, k, m)["p95_ms"] for k in group))
            for m in FIGURE_10 if m != "MOPAR")))

    print("\n== Table III: Eq. (6)'s bill -- cost per request in units of %g $, "
          "and Little's-law memory per 100 RPS in GB" % TABLE_III_UNIT)
    print("%-22s %9s %9s %6s %9s %9s %6s"
          % ("service", "cost M", "cost U", "x", "mem M", "mem U", "x"))
    for key in keys:
        cost, unsplit = bill(key, "MOPAR", "cost_table_iii"), \
            bill(key, "Unsplit", "cost_table_iii")
        mem, mem_u = bill(key, "MOPAR", "memory_per_100rps_gb"), \
            bill(key, "Unsplit", "memory_per_100rps_gb")
        print("%-22s %9.3f %9.3f %6s %9.1f %9.1f %6s" % (
            key, cost, unsplit,
            "%.2f" % (unsplit / cost) if cost else "-",
            mem, mem_u, "%.2f" % (mem_u / mem) if mem else "-"))
    for name, group in groups:
        if group:
            print("%-22s %9.3f %9.3f %6.2f %9.1f %9.1f %6.2f" % (
                "mean " + name,
                sum(bill(k, "MOPAR", "cost_table_iii") for k in group) / len(group),
                sum(bill(k, "Unsplit", "cost_table_iii") for k in group) / len(group),
                sum(bill(k, "Unsplit", "cost_table_iii") for k in group)
                / max(1e-12, sum(bill(k, "MOPAR", "cost_table_iii") for k in group)),
                sum(bill(k, "MOPAR", "memory_per_100rps_gb") for k in group) / len(group),
                sum(bill(k, "Unsplit", "memory_per_100rps_gb") for k in group) / len(group),
                sum(bill(k, "Unsplit", "memory_per_100rps_gb") for k in group)
                / max(1e-12, sum(bill(k, "MOPAR", "memory_per_100rps_gb")
                                 for k in group))))

    print("\n== Memory: Eq. (6)'s quota M_g against what a slice actually holds "
          "(MB; base is RSS after release(), peak is the most seen while serving)")
    for key in keys:
        quotas = services[key]["MOPAR"]["analytic"]["alloc_mb"]
        run = _measured(services, key, "MOPAR")
        bases, peaks = run.get("base_mb") or [], _peaks(run)
        print("%-22s %3d slices  quota %8.0f  base %8.0f  peak %8.0f  base/quota %5.2f"
              % (key, len(quotas), sum(quotas), sum(bases), sum(peaks),
                 sum(bases) / max(1e-9, sum(quotas))))
        for index, quota in enumerate(quotas[1:], start=1):
            print("%-22s      slice %d  quota %8.0f  base %8.0f  peak %8.0f"
                  % ("", index, quota, _at(bases, index), _at(peaks, index)))

    print("\n== Fig. 12b: the four Transformers, MOPAR vs Unsplit")
    for key in transformers:
        slicer, unsplit = services[key]["MOPAR"], services[key]["Unsplit"]
        mine, theirs = (_window(services, key, m)["p95_ms"]
                        for m in ("MOPAR", "Unsplit"))
        print("%-22s measured p95 %8.0f vs %8.0f ms = %+7.2f%% | analytic %7.1f vs "
              "%7.1f ms = %+7.2f%% | cost %.2fx" % (
                  key, mine, theirs, 100.0 * (theirs - mine) / theirs,
                  slicer["analytic"]["latency_ms"], unsplit["analytic"]["latency_ms"],
                  100.0 * (unsplit["analytic"]["latency_ms"]
                           - slicer["analytic"]["latency_ms"])
                  / unsplit["analytic"]["latency_ms"],
                  unsplit["analytic"]["cost_table_iii"]
                  / max(1e-12, slicer["analytic"]["cost_table_iii"])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="gcn1,vgg16,bert_1_3b")
    parser.add_argument("--batches", default="1")
    parser.add_argument("--concurrencies", default="1,2,4,8",
                        help="closed-loop windows to sweep inside one deployment")
    parser.add_argument("--requests", type=int, default=20,
                        help="requests per window")
    parser.add_argument("--mb-per-vcpu", type=float, default=1792.0,
                        help="lambda of Eq. (6)")
    parser.add_argument("--methods", default="",
                        help="only start a service for these methods, e.g. MOPAR,Uniform")
    parser.add_argument("--out", default="runs/e5_end_to_end.json")
    parser.add_argument("--report", metavar="JSON", default=None,
                        help="print the Fig. 10/12/Table III readings from an existing "
                             "run instead of running anything")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="say which stage is running and how long it has been at "
                             "it: the SP passes, each function process as it comes up, "
                             "each window as it finishes. The JSON is the same either "
                             "way.")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    if args.report:
        with open(args.report) as handle:
            _report(json.load(handle))
        return 0

    concurrencies = [int(v) for v in args.concurrencies.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or None
    _log("planned: %s at batch %s, windows C=%s of %d requests each, "
         "services started for %s, output %s"
         % (args.models, args.batches, args.concurrencies, args.requests,
            ",".join(methods) if methods else "all six Fig. 10 methods", args.out))
    rows = []
    for key in [k.strip() for k in args.models.split(",")]:
        for batch in [int(v) for v in args.batches.split(",")]:
            started = time.time()
            try:
                rows += run(key, batch, concurrencies, args.requests,
                            args.mb_per_vcpu, methods)
            except Exception as error:  # a service that will not profile is a row too
                print("%-12s b=%-4d FAIL %s: %s"
                      % (key, batch, type(error).__name__, error), flush=True)
                rows.append({"key": key, "batch": batch, "ok": False,
                             "error": "%s: %s" % (type(error).__name__, error)})
            print("%-12s b=%-4d %.0f s" % (key, batch, time.time() - started),
                  flush=True)
    directory = os.path.dirname(args.out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(rows, handle, indent=1)
    print("wrote %d rows to %s" % (len(rows), args.out))
    _report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
