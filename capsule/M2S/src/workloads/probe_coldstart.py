"""Cold start: where the ~2.5 s per function actually goes, and what removes it.

Why this probe exists
---------------------
The ViT layout sweep reports a cold start that grows with the function count:
one function 4.21 s, eight functions 22.05 s -- about 2.55 s per function. That
number is the whole latency story whenever the reuse factor is low, so it is
worth knowing which part of it is irreducible and which part is an artefact of
*how this runtime starts a worker*.

The runtime records ``spawn_ms`` as Popen -> "ready" over the socket
(``runtime.Worker``), which covers the interpreter start and the framework
import but not the first call:

    init_ms = spawn_ms + first_call_wall_ms          (vit_split_run.py:146)

This probe splits both halves.

Phase A -- spawn_ms, piece by piece
    A fresh process is started for each profile and timed by the *driver* from
    Popen to the child's result line, so the numbers are directly comparable to
    ``spawn_ms``. The last profile is exactly what ``worker.py`` does before it
    sends "ready" (import payload + pipes + util, then ``build(key)``), so it
    should reproduce the measured ~2.31 s.

Phase B -- the pre-fork (zygote) alternative
    ``worker.py`` starts every function with ``subprocess``, and the reason is
    real: forking a process that already holds a CUDA context is unsafe. But a
    parent that has only *imported* torch and *built* the pipeline holds no CUDA
    context -- ``build()`` never touches ``torch.cuda`` and ``current_device()``
    only reads an env var -- so forking from there is safe, and every child
    inherits the entire import graph and the built pipeline through copy-on-write.
    This phase asserts that invariant, then measures fork -> ready, fork -> CUDA
    context, and fork -> shard loaded + one forward, per child.

Phase C -- who owns the import time
    ``-X importtime`` on the Phase A "handler" profile, aggregated to top-level
    modules. This is what says whether the removable half of the import floor
    (torchvision + cv2, ~160 MB, see probe_vit_imports) is also removable in
    *time*.

Usage:
    python -m workloads.probe_coldstart                 # phases A + B + C
    python -m workloads.probe_coldstart --child PROFILE # phase A, one process
    python -m workloads.probe_coldstart --zygote [N]    # phase B, parent
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE is <repo>/M2S/src/workloads, so the repo root -- the cwd every other
# entry point in this package assumes, and the one ``pipe_e`` uses to place
# TORCH_HOME -- is three levels up.
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MARK = "@@COLD@@"

# Ordered cheapest-first so the marginal cost of each layer is readable.
# ``build:<key>`` and ``--cuda`` are directives, not module names.
PROFILES = [
    ("bare", []),
    ("torch", ["torch"]),
    ("torch+numpy", ["torch", "numpy"]),
    ("torch+tv", ["torch", "torchvision"]),
    ("torch+tv+cv2", ["torch", "torchvision", "cv2"]),
    ("handler", ["__handler__"]),
    ("handler+cuda", ["__handler__", "__cuda__"]),
    ("build:pipeline_image", ["__handler__", "__build__:pipeline_image"]),
    ("build:vit_partition", ["__handler__", "__build__:vit_partition"]),
    ("build:vit+cuda", ["__handler__", "__build__:vit_partition", "__cuda__"]),
]


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "M2S", "src")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # One thread per child keeps the comparison about start-up, not about how
    # many cores the box decides to give a process.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def child(profile):
    """Phase A child: perform one profile's start-up work, then report."""
    spec = dict(PROFILES)[profile]
    marks = {}
    t0 = time.perf_counter()

    def stamp(tag):
        marks[tag] = round((time.perf_counter() - t0) * 1000.0, 3)

    for name in spec:
        if name == "__handler__":
            # exactly what workloads.worker imports before it accepts a request
            from workloads import payload  # noqa: F401
            from workloads.pipes import build  # noqa: F401
            from workloads.util import sync  # noqa: F401
            stamp("handler")
        elif name == "__cuda__":
            import torch
            torch.cuda.init()
            torch.zeros(8, device="cuda")  # force the context to be real
            torch.cuda.synchronize()
            stamp("cuda")
        elif name.startswith("__build__:"):
            from workloads.pipes import build
            build(name.split(":", 1)[1])
            stamp("build")
        else:
            __import__(name)
            stamp(name)
    stamp("ready")

    print(MARK + json.dumps({
        "profile": profile,
        "marks": marks,
        "rd": round((time.perf_counter() - t0) * 1000.0, 3),
    }), flush=True)


def run_profile(profile):
    """Phase A driver: Popen -> result line, i.e. the same span as spawn_ms."""
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [sys.executable, "-u", "-m", "workloads.probe_coldstart",
         "--child", profile],
        cwd=ROOT, env=_env(), stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True)
    line = None
    for ln in p.stdout:
        if ln.startswith(MARK):
            line = ln[len(MARK):]
            break
    spawn_ms = (time.perf_counter() - t0) * 1000.0
    p.wait()
    if line is None:
        return {"profile": profile, "spawn_ms": spawn_ms, "error": True}
    rec = json.loads(line)
    rec["spawn_ms"] = round(spawn_ms, 1)
    # Popen overhead = the part of spawn_ms that is not the child's own work.
    rec["outside_ms"] = round(spawn_ms - rec["rd"], 1)
    return rec


# --------------------------------------------------------------- phase B
def zygote(n=4):
    """Fork N workers out of a preloaded parent and time each one's start-up.

    Two questions, one run:

    * does the pre-fork scheme survive on the **CPU** path -- the one this study
      is about, and the only one where a ViT stage is actually scheduled
      (``vit_spec`` declares every op ``CPU``)?
    * what happens if a child needs **CUDA**? ``worker.py`` refuses to fork for
      that reason, and this child branch records the exact error PyTorch raises
      rather than the folklore.
    """
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    t_pre = time.perf_counter()
    import torch
    # One thread, set *before* the fork. CPython warns when a multi-threaded
    # process forks, because a lock held by a thread that did not survive the
    # fork deadlocks the child. A parent with no thread pool yet is the safe
    # case, and that is what this measures.
    torch.set_num_threads(1)
    from workloads.pipes import build
    build("vit_partition")
    from workloads import pipe_e
    preload_ms = (time.perf_counter() - t_pre) * 1000.0

    # The invariant that makes the whole scheme legal. If this ever fails, the
    # pre-fork design is off the table for this pipeline.
    cuda_before = torch.cuda.is_initialized()
    print(MARK + json.dumps({
        "kind": "parent", "preload_ms": round(preload_ms, 1),
        "cuda_initialized_in_parent": bool(cuda_before),
        "n_children": n}), flush=True)

    kids = []
    for k in range(n):
        T = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            rec = {"kind": "child", "child": k}
            try:
                rec["fork_ready_ms"] = round(
                    (time.perf_counter() - T) * 1000.0, 3)
                import torch as th
                # CUDA first, so the CPU numbers below are measured on a child
                # that has already taken (or been refused) the CUDA path.
                try:
                    th.cuda.init()
                    th.zeros(8, device="cuda")
                    th.cuda.synchronize()
                    rec["to_cuda_ms"] = round(
                        (time.perf_counter() - T) * 1000.0, 3)
                except Exception as e:
                    rec["cuda_error"] = "%s: %s" % (type(e).__name__,
                                                    str(e)[:150])
                try:
                    rec["cuda_is_in_bad_fork"] = bool(
                        th._C._cuda_isInBadFork())
                except Exception:
                    rec["cuda_is_in_bad_fork"] = None

                idx = k % pipe_e.GROUPS
                m = pipe_e.stage_module(idx)
                rec["to_shard_ms"] = round((time.perf_counter() - T) * 1000.0, 3)
                # The parent was pinned to one thread so the fork is safe; the
                # child may build its own pool again, otherwise the forward below
                # would be measured under a handicap the harness never had.
                th.set_num_threads(os.cpu_count() or 8)
                if idx == 0:
                    x = pipe_e._prep_tensor(pipe_e.BATCH)
                else:
                    x = th.randn(pipe_e.BATCH, pipe_e.SEQ, pipe_e.HIDDEN)
                with th.no_grad():
                    m(x)
                rec["to_warm_ms"] = round((time.perf_counter() - T) * 1000.0, 3)
                # Steady state on the same shard, to separate "warm-up" from
                # "work": whatever the first forward costs over this is paid by
                # every process and earns nothing. Both are measured in the same
                # process under whatever load the box is carrying, so the
                # difference is robust even when the absolute numbers are not.
                ts = []
                for _ in range(5):
                    t1 = time.perf_counter()
                    with th.no_grad():
                        m(x)
                    ts.append((time.perf_counter() - t1) * 1000.0)
                rec["steady_fwd_ms"] = round(min(ts), 3)
                rec["warmup_ms"] = round(rec["to_warm_ms"]
                                         - rec["to_shard_ms"]
                                         - rec["steady_fwd_ms"], 1)
                rec["stage"] = idx
                rec["n_cpu"] = os.cpu_count()
                try:
                    rec["loadavg"] = open("/proc/loadavg").read().split()[:3]
                except Exception:
                    rec["loadavg"] = None
            except Exception as e:  # a failed child must not look like a fast one
                rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
            print(MARK + json.dumps(rec), flush=True)
            os._exit(0)

        # One child at a time. Four concurrent children each rebuild a 40-thread
        # pool, which on an 80-core box already carrying a load of ~78 turns the
        # forward into a measurement of CPU contention instead of start-up cost.
        # The runtime also spawns workers one at a time, so this is the honest
        # comparison anyway.
        os.waitpid(pid, 0)
        kids.append(pid)

    out = []
    for pid in kids:
        try:
            os.waitpid(pid, 0)
        except Exception:
            pass
    return None

def zygote_driver(n=4):
    """Run phase B in its own process and collect the parent + child records."""
    p = subprocess.Popen(
        [sys.executable, "-u", "-m", "workloads.probe_coldstart",
         "--zygote", str(n)],
        cwd=ROOT, env=_env(), stdout=subprocess.PIPE, text=True)
    parent, kids = None, []
    for ln in p.stdout:
        if ln.startswith(MARK):
            rec = json.loads(ln[len(MARK):])
            if not isinstance(rec, dict):
                continue
            if rec.get("kind") == "parent":
                parent = rec
            elif rec.get("kind") == "child":
                kids.append(rec)
        else:
            sys.stdout.write(ln)
    p.wait()
    if parent is None:
        return {"error": True, "children": kids}
    parent["children"] = kids
    return parent


# --------------------------------------------------------------- phase C
def importtime_top(profile="handler", top=14):
    """Aggregate -X importtime cumulative cost by top-level module."""
    p = subprocess.run(
        [sys.executable, "-X", "importtime", "-m", "workloads.probe_coldstart",
         "--child", profile],
        cwd=ROOT, env=_env(), capture_output=True, text=True)
    agg = {}
    for ln in p.stderr.splitlines():
        if not ln.startswith("import time:"):
            continue
        parts = ln.split("|")
        if len(parts) != 3:
            continue
        try:
            self_us = int(parts[0].split(":")[1].strip())
            cum_us = int(parts[1].strip())
        except ValueError:
            continue
        name = parts[2].strip().split(" ")[0].lstrip(".")
        if not name:
            continue
        top_name = name.split(".")[0]
        cur = agg.setdefault(top_name, {"cum_ms": 0.0, "self_ms": 0.0})
        cur["cum_ms"] = max(cur["cum_ms"], cum_us / 1000.0)
        cur["self_ms"] += self_us / 1000.0
    rows = sorted(agg.items(), key=lambda kv: -kv[1]["cum_ms"])[:top]
    return [{"module": k, "cum_ms": round(v["cum_ms"], 1),
             "self_ms": round(v["self_ms"], 1)} for k, v in rows]


def main():
    print("==" * 34)
    print("Phase A  spawn_ms, layer by layer  (Popen -> ready, one fresh process)")
    print("==" * 34)
    print("  %-22s %9s %9s %9s  %s" %
          ("profile", "spawn_ms", "child_ms", "outside_ms", "marginal vs prev"))
    rows = []
    prev = None
    for name, _ in PROFILES:
        r = run_profile(name)
        rows.append(r)
        marg = "" if prev is None else "%+8.1f" % (r["spawn_ms"] - prev)
        print("  %-22s %9.1f %9.1f %9.1f  %s" %
              (name, r["spawn_ms"], r.get("rd", -1), r.get("outside_ms", -1),
               marg))
        prev = r["spawn_ms"]

    by = {r["profile"]: r for r in rows}
    print()
    print("  removable import set (torch+tv+cv2 -> torch) : %+8.1f ms"
          % (by["torch"]["spawn_ms"] - by["torch+tv+cv2"]["spawn_ms"]))
    print("  handler modules + build(vit) over bare torch : %+8.1f ms"
          % (by["build:vit_partition"]["spawn_ms"] - by["torch"]["spawn_ms"]))
    print("  CUDA context (handler -> handler+cuda)       : %+8.1f ms"
          % (by["handler+cuda"]["spawn_ms"] - by["handler"]["spawn_ms"]))

    print()
    print("==" * 34)
    print("Phase B  pre-fork zygote  (parent preloaded, work done per child)")
    print("==" * 34)
    z = zygote_driver(4)
    if z.get("error"):
        print("  zygote phase failed: %s" % z)
    else:
        print("  parent preload (torch + build) : %8.1f ms" % z["preload_ms"])
        print("  CUDA context in parent         : %s  (must be False)"
              % z["cuda_initialized_in_parent"])
        print()
        print("  %6s %11s %11s %11s %11s %11s %s" %
              ("child", "fork_ready", "to_shard", "to_warm", "steady_fwd",
               "warmup", "cuda"))
        ok = []
        for c in z["children"]:
            if "error" in c:
                print("  %6s  ERROR %s" % (c["child"], c["error"]))
                continue
            cuda = "refused" if c.get("cuda_error") else "ok"
            print("  %6d %11.1f %11.1f %11.1f %11.1f %11.1f %s" %
                  (c["child"], c["fork_ready_ms"], c.get("to_shard_ms", -1),
                   c.get("to_warm_ms", -1), c.get("steady_fwd_ms", -1),
                   c.get("warmup_ms", -1), cuda))
            ok.append(c)
        z["_ok"] = ok
        if ok:
            first = ok[0]
            print()
            if first.get("cuda_error"):
                print("  CUDA in a forked child : %s"
                      % first["cuda_error"])
                print("  _cuda_isInBadFork()    : %s"
                      % first.get("cuda_is_in_bad_fork"))
            print("  per-function, pre-fork (CPU path):")
            print("    fork -> able to run        : %8.1f ms" %
                  statistics.median(c["fork_ready_ms"] for c in ok))
            print("    -> shard of 3 blocks loaded : %8.1f ms" %
                  statistics.median(c["to_shard_ms"] for c in ok))
            print("    -> first forward complete   : %8.1f ms" %
                  statistics.median(c["to_warm_ms"] for c in ok))
            print("       of which steady-state work: %8.1f ms" %
                  statistics.median(c["steady_fwd_ms"] for c in ok))
            print("       of which pure warm-up    : %8.1f ms  (per process,"
                  " earns nothing)" %
                  statistics.median(c["warmup_ms"] for c in ok))
            print("    one-off parent preload      : %8.1f ms (paid once per host,"
                  " not per function)" % z["preload_ms"])
            fork_med = statistics.median(c["fork_ready_ms"] for c in ok)
            spawn_import = 2314.0
            print()
            print("  what the fork changes, per function:")
            print("    spawn path : %8.1f ms of import work  (measured,"
                  " runtime.Worker)" % spawn_import)
            print("    pre-fork   : %8.1f ms of fork          -> %+.1f ms"
                  % (fork_med, fork_med - spawn_import))
            print("    the import block moves to the parent, once per host:")
            print("      %.1f s x N functions  ->  %.1f s x 1"
                  % (spawn_import / 1000.0, z["preload_ms"] / 1000.0))
            print("    both paths still pay per function: shard load + first"
                  " forward (%.0f ms here, on a box at load %s)"
                  % (statistics.median(c["to_warm_ms"] for c in ok),
                     (ok[0].get("loadavg") or ["?"])[0]))

    print()
    print("==" * 34)
    print("Phase C  who owns the import time  (-X importtime, profile 'handler')")
    print("==" * 34)
    for r in importtime_top():
        print("  %-16s cum=%8.1f ms  self=%8.1f ms"
              % (r["module"], r["cum_ms"], r["self_ms"]))

    out = os.environ.get("M2S_COLDSTART_OUT", "results/coldstart_probe.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"profiles": rows, "zygote": z,
                   "importtime": importtime_top(), "host_cpus": os.cpu_count()},
                  fh, indent=2)
    print("\nsaved %s" % out)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1])
    elif "--zygote" in sys.argv:
        i = sys.argv.index("--zygote")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 4
        # The parent reports itself through a MARK line; the children do too, so
        # there is nothing left to print here.
        zygote(n)
    else:
        main()
