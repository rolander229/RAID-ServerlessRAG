"""Local FaaS runtime: execute a partition plan as real, separate processes.

This is the ground truth the old optimizer never had. A plan is a list of
fusion groups; each group becomes its own OS process (see ``worker.py``), and
every cut between groups pays a real serialisation + object-store hop.

What is real here:
  * one OS process per function      -> isolation, per-function peak RSS
  * a real encode/decode round trip  -> wire-format bytes, not tensor.nbytes
  * a real tmpfs write/read per hop  -> bytes actually leave the process
  * a real cold start                -> spawn + first-call model load
  * a real timeout                   -> the worker is killed, not merely flagged

What is NOT real, and must be stated in any write-up:
  * the hop is tmpfs, not a network. There is no WAN RTT, no TLS, no S3.
    The measured hop cost is therefore a *lower bound* on a real deployment.
  * a memory ceiling is measured (VmHWM) and reported as a violation, not
    enforced by the kernel, because RLIMIT_AS is meaningless for CUDA
    processes which map tens of GB of virtual address space.

Two start paths, one protocol
-----------------------------
``M2S_SPAWN`` selects how a function process comes into being:

  ``subprocess`` (default)
      ``Popen`` a fresh interpreter per function. Every function pays the whole
      framework import. Measured on ViT-L/16: 2314 ms per function, of which
      ``import torch`` is 1263 ms and ``import torchvision`` 918 ms, and none of
      it is model loading (``build()`` is ~0 ms). At eight functions that is
      18.5 s of the 22.0 s cold start -- which is the *entire* reason cold start
      appears to scale with the function count.

  ``fork``
      A preloaded parent (``Zygote``) imports once and builds the pipeline once;
      each worker is forked out of it and inherits the import graph through
      copy-on-write. Measured: fork -> ready is 8.4 ms instead of 2314 ms.

Both paths drive the worker over the same Unix socket with the same messages, so
``spawn_ms`` means the same span in both (Popen->ready vs fork->ready) and the
warm numbers are unaffected by the choice. ``fork`` requires ``M2S_DEVICE=cpu``:
PyTorch refuses to initialise CUDA in a forked child, so offering fork for a GPU
plan would produce a number we could not stand behind.

Usage:
    from workloads.runtime import compile_plan, LocalFaaSRuntime
    plan = compile_plan(pipeline, [["op1","op2"], ["op3"]])
    rt = LocalFaaSRuntime()
    res = rt.invoke(plan)
"""
import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Listener
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads.pipes import Pipeline  # noqa: E402

AUTHKEY = b"m2s-runtime"


def worker_threads() -> int:
    """Threads a forked worker should run with.

    Read from the environment rather than from ``torch.get_num_threads()``,
    because the supervisor deliberately pins *itself* to one thread (see
    ``Zygote.preload``) and the worker's thread count is a property of the
    deployment, not of whoever forked it.

    Precedence: ``M2S_WORKER_THREADS`` > ``OMP_NUM_THREADS`` > cpu count.
    """
    for var in ("M2S_WORKER_THREADS", "OMP_NUM_THREADS"):
        v = (os.environ.get(var) or "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    return os.cpu_count() or 1


def pin_supervisor() -> None:
    """Take the forking process down to one thread.

    Forking out of a process whose OpenMP pool has already executed a parallel
    region is not safe: the child inherits the pool's bookkeeping but not the
    threads that would release its locks, and the child then hangs on its first
    GEMM. Measured on this box, 8 threads requested either side:

        parent ran a 1024x1024 matmul, child set to 8 threads -> never returned
        parent stayed single-threaded,   child set to 8 threads -> 12.7 ms

    The rule that follows is "the forking process must never run a parallel
    region", not "the child must be slow" -- hence this pin, paired with
    ``worker_threads()`` so that children still get the parallelism the
    deployment asks for.
    """
    try:
        import torch
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
    except Exception:
        pass
ROOT = os.environ.get("M2S_ROOT", os.getcwd())
SPOOL = os.environ.get("M2S_SPOOL", "/dev/shm/m2s-spool")

SPAWN = os.environ.get("M2S_SPAWN", "subprocess").strip().lower()
if SPAWN not in ("subprocess", "fork"):
    raise ValueError("M2S_SPAWN must be 'subprocess' or 'fork', got %r" % SPAWN)


@dataclass
class FunctionSpec:
    name: str
    ops: List[str]
    inbound: List[str] = field(default_factory=list)
    outbound: List[str] = field(default_factory=list)
    memory_mb: int = 3008
    # Per-function VRAM quota. 0 means "no GPU ceiling declared". This is a
    # separate constraint from the RSS ceiling because the two observe
    # different pools: an RSS ceiling cannot see weights that live in VRAM.
    gpu_memory_mb: int = 0
    timeout_s: float = 60.0
    device: str = "CPU"

    @property
    def n_ops(self) -> int:
        return len(self.ops)


@dataclass
class Plan:
    pipeline: str
    functions: List[FunctionSpec]

    def n_functions(self) -> int:
        return len(self.functions)

    def group_of(self, op: str) -> int:
        for i, f in enumerate(self.functions):
            if op in f.ops:
                return i
        return -1

    def is_cpu_only(self) -> bool:
        """True when no function declares a GPU.

        The pre-fork start path is gated on this: a forked child cannot use
        CUDA (PyTorch marks it ``_cuda_isInBadFork``), so a plan that schedules
        any operator on the GPU has to be started the ordinary way.
        """
        return all(f.device != "GPU" for f in self.functions)

    def render(self) -> str:
        out = []
        for i, f in enumerate(self.functions, 1):
            out.append("  f%d %-26s %-4s mem=%-5d timeout=%-4gs ops=%s"
                       % (i, f.name, f.device, f.memory_mb, f.timeout_s,
                          ",".join(f.ops)))
        return "\n".join(out)


def compile_plan(p: Pipeline, groups: List[List[str]], memory_mb: int = 3008,
                 timeout_s: float = 60.0, gpu_memory_mb: int = 0) -> Plan:
    """Turn a partition of operator names into a runnable plan.

    Inbound/outbound key sets are derived from the operators' declared I/O, so
    the plan says exactly which values cross each boundary.
    """
    flat = [n for g in groups for n in g]
    assert sorted(flat) == sorted(p.names()), \
        "plan must cover every operator exactly once"

    funcs: List[FunctionSpec] = []
    produced_before: set = set()
    for gi, g in enumerate(groups):
        have: set = set()
        inbound: List[str] = []
        writes: List[str] = []
        for n in g:
            o = p.by_name(n)
            for r in o.reads:
                if r not in have and r in produced_before and r not in inbound:
                    inbound.append(r)
            for w in o.writes:
                have.add(w)
                if w not in writes:
                    writes.append(w)
        devs = {p.by_name(n).device for n in g}
        funcs.append(FunctionSpec(
            name=("%s-%s" % (g[0], "fused" if len(g) > 1 else "fn")),
            ops=list(g),
            inbound=sorted(inbound),
            outbound=[],
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            timeout_s=timeout_s,
            device="GPU" if "GPU" in devs else "CPU",
        ))
        produced_before |= have

    for i, f in enumerate(funcs):
        later: set = set()
        for g2 in funcs[i + 1:]:
            later |= set(g2.inbound)
        f.outbound = sorted(set(_writes_of(p, f.ops)) & later)
    return Plan(pipeline=p.key, functions=funcs)


def _writes_of(p: Pipeline, ops: List[str]) -> List[str]:
    out: List[str] = []
    for n in ops:
        for w in p.by_name(n).writes:
            if w not in out:
                out.append(w)
    return out


def _logpath(key: str, label: str, logdir: Optional[str] = None) -> str:
    logdir = logdir or os.path.join(ROOT, "workloads_data", "workerlogs")
    os.makedirs(logdir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return os.path.join(
        logdir, "%s-%s-%d.log"
        % (key, safe[:40], int(time.time() * 1000) % 10 ** 8))


class _BaseWorker:
    """The driver side of the worker protocol.

    Both start paths end up with a connection to a process that speaks the
    messages documented in ``worker.py``, so the calling, logging and teardown
    logic lives here once.
    """

    def __init__(self):
        self.n_calls = 0
        self.spawn_ms = 0.0
        # Set by ForkedWorker to the parent's file-backed pages the kernel does
        # not charge the child for; 0 for a process started from scratch.
        self.shared_file_mb = 0.0

    def call(self, ops, inbound, outbound, timeout_s=60.0) -> dict:
        self.conn.send({"ops": ops, "inbound": inbound, "outbound": outbound})
        if not self.conn.poll(timeout_s):
            self.kill()
            raise TimeoutError("function %s exceeded %.1fs"
                               % (self.key, timeout_s))
        self.n_calls += 1
        try:
            return self.conn.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError):
            # A worker that dies mid-send leaves the socket reset rather than
            # cleanly closed, and recv() reports that as ConnectionResetError.
            # Catching only EOFError turns a diagnosable crash into a bare
            # traceback with no function name and no worker log.
            self.kill()
            raise RuntimeError("function %s died. log tail:\n%s"
                               % (self.key, self.log_tail()))

    def log_tail(self, n=2500) -> str:
        try:
            self.log.flush()
            with open(self.log_path, "rb") as fh:
                return fh.read()[-n:].decode("utf-8", "replace")
        except Exception as e:
            return "<no log: %s>" % e

    def close(self):
        try:
            self.conn.send(None)
            self.conn.close()
        except Exception:
            pass
        self.kill()
        try:
            self.listener.close()
        except Exception:
            pass

    def kill(self):
        raise NotImplementedError


class Worker(_BaseWorker):
    """A warm serverless function process started with ``subprocess``."""

    def __init__(self, key: str, label: str = "fn", rlimit_mb: int = 0,
                 log=None, logdir: Optional[str] = None):
        super().__init__()
        self.key = key
        self.label = label
        self.dir = tempfile.mkdtemp(prefix="m2s-")
        self.addr = os.path.join(self.dir, "s.sock")
        self.listener = Listener(self.addr, family="AF_UNIX", authkey=AUTHKEY)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "M2S", "src")
        env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self.log_path = _logpath(key, label, logdir)
        self.log = log or open(self.log_path, "wb")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "workloads.worker", key, self.addr,
             "--label", label, "--mem-mb", str(rlimit_mb)],
            cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.conn = self.listener.accept()
        tag, self.pid = self.conn.recv()
        self.spawn_ms = (time.perf_counter() - t0) * 1000.0

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass


class Zygote:
    """A preloaded parent whose children become workers by ``fork``.

    Why this exists
    ---------------
    Every function otherwise re-imports the framework stack. That is a fixed
    cost per function and it dominates the cold start of a split layout: 2314 ms
    out of 2756 ms per function on ViT-L/16, so eight functions pay 18.5 s that
    a single function pays once. Nothing about partitioning *requires* that; it
    is an artefact of starting each function from a bare interpreter.

    Importing once and forking is the standard answer (a forkserver, a warm
    pool, or -- in a managed platform -- a snapshot restore). What it buys here
    is measured in ``probe_coldstart.py``: fork -> able to serve is 8.4 ms.

    Constraints, both real
    ----------------------
    * The parent must be single-threaded at fork time. A lock held by a thread
      that does not survive the fork deadlocks the child, so ``torch``'s thread
      pool is pinned to one before the first fork and each child restores the
      thread count the subprocess path would have run with.
    * CUDA is unavailable to a forked child: PyTorch raises "Cannot
      re-initialize CUDA in forked subprocess" and sets ``_cuda_isInBadFork()``.
      ``fork`` mode is therefore refused unless ``M2S_DEVICE=cpu``, and a GPU
      plan falls back to ``subprocess`` (recorded in the result as
      ``spawn_fallback``, never silently).
    """

    _pool: Dict[str, "Zygote"] = {}

    def __init__(self, key: str):
        self.key = key
        self.pipeline = None
        self.preload_ms = 0.0
        self.worker_threads = 0
        self.dir = None
        self.addr = None
        self.listener = None

    @classmethod
    def get(cls, key: str) -> "Zygote":
        """One preloaded parent per pipeline, shared by every runtime in the
        process. That is the deployment being modelled: a warm host that has
        already paid the framework import, serving many functions."""
        if key not in cls._pool:
            z = cls(key)
            z.preload()
            cls._pool[key] = z
        return cls._pool[key]

    @classmethod
    def reset(cls):
        for z in cls._pool.values():
            try:
                z.listener.close()
            except Exception:
                pass
        cls._pool.clear()

    def preload(self):
        import torch
        from workloads.pipes import build

        # The thread count a child should run with. Deliberately *not* read
        # back from ``torch.get_num_threads()``: the supervisor pins itself to
        # one thread below, and the worker's parallelism is a deployment
        # parameter that must not inherit that pin.
        self.worker_threads = worker_threads()

        # The one-off import the whole class exists to avoid repeating. Measured
        # by probe_coldstart.py at ~2331 ms when it is not already paid; when the
        # driver has imported the framework itself this is only build(), which
        # the same probe shows is ~0 ms.
        t0 = time.perf_counter()
        pin_supervisor()
        self.pipeline = build(self.key)

        # One listener for the whole pool. A listener per worker would leak a
        # socket pair into every later child, and a 13-layout sweep forks
        # hundreds of them.
        self.dir = tempfile.mkdtemp(prefix="m2s-zyg-")
        self.addr = os.path.join(self.dir, "s.sock")
        self.listener = Listener(self.addr, family="AF_UNIX", authkey=AUTHKEY)
        self.preload_ms = (time.perf_counter() - t0) * 1000.0

    def fork_worker(self, label: str) -> "ForkedWorker":
        return ForkedWorker(self, label)

    def shared_file_mb(self) -> float:
        """File-backed resident MB this parent is about to share with a child.

        Needed because ``/proc/self/status`` does not charge a forked child for
        the parent's file-backed pages: measured on ViT-L/16, a child reports
        VmHWM 537.5 MB right after a forward where the same work in a fresh
        process peaks at 775.4 MB, and the whole 238 MB gap is ``RssFile``.

        The gap is an accounting artefact, not a saving. A deployed function
        runs in its own container, and a container is charged for the framework
        pages it maps. So the billed number below adds this back, and the raw
        kernel view is kept alongside as ``rss_private_mb`` so a reader can undo
        the correction.
        """
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("RssFile:"):
                        return int(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0


class ForkedWorker(_BaseWorker):
    """A warm serverless function process forked out of a ``Zygote``.

    ``spawn_ms`` spans fork -> ready, which is the same span ``Worker`` measures
    (Popen -> ready). That is what makes the two start paths comparable rather
    than merely different.
    """

    def __init__(self, zyg: Zygote, label: str = "fn",
                 logdir: Optional[str] = None):
        super().__init__()
        self.key = zyg.key
        self.label = label
        # What the kernel will *not* attribute to this child even though a
        # deployed container would be charged for it. See Zygote.shared_file_mb.
        self.shared_file_mb = zyg.shared_file_mb()
        self.dir = tempfile.mkdtemp(prefix="m2s-")
        addr = os.path.join(self.dir, "s.sock")
        self.listener = Listener(addr, family="AF_UNIX", authkey=AUTHKEY)
        self.log_path = _logpath(zyg.key, label, logdir)
        self.log = open(self.log_path, "wb")

        # A sweep forks hundreds of children and each inherits every descriptor
        # the parent holds open, so a soft limit of 1024 runs out. The hard
        # limit allows more, so this needs no privilege.
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < hard:
                resource.setrlimit(resource.RLIMIT_NOFILE,
                                   (min(hard, 65536), hard))
        except Exception:
            pass

        sys.stdout.flush()
        sys.stderr.flush()
        t0 = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.dup2(self.log.fileno(), 1)
                os.dup2(self.log.fileno(), 2)
                import torch
                if zyg.worker_threads:
                    torch.set_num_threads(zyg.worker_threads)
                from workloads import worker as W
                W._FORKED = True
                conn = W.connect(addr)
                W.loop(zyg.pipeline, conn, mem_mb=0, label=label)
                code = 0
            except BaseException:
                traceback.print_exc()
            finally:
                # _exit, not sys.exit: the child inherited the parent's
                # listeners and must not run their cleanup, which would unlink
                # socket paths the parent and its other children still use.
                os._exit(code)
        self.pid = pid
        self.conn = self.listener.accept()
        tag, self.pid = self.conn.recv()
        self.spawn_ms = (time.perf_counter() - t0) * 1000.0

    def kill(self):
        try:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
        except Exception:
            pass


class LocalFaaSRuntime:
    def __init__(self, root: Optional[str] = None, spool: Optional[str] = None,
                 spawn: Optional[str] = None):
        self.root = root or ROOT
        self.spool = spool or SPOOL
        os.makedirs(self.spool, exist_ok=True)
        self.spawn = spawn or SPAWN
        if self.spawn == "fork":
            forced = os.environ.get("M2S_DEVICE", "").strip().lower()
            if forced != "cpu":
                raise RuntimeError(
                    "M2S_SPAWN=fork requires M2S_DEVICE=cpu: PyTorch refuses to "
                    "initialise CUDA in a forked child, so a fork-mode run has "
                    "to be CPU-only to mean anything. Set both, or leave "
                    "M2S_SPAWN unset.")
        self.workers: Dict[str, Worker] = {}
        self.cold_start_ms: Dict[str, float] = {}
        self.zygote: Optional[Zygote] = None
        self.spawn_fallback: Optional[str] = None
        self.spawn_path: Optional[str] = None

    def _worker(self, plan: Plan, f: FunctionSpec,
                rlimit_mb: int) -> Worker:
        if f.name in self.workers:
            return self.workers[f.name]
        if self.spawn == "fork":
            if not plan.is_cpu_only():
                # Loud, recorded, and never silent: a fallback that produced a
                # different kind of number without saying so would be worse than
                # refusing to run.
                self.spawn_fallback = (
                    "plan %s schedules GPU operators; a forked child cannot use "
                    "CUDA, so this plan was started with subprocess"
                    % plan.pipeline)
            else:
                if self.zygote is None:
                    self.zygote = Zygote.get(plan.pipeline)
                w = self.zygote.fork_worker(label=f.name)
                self.spawn_path = "fork"
                self.workers[f.name] = w
                self.cold_start_ms[f.name] = w.spawn_ms
                return w
        w = Worker(plan.pipeline, label=f.name, rlimit_mb=rlimit_mb)
        self.spawn_path = self.spawn_path or "subprocess"
        self.workers[f.name] = w
        self.cold_start_ms[f.name] = w.spawn_ms
        return w

    def invoke(self, plan: Plan, rlimit_mb: int = 0,
               timeout_s: float = 60.0,
               capture: Optional[List[str]] = None) -> dict:
        """One end-to-end request through the function graph.

        ``rlimit_mb`` defaults to 0 = do not touch RLIMIT_AS. Leave it at 0
        unless you know what you are doing: a CUDA process maps tens of GB of
        virtual address space, so capping RLIMIT_AS kills it outright. The
        declared per-function ceiling lives in ``FunctionSpec.memory_mb`` and is
        checked against the measured ``rss_mb`` instead.

        ``capture`` names intermediate values that the driver wants back, so a
        partitioned run can be checked against a monolithic reference. It is
        implemented by *appending* those keys to the last function's outbound
        set, which means the function performs an extra encode. Use it on a
        dedicated accuracy pass, never on a timing pass -- the added encode is
        real work and would contaminate the latency numbers.

        A monolith plan has an empty outbound set (nothing downstream needs its
        values), so without ``capture`` its results never leave the worker. That
        is intentional: a deployed monolith would not pay that encode either.
        """
        wire: Dict[str, bytes] = {}
        rows: List[dict] = []
        t_req = time.perf_counter()

        if capture and plan.functions:
            last = plan.functions[-1]
            last.outbound = sorted(set(last.outbound) | set(capture))

        for f in plan.functions:
            w = self._worker(plan, f, rlimit_mb)

            hop0 = time.perf_counter()
            inbound = {}
            hop_bytes = 0
            for k in f.inbound:
                buf = wire.get(k, b"")
                p = os.path.join(self.spool, "%s.%s.bin" % (f.name, k))
                with open(p, "wb") as fh:
                    fh.write(buf)
                with open(p, "rb") as fh:
                    inbound[k] = fh.read()
                hop_bytes += len(buf)
            hop_ms = (time.perf_counter() - hop0) * 1000.0

            t0 = time.perf_counter()
            try:
                r = w.call(f.ops, inbound, f.outbound, timeout_s)
            except TimeoutError as e:
                return {"pipeline": plan.pipeline, "timeout": str(e),
                        "rows": rows, "n_functions": plan.n_functions()}
            r["wall_ms"] = (time.perf_counter() - t0) * 1000.0
            for k, buf in r["out"].items():
                wire[k] = buf

            rows.append({
                "function": f.name,
                "device": f.device,
                "n_ops": f.n_ops,
                "ops": f.ops,
                "pid": w.pid,
                "compute_ms": round(r["compute_ms"], 3),
                "decode_ms": round(r["decode_ms"], 3),
                "encode_ms": round(r["encode_ms"], 3),
                "wall_ms": round(r["wall_ms"], 3),
                "in_mb": round(hop_bytes / 1048576.0, 4),
                "out_mb": round(sum(len(b) for b in r["out"].values())
                                / 1048576.0, 4),
                "hop_ms": round(hop_ms, 3),
                # What the function is charged for: the kernel's view plus the
                # framework pages it shares with the rest of the host. Equal to
                # the kernel's view in subprocess mode, where there is nothing
                # shared to add back.
                "rss_mb": round(r["rss_mb"] + w.shared_file_mb, 2),
                "rss_private_mb": r["rss_mb"],
                "shared_file_mb": round(w.shared_file_mb, 2),
                "gpu_mb": r.get("gpu_mb", 0.0),
                "gpu_reserved_mb": r.get("gpu_reserved_mb", 0.0),
                "mem_ceiling_mb": f.memory_mb,
                "mem_ok": (r["rss_mb"] + w.shared_file_mb) <= f.memory_mb,
                "gpu_ceiling_mb": f.gpu_memory_mb,
                "mem_ok_gpu": (f.gpu_memory_mb <= 0
                               or r.get("gpu_mb", 0.0) <= f.gpu_memory_mb),
                "cold_start_ms": round(self.cold_start_ms.get(f.name, 0.0), 1),
            })

        total_ms = (time.perf_counter() - t_req) * 1000.0
        res = {
            "pipeline": plan.pipeline,
            "n_functions": plan.n_functions(),
            "e2e_ms": round(total_ms, 3),
            "rows": rows,
            "wire_total_mb": round(sum(len(b) for b in wire.values())
                                   / 1048576.0, 4),
            "spawn_mode": self.spawn_path or self.spawn,
        }
        adj = max([r["shared_file_mb"] for r in rows], default=0.0)
        if adj:
            res["rss_correction_mb"] = round(adj, 2)
            res["rss_correction_note"] = (
                "rss_mb includes rss_correction_mb of framework file-backed pages "
                "shared with the pre-fork parent, which /proc does not charge to "
                "the child but a deployed container would be charged for")
        if self.zygote is not None:
            res["zygote_shared_file_mb"] = round(self.zygote.shared_file_mb(), 2)
        if self.spawn_fallback:
            res["spawn_fallback"] = self.spawn_fallback
        if self.zygote is not None:
            res["zygote_preload_ms"] = round(self.zygote.preload_ms, 1)
        if capture:
            from workloads import payload as P
            res["captured"] = {k: P.decode(wire[k]) for k in capture
                               if k in wire}
        return res

    def shutdown(self):
        for w in self.workers.values():
            w.close()
        self.workers.clear()
