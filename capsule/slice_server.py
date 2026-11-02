"""One HTTP service per slice: the bare-metal ancestor of the K8s slice-runner.

The K8s plan (slice-runner-k8s-plan.md §1) wants "每片一个网络服务" without
rewriting a line of the executor. This module is that shell: a process loads its
slice exactly the way :func:`slicer.faas.runtime._run_slice` does -- record, cut,
``release`` every weight its slice does not read -- and then serves
``POST /invoke`` with :class:`std.http.server` instead of ``/dev/shm`` lanes.
Every request replays :meth:`~slicer.faas.program.SliceProgram.run`, so the
numbers a K8s chain would produce are the numbers this chain produces.

Wire contract (slice-runner-k8s-plan.md §1.2):

- body: base64(torch.save) payload, or ``{"storage_ref": path}`` once the tensor
  bytes pass 2 MB (the dual-track rule; the file lands under ``runs/activations``);
- headers: ``X-Mopar-Request-Id`` / ``X-Mopar-Hop`` ride the whole chain;
- a slice runs its steps, then POSTs its outbound tensors to ``--next-peer``;
  the last slice answers with the service's final output, and every slice
  upstream relays that answer back verbatim, so the caller of slice 0 receives
  what the monolith would have returned;
- downstream calls time out at 30 s; any failure answers 5xx with
  ``{"failed_hop": i}`` naming the hop that broke;
- each request appends one JSON line to ``runs/slice_server_logs/``:
  ``{"slice_id", "request_id", "hop_ms", "moved_mb", "rss_mb", "ts"}``.

Linear chains only, on purpose: the plan's slices talk to consecutive peers, so
a slice with more than one inbound or outbound peer refuses to start rather than
silently drop a lane. (That refusal is also what makes "一层划错片" observable --
see tests/test_metrics_negative_controls.py.)

Run:  python -m capsule.slice_server --model resnet50 --slice 0 --port 8080 \
          --next-peer http://127.0.0.1:8081/invoke
Probe the whole chain from one command:
      python -m capsule.slice_server --model resnet50 --probe http://127.0.0.1:8080/invoke
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import torch

# capsule is a top-level package in this repo; slicer lives beside it, which is
# why the imports here are absolute (the same pattern capsule/bridge.py uses).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import _iter_tensors, move_tensors  # noqa: E402
from slicer.faas.program import _fill, record  # noqa: E402
from slicer.models.weight_shard import expand_parts, load_shard  # noqa: E402

MB = 2 ** 20
INLINE_LIMIT = 2 * MB          # the dual-track threshold: base64 below, file above
DOWNSTREAM_TIMEOUT_S = 30.0    # the SAM template's Timeout: 30, enforced by the caller
RESULT_KIND, HOPS_KIND, REQUEST_KIND = "result", "hops", "request"


def _rss_mb() -> float:
    """VmHWM in MB: the resident peak this process has ever reached."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _encode(tensors: list[torch.Tensor], kind: str, keys: list[int] | None,
            request_id: str, hop: int, activations_dir: Path) -> tuple[bytes, str, float]:
    """Serialise tensors as base64, or as a file once they pass the inline limit.

    Returns (body, content_type, moved_bytes); moved_bytes counts the raw tensor
    payload, before base64's ~33% -- the quantity `moved_mb` has always meant.
    """
    buf = BytesIO()
    torch.save({"kind": kind, "keys": keys, "tensors": tensors}, buf)
    raw = buf.getvalue()
    if len(raw) <= INLINE_LIMIT:
        return base64.b64encode(raw), "text/plain", float(len(raw))
    out = activations_dir / request_id / ("hop_%d.pt" % hop)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    body = json.dumps({"storage_ref": str(out.resolve()), "kind": kind,
                       "keys": keys}).encode()
    return body, "application/json", float(len(raw))


def _decode(body: bytes, content_type: str) -> dict:
    """The inverse of :func:`_encode`: a payload dict with ``tensors`` in it."""
    if content_type.startswith("application/json"):
        meta = json.loads(body)
        raw = Path(meta["storage_ref"]).read_bytes()
        keys = meta.get("keys")
        kind = meta.get("kind", HOPS_KIND)
    else:
        raw = base64.b64decode(body)
        keys = kind = None
    blob = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "tensors" not in blob:
        raise ValueError("payload is not a slice_runner envelope")
    return {"kind": blob.get("kind", kind), "keys": blob.get("keys", keys),
            "tensors": list(blob["tensors"]), "bytes": float(len(raw))}


class _SliceState:
    """Everything one slice's process needs to serve requests; built once at boot."""

    def __init__(self, args) -> None:
        self.args = args
        model, inputs = zoo.load(args.model, device=args.device,
                                 batch=args.batch, seed=args.seed)
        self.model, self.inputs = model, inputs
        self.program = record(model, inputs, name=args.model, device=args.device)
        # The plan comes from the shard manifest when one exists (A1's output is
        # the single source of truth), otherwise the DP re-derives it.
        parts = None
        for candidate in filter(None, [args.plan,
                                       Path(args.weights_dir or
                                            (REPO_ROOT / "runs" / "weights" / args.model))
                                       / "shard_manifest.json"]):
            if Path(candidate).is_file():
                parts = json.loads(Path(candidate).read_text())["expanded_parts"]
                print("[slice_server] plan from %s" % candidate)
                break
        if parts is None:
            parts = expand_parts(self.program, args.slices)
            print("[slice_server] plan re-derived by vertical_partition(slices=%d)" % args.slices)
        cases = self.program.slices(parts)
        self.case = cases[args.slice]
        # Linear-chain contract: one producer, one consumer. A slice that would
        # have to fan out has no --next-peer to give the extra lane to.
        if len(self.case.inbound) > 1 or len(self.case.outbound) > 1:
            raise SystemExit("slice %d is not linear: inbound=%s outbound=%s -- a "
                             "HTTP chain needs consecutive peers" %
                             (args.slice, [p for p, _ in self.case.inbound],
                              [p for p, _ in self.case.outbound]))
        if self.case.outbound and not args.next_peer:
            raise SystemExit("slice %d ships %d values downstream but --next-peer is unset"
                             % (args.slice, len(self.case.outbound[0][1])))
        # Quota semantics as in runtime._run_slice: drop the capture that pins
        # every slice's weights, then free the weights this slice never reads.
        # The shard loads BEFORE the release: release() nulls table entries the
        # steps never read (BatchNorm's num_batches_tracked among them), and a
        # shard loaded afterwards would find those names gone. Loading first
        # makes the shard files the source of the executed weights, then the
        # release prunes whatever the slice still does not need.
        weights_dir = Path(args.weights_dir or (REPO_ROOT / "runs" / "weights" / args.model))
        shard = weights_dir / ("slice_%d.pt" % args.slice)
        self.shard_tensors = 0
        if shard.is_file():
            self.shard_tensors = len(load_shard(self.model, shard))
            print("[slice_server] weights loaded from %s (%d tensors)" % (shard, self.shard_tensors))
        else:
            print("[slice_server] no shard at %s; running on the zoo-built weights" % shard)
        self.program.steps = ()
        self.freed_mb = self.program.release(self.case.keeps) / MB
        self.in_keys = tuple(self.case.inbound[0][1]) if self.case.inbound else ()
        self.out_keys = tuple(self.case.outbound[0][1]) if self.case.outbound else ()
        self.activations_dir = Path(args.activations_dir or REPO_ROOT / "runs" / "activations")
        self.log_path = Path(args.log_dir or REPO_ROOT / "runs" / "slice_server_logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / ("%s_slice%d.jsonl" % (args.model, args.slice))
        print("[slice_server] slice %d: %d layers, %d steps, freed %.1f MB, "
              "in=%s out=%s -> %s" %
              (args.slice, len(self.case.layers), len(self.case.steps), self.freed_mb,
               len(self.in_keys), len(self.out_keys), args.next_peer))

    def log(self, row: dict) -> None:
        row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run_steps(self, payload: dict) -> list[torch.Tensor]:
        """Replay this slice's steps over one request's environment."""
        env: dict[int, torch.Tensor] = {}
        if self.case.index == 0 or payload["kind"] == REQUEST_KIND:
            env.update(dict(zip(self.program.inputs,
                                _iter_tensors(move_tensors(payload["tensors"],
                                                           self.args.device)))))
        else:
            if len(payload["tensors"]) != len(self.in_keys):
                raise ValueError("hop payload carries %d tensors, slice expects %d"
                                 % (len(payload["tensors"]), len(self.in_keys)))
            env.update(dict(zip(self.in_keys, payload["tensors"])))
        self.case.run(env)
        return [env[key] for key in self.out_keys] if self.out_keys else \
            list(_iter_tensors(_fill(self.program.template, env)))


class _Handler(BaseHTTPRequestHandler):
    state: _SliceState  # injected by serve(); shared, read-only after boot
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # default logging would fight the JSONL
        pass

    def _answer(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802  (http.server API)
        if self.path != "/healthz":
            self._answer(404, b'{"error": "not found"}', "application/json")
            return
        body = json.dumps({"model": self.state.args.model,
                           "slice_id": self.state.args.slice,
                           "layers": len(self.state.case.layers),
                           "steps": len(self.state.case.steps),
                           "rss_hwm_mb": round(_rss_mb(), 1)}).encode()
        self._answer(200, body, "application/json")

    def do_POST(self):  # noqa: N802
        if self.path != "/invoke":
            self._answer(404, b'{"error": "not found"}', "application/json")
            return
        state = self.state
        t0 = time.perf_counter()
        request_id = self.headers.get("X-Mopar-Request-Id") or str(uuid.uuid4())
        try:
            hop = int(self.headers.get("X-Mopar-Hop", state.args.slice))
        except ValueError:
            hop = state.args.slice
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        row = {"slice_id": state.args.slice, "request_id": request_id, "hop": hop}
        try:
            payload = _decode(body, self.headers.get("Content-Type", "text/plain"))
            row["moved_in_mb"] = round(payload["bytes"] / MB, 3)
            t1 = time.perf_counter()
            outputs = state.run_steps(payload)
            busy_ms = (time.perf_counter() - t1) * 1e3
            if state.out_keys and state.args.next_peer:
                answer = self._forward(state, request_id, outputs)
            else:
                answer = self._finish(state, request_id, outputs)
            row.update({"hop_ms": round((time.perf_counter() - t0) * 1e3, 3),
                        "busy_ms": round(busy_ms, 3),
                        "moved_mb": round(answer["moved_mb"] / MB, 3),
                        "rss_mb": round(_rss_mb(), 1), "status": "ok"})
            state.log(row)
            self._answer(answer["status"], answer["body"], answer["content_type"])
        except Exception as exc:  # noqa: BLE001  (any hop failure is a 5xx)
            failed = int(_failed_hop(exc) or state.args.slice)
            row.update({"hop_ms": round((time.perf_counter() - t0) * 1e3, 3),
                        "rss_mb": round(_rss_mb(), 1), "status": "error",
                        "failed_hop": failed, "error": "%s: %s" % (type(exc).__name__, exc)})
            state.log(row)
            body = json.dumps({"failed_hop": failed,
                               "error": "%s: %s" % (type(exc).__name__, exc)}).encode()
            self._answer(500, body, "application/json")
            traceback.print_exc()

    def _forward(self, state: _SliceState, request_id: str,
                 outputs: list[torch.Tensor]) -> dict:
        """Ship the outbound tensors to --next-peer; relay its answer upstream.

        The relay is what lets the caller of slice 0 hold the service's real
        output: the last slice answers with the result and every hop between
        passes that answer back untouched.
        """
        body, content_type, moved = _encode(outputs, HOPS_KIND, list(state.out_keys),
                                            request_id, state.args.slice,
                                            state.activations_dir)
        request = urllib.request.Request(
            state.args.next_peer, data=body, method="POST",
            headers={"Content-Type": content_type,
                     "X-Mopar-Request-Id": request_id,
                     "X-Mopar-Hop": str(state.args.slice + 1)})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=DOWNSTREAM_TIMEOUT_S) as resp:
                answer = resp.read()
                status, content_type = resp.status, resp.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raise DownstreamError(state.args.slice + 1,
                                  "peer answered %d: %s" % (exc.code, exc.read()[:200])) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DownstreamError(state.args.slice + 1,
                                  "peer unreachable in %.0fs: %s"
                                  % (DOWNSTREAM_TIMEOUT_S, exc)) from exc
        state.last_downstream_ms = (time.perf_counter() - t0) * 1e3
        return {"status": status, "body": answer, "content_type": content_type,
                "moved_mb": moved}

    def _finish(self, state: _SliceState, request_id: str,
                outputs: list[torch.Tensor]) -> dict:
        body, content_type, moved = _encode(outputs, RESULT_KIND, None,
                                            request_id, state.args.slice,
                                            state.activations_dir)
        return {"status": 200, "body": body, "content_type": content_type,
                "moved_mb": moved}


class DownstreamError(Exception):
    """A hop downstream of this slice failed; the 5xx names *that* hop."""

    def __init__(self, hop: int, message: str) -> None:
        super().__init__(message)
        self.hop = hop


def _failed_hop(exc: BaseException) -> int | None:
    return getattr(exc, "hop", None)


def probe(entry: str, args) -> int:
    """Client side of the acceptance: one request through the chain vs the monolith."""
    model, inputs = zoo.load(args.model, device=args.device, batch=args.batch, seed=args.seed)
    with torch.no_grad():
        monolith = model(*inputs)
    body, content_type, _moved = _encode(list(_iter_tensors(inputs)), REQUEST_KIND, None,
                                         "probe-%s" % uuid.uuid4().hex[:8], 0,
                                         Path(args.activations_dir or REPO_ROOT / "runs" / "activations"))
    request = urllib.request.Request(entry, data=body, method="POST",
                                     headers={"Content-Type": content_type,
                                              "X-Mopar-Request-Id": "probe-%s" % uuid.uuid4().hex[:8],
                                              "X-Mopar-Hop": "0"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=DOWNSTREAM_TIMEOUT_S * 2) as resp:
        content_type = resp.headers.get("Content-Type", "")
        payload = _decode(resp.read(), content_type)
    wall_ms = (time.perf_counter() - t0) * 1e3
    diffs = [(a.float() - b.float()).abs().max().item()
             for a, b in zip(_iter_tensors(monolith), payload["tensors"])]
    max_diff = max(diffs) if diffs else 0.0
    print("[probe] transport=%s outputs=%d wall=%.1fms" %
          ("file" if content_type.startswith("application/json") else "inline",
           len(payload["tensors"]), wall_ms))
    print("[probe] chained vs monolith max-diff = %.3e (tolerance 1e-4) -> %s"
          % (max_diff, "PASS" if max_diff <= 1e-4 else "FAIL"))
    return 0 if max_diff <= 1e-4 else 1


def serve(args) -> int:  # noqa: ANN001 (argparse namespace)
    state = _SliceState(args)
    handler = type("_BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    server.daemon_threads = True
    print("[slice_server] listening on %s:%d (pid %d)"
          % (args.bind, args.port, __import__("os").getpid()), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve one model slice over HTTP.")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--next-peer", default=None, help="URL of the next slice's /invoke")
    parser.add_argument("--slices", type=int, default=3, help="slice count when no plan file exists")
    parser.add_argument("--plan", default=None, help="shard_manifest.json / plan JSON to take expanded_parts from")
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe", default=None, metavar="ENTRY_URL",
                        help="do not serve: send one request through the chain and diff it")
    args = parser.parse_args(argv)
    if args.probe:
        return probe(args.probe, args)
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
