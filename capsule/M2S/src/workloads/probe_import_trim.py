"""How much of the per-function cold start is a framework the function uses?

Each profile runs in its OWN fresh interpreter, because VmHWM is a
high-water mark: measuring two profiles in one process would report the max of
both rather than the cost of the second. The child reports its own VmHWM and
import wall time over a pipe.

Run from the repository root:
    PYTHONPATH=M2S/src ./.venv/bin/python -u -m workloads.probe_import_trim
"""
import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable

# Each entry: (label, body). ``body`` runs after the timing window opens and
# must be self-contained; the harness provides ``sys`` already configured.
PROFILES = [
    ("bare", "pass"),

    ("pipes_import", "from workloads import pipes"),

    ("payload_import", "from workloads import payload"),

    ("vit_build_1stage",
     "from workloads import pipes; pipes.build('vit_partition')"),

    ("vit_stage1_module",
     textwrap.dedent("""
        from workloads import pipes, pipe_e
        pipe_e.ensure_shards(verbose=False)
        m = pipe_e.stage_module(1)
        x = __import__('torch').randn(4, 197, 1024)
        with __import__('torch').no_grad():
            m(x)
     """)),

    ("vit_light_fn",
     textwrap.dedent("""
        from workloads import pipes, pipe_e
        p = pipes.build('vit_partition')
        p.run_ops(['vit_ingest'])
     """)),

    ("image_pipeline_build",
     "from workloads import pipes; pipes.build('pipeline_image')"),

    ("dnn_pipeline_build",
     "from workloads import pipes; pipes.build('dnn_pipeline')"),
]

CHILD = textwrap.dedent("""
    import json, os, resource, sys, time
    sys.path.insert(0, %(src)r)
    body = %(body)r
    t0 = time.perf_counter()
    exec(compile(body, "<profile>", "exec"), {"__name__": "__main__"})
    wall = (time.perf_counter() - t0) * 1000.0
    hwm = 0.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) / 1024.0
    print(json.dumps({"wall_ms": wall, "rss_mb": hwm}))
""")


def main():
    src = os.path.join(ROOT, "M2S", "src")
    rows = []
    for label, body in PROFILES:
        code = CHILD % {"src": src, "body": body}
        r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
        if r.returncode != 0:
            rows.append({"profile": label, "error": r.stderr.strip()[-400:]})
            print("  %-22s FAILED" % label)
            continue
        d = json.loads(r.stdout.strip().splitlines()[-1])
        rows.append({"profile": label, **d})
        print("  %-22s %8.1f ms   %8.1f MB" % (label, d["wall_ms"],
                                                d["rss_mb"]))

    base = next((r for r in rows if r["profile"] == "bare"), None)
    if base:
        print()
        print("  incremental over a bare interpreter:")
        for r in rows:
            if "error" in r or r["profile"] == "bare":
                continue
            print("  %-22s %8.1f ms   %8.1f MB"
                  % (r["profile"], r["wall_ms"] - base["wall_ms"],
                     r["rss_mb"] - base["rss_mb"]))

    out = os.environ.get("M2S_TRIM_OUT", "results/import_trim.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"profiles": rows}, f, indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
