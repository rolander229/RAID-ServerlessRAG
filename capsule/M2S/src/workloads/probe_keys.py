"""Per-state-key wire cost table: the optimizer's input.

The solver has to decide whether cutting a given edge is worth it. That
decision needs, for every value that could cross a boundary, its real encoded
size and real serialisation cost -- not a fabricated ms/MB constant.

This replays each pipeline operator by operator in one process and meters every
state key each operator writes.

Run:  cd <root> && PYTHONPATH=M2S/src HF_ENDPOINT=https://hf-mirror.com \
        .venv/bin/python -m workloads.probe_keys
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workloads import payload as P  # noqa: E402
from workloads.pipes import build, keys  # noqa: E402


def main():
    out = {}
    for k in keys():
        p = build(k)
        rows = []
        for op in p.ops:
            try:
                p.run_ops([op.name])
            except Exception as e:
                rows.append({"op": op.name, "error": "%s: %s"
                             % (type(e).__name__, str(e)[:120])})
                continue
            for key in op.writes:
                if key not in p.state:
                    rows.append({"op": op.name, "key": key,
                                 "error": "declared write missing"})
                    continue
                m = P.metered(p.state[key])
                rows.append({
                    "op": op.name, "key": key, "device": op.device,
                    **m,
                })
        out[k] = rows
        print("=" * 70)
        print(k)
        for r in rows:
            if "error" in r:
                print("  %-32s %-10s ERROR %s"
                      % (r["op"], r.get("key", "-"), r["error"][:70]))
                continue
            print("  %-32s %-10s raw %8.4f -> wire %8.4f MB  %-8s "
                  "serde %7.3f ms  rt=%s"
                  % (r["op"], r["key"], r["raw_mb"], r["wire_mb"], r["kind"],
                     r["serde_ms"], r["roundtrip_ok"]))

    os.makedirs("results", exist_ok=True)
    path = "results/op_key_costs.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    n = sum(len(v) for v in out.values())
    print("\nsaved %s (%d records)" % (path, n))


if __name__ == "__main__":
    main()
