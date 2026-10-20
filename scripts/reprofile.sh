#!/usr/bin/env bash
# Re-measure everything downstream of the layer graph, in the order results need it.
#
# Both fixes of 2026-09-22 -- Eq.(6)'s eta cap, and a container's weights never
# being billed to a node -- change the numbers every experiment reports: e3 sizes
# a slice against the weights it holds, and e2's Table I predictors take per-layer
# parameter count as a feature. Run on v100, detached:
#
#   ssh v100 'cd /storage/xuedh_workspace/mopar-impl && \
#     (setsid nohup scripts/reprofile.sh > runs/reprofile.log 2>&1 </dev/null &)'
#
# The order is deliberate: the analytic sweep is what fills the slice-plan and
# Fig. 2b/2d tables and it finishes first, so those can be written while the
# profiling passes are still running.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
mkdir -p runs

step() {  # step <artifact-basename> <command...>: progress on stdout, output to runs/
  local name="$1"
  shift
  echo "== $(date '+%H:%M:%S') $name: $* =="
  "$@" > "runs/$name.log" 2>&1
  echo "-- $(date '+%H:%M:%S') $name exit=$?"
}

step e3_analytic "$PY" -u experiments/e3_partition.py --device cpu --batches 1 4 \
  --quota eq3 --timing analytic --curve 1 2 3 4 5 6 8 12 16 24 32 \
  --out runs/e3_analytic.json

step e1_layer_graphs "$PY" -u experiments/e1_layer_graphs.py --device cpu \
  --out runs/e1_layer_graphs.json

step e2_cpu "$PY" -u experiments/e2_service_profiling.py --device cpu \
  --out runs/e2_cpu.json

step e2_cuda "$PY" -u experiments/e2_service_profiling.py --device cuda \
  --out runs/e2_cuda.json

# The expensive one last: measure_scaling repeats a forward for every eta level, so
# it runs at one request size and three levels -- enough to say whether the
# analytic Amdahl term was what held a plan up.
step e3_measured "$PY" -u experiments/e3_partition.py --device cpu --batches 1 \
  --quota eq3 --timing measured --max-eta-levels 3 --out runs/e3_measured.json

echo "== $(date '+%H:%M:%S') done =="
