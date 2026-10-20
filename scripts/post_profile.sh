#!/usr/bin/env bash
# The measurements that must not overlap a profiling pass, queued behind the one in flight.
#
# e3 re-profiles every service before it solves anything, and e4 times a hop to the
# microsecond, so both would be corrupted by -- and would corrupt -- the thread-scaling
# curves scripts/reprofile.sh is measuring on the same 80 cores. Wait for that chain to
# finish (or die), then run the checks the waiting was for.
#
#   ssh v100 'cd /storage/xuedh_workspace/mopar-impl && \
#     (setsid nohup scripts/post_profile.sh > runs/post_profile.log 2>&1 </dev/null &)'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
mkdir -p runs

wait_for() {  # wait_for <marker> <file>: until <file> carries it, or the chain is gone
  local marker="$1" file="$2"
  while ! grep -q "$marker" "$file" 2>/dev/null \
      && pgrep -f 'scripts/reprofile.sh' >/dev/null; do
    sleep 60
  done
}

step() {  # step <artifact-basename> <command...>: progress on stdout, output to runs/
  local name="$1"
  shift
  echo "== $(date '+%H:%M:%S') $name: $* =="
  "$@" > "runs/$name.log" 2>&1
  echo "-- $(date '+%H:%M:%S') $name exit=$?"
}

wait_for '== done ==' runs/reprofile.log

# First, because the sweep below reports what these assert: Eq. (6)'s latency
# constraint is now what chooses the slice count, and only a third of that claim is
# tested by the pure-standard-library suite that runs locally.
step unittests "$PY" -m unittest discover -s tests -q

# Same service list, batch sizes and pinned counts as the pass in scripts/reprofile.sh,
# so the two JSONs differ only in what the partitioner was told to solve.
step e3_analytic "$PY" -u experiments/e3_partition.py --device cpu --batches 1 4 \
  --quota eq3 --timing analytic --curve 1 2 3 4 5 6 8 12 16 24 32 \
  --out runs/e3_analytic.json

# Fig. 3 on the services whose widest edges are megabytes rather than kilobytes, and
# Fig. 8's two transports over the same tensor.
step e4_communication "$PY" -u experiments/e4_communication.py --device cpu \
  --models vgg16 inception_v3 convnext_tiny inception_transformer disbert \
  --edges 2 --repeats 3 --out runs/e4_communication.json

echo "== $(date '+%H:%M:%S') done =="
