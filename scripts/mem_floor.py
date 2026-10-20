"""What a slice's memory quota actually buys on bare metal.

Table III bills a function by the memory it is *given*, and on OpenFaaS that given
amount is a cgroup: the container cannot hold more than it pays for. Here a quota is a
``RLIMIT_AS`` on a forked child, and the child got there by loading the *whole* service
and dropping what its slice does not own. Whether the drop returns anything to the
operating system is the question this script answers, one resident-size reading at a
time, because the answer decides whether ``M_g`` in Eq. (6) measures a real footprint.

The readings are deliberately separated by what could plausibly hold the memory back:

    after ``release()``    -- the parameter table was cleared, refcounts should drop
    after dropping steps   -- a recorded step holds the tensors it read *by value*, so
                              the other slices' steps keep this slice's weights alive
    after ``gc.collect()`` -- a reference cycle was holding a tensor alive
    after ``malloc_trim``  -- glibc kept the top of the heap instead of unmapping it

On Inception-Transformer the steps are all of it: 12211 MB survives ``release()``, and
once the other slices' steps are gone a collection takes the process to 3138 MB and a
trim to 1722 MB. A quota is only a footprint after that.

Run on v100:

    python scripts/mem_floor.py --model inception_transformer --slice 0
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slicer.faas.program import record  # noqa: E402
from slicer.faas.runtime import _resident  # noqa: E402
from slicer.models import zoo  # noqa: E402

MB = 2 ** 20


def _equal_layers(order, parts: int) -> list[list[str]]:
    """The same cut Uniform makes: contiguous runs of layers of equal length."""
    step = -(-len(order) // max(1, parts))
    return [list(order[i:i + step]) for i in range(0, len(order), step)]


def _trim() -> int:
    """Ask glibc to unmap the top of the heap; -1 where libc has no such symbol."""
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        return -1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="inception_transformer")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--slices", type=int, default=8,
                        help="cut the service into this many slices before picking one")
    parser.add_argument("--slice", type=int, default=0,
                        help="which slice to keep; -1 keeps the whole service")
    args = parser.parse_args()

    def say(stage: str, **extra: object) -> None:
        resident, virtual = _resident()
        print("%-18s rss %10.1f MB  virt %10.1f MB  %s"
              % (stage, resident / MB, virtual / MB,
                 "  ".join("%s=%s" % item for item in extra.items())), flush=True)

    say("imported")
    started = time.time()
    model, inputs = zoo.load(args.model, device="cpu", batch=args.batch)
    say("loaded", seconds=round(time.time() - started, 1))
    started = time.time()
    program = record(model, inputs, name=args.model, device="cpu")
    say("recorded", steps=len(program.steps), seconds=round(time.time() - started, 1))

    if args.slice >= 0:
        cases = program.slices(_equal_layers(program.graph.order, args.slices))
        case = cases[min(args.slice, len(cases) - 1)]
        freed = program.release(case.keeps)
        say("released", slice=case.index, layers=len(case.layers),
            keeps=len(case.keeps), freed_mb=round(freed / MB, 1))
        # A recorded step holds the weights it read by value, so until the *other*
        # slices' steps go, clearing the parameter tables frees nothing.
        held = len(program.steps)
        program.steps = ()
        say("dropped steps", steps=held)
        del cases, case
    else:
        say("released", slice="-", freed_mb=0.0)
    gc.collect()
    say("gc")
    _trim()
    say("malloc_trim")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0
    print("peak rss for the whole process: %.1f MB" % (peak / MB,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
