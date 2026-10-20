"""Self-checks for the bare-metal stand-in for OpenFaaS (P5's instrument).

Everything Fig. 10 and Table III say is a statement about numbers this module
produces, so the tests are about the *mapping*: does a plan's eta become that many
cores, does its quota become a limit the process actually obeys, does a cut's bytes
get counted once and only where they were written, and does a service that dies
report itself instead of hanging the sweep. Timings are asserted only as
"the split service is not slower than its own sum of slices", never against a
reference millisecond value -- an absolute latency on a shared box is not a fact
about the framework.

Run: python -m unittest tests.test_faas_runtime   (needs torch; spawns processes)
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicer.faas.runtime import Deployment, equal_layers  # noqa: E402
from slicer.models import zoo  # noqa: E402
from slicer.models.graphify import _iter_tensors  # noqa: E402

KEY = "gcn1"  # four layers, loads in a fifth of a second, and still has a cut


def _close(*services):
    for service in services:
        if service is not None:
            service.close()


class TestSplitExecution(unittest.TestCase):
    """The premise: a plan run as functions computes what the model computes."""

    @classmethod
    def setUpClass(cls):
        from slicer.faas.program import record
        model, cls.inputs = zoo.load(KEY, device="cpu")
        cls.program = record(model, cls.inputs, name=KEY, device="cpu")
        cls.order = list(cls.program.graph.order)
        cls.expected = list(_iter_tensors(cls.program.run(cls.inputs)))

    def assert_matches_whole_service(self, result):
        got = list(_iter_tensors(result))
        self.assertEqual(len(got), len(self.expected))
        for want, value in zip(self.expected, got):
            self.assertTrue(torch.allclose(want, value, atol=1e-4),
                            "a split service computed a different answer")

    def test_one_slice_through_the_deployment_path(self):
        """Fig. 10's ``Unsplit`` is a *function*, so it goes through the same machinery.

        One process is both the entry and the terminal, which is the case the
        two-ended completion pipe exists to serve: with a single field for it, the
        whole service in one function reads a connection it only opened for writing.
        """
        service = None
        try:
            service = Deployment.build(KEY, [self.order]).start()
            self.assertEqual(len(service.slices), 1)
            self.assertEqual(service.entry_rank, service.ranks[service.terminal_index])
            load = service.load(self.inputs, requests=3, concurrency=1)
            self.assertEqual(len(load.latencies_s), 3)
            self.assert_matches_whole_service(
                next(row["result"] for row in load.per_slice if row["result"] is not None))
        finally:
            _close(service)

    def test_two_slices_on_one_server(self):
        service = None
        try:
            service = Deployment.build(KEY, equal_layers(self.order, 2)).start()
            round_ = service.invoke(self.inputs)
            self.assert_matches_whole_service(round_.result)
            self.assertEqual(len(round_.per_slice), 2)
        finally:
            _close(service)

    def test_two_slices_on_two_servers(self):
        service = None
        try:
            service = Deployment.build(KEY, equal_layers(self.order, 2),
                                       servers=[0, 1]).start()
            round_ = service.invoke(self.inputs)
            self.assert_matches_whole_service(round_.result)
        finally:
            _close(service)

    def test_a_slice_order_that_disagrees_with_the_layer_order_is_refused(self):
        """Interleaving a chain makes the *slices* wait on each other.

        The runtime is not obliged to run that, but it has to say so: without the
        check this build succeeds and the first request deadlocks in a handshake,
        which is the failure mode a twelve-service sweep cannot debug from a log.
        """
        o = self.order
        with self.assertRaises(ValueError) as raised:
            Deployment.build(KEY, [[o[0], o[2]], [o[1], o[3]]])
        self.assertIn("first step", str(raised.exception))

    def test_a_straddling_plan_from_the_partitioner_runs(self):
        """Whatever NonSplit decides, the deployment has to be able to execute it.

        NonSplit's move is only safe when every producer of the moved node sits in an
        earlier slice, and that is the same condition :func:`_rank_order` checks -- so
        this is the two modules agreeing on one claim. On a model with no residual to
        straddle the plan comes back contiguous, and there is nothing to test.
        """
        from slicer.partition.baselines import nonsplit
        from slicer.partition.cost import PlatformConfig, TimingModel
        from slicer.faas.program import record

        platform = PlatformConfig()
        model, inputs = zoo.load("resnet50", device="cpu")
        program = record(model, inputs, name="resnet50", device="cpu")
        position = {name: index for index, name in enumerate(program.graph.order)}
        plan = nonsplit(program.graph, TimingModel(platform), platform)
        # A plan's ``nodes`` are the simplified graph's aggregates; ``members`` are the
        # layers a function actually runs, which is what a deployment is built from.
        groups = [list(s.members) for s in plan.slices]
        straddling = [g for g in groups
                      if [position[n] for n in g]
                      != list(range(position[g[0]], position[g[0]] + len(g)))]
        if not straddling:
            self.skipTest("NonSplit returned a contiguous plan for resnet50")
        service = None
        try:
            service = Deployment.build("resnet50", groups).start()
            got = list(_iter_tensors(service.invoke(inputs).result))
            want = list(_iter_tensors(program.run(inputs)))
            self.assertEqual(len(got), len(want))
            for expected, value in zip(want, got):
                self.assertTrue(torch.allclose(expected, value, atol=1e-4))
        finally:
            _close(service)


class TestKnobs(unittest.TestCase):
    """eta -> cores, alloc -> limit: the two knobs the plan is billed on."""

    @classmethod
    def setUpClass(cls):
        from slicer.faas.program import record
        model, cls.inputs = zoo.load(KEY, device="cpu")
        cls.order = list(record(model, cls.inputs, name=KEY, device="cpu").graph.order)

    def test_eta_hands_out_that_many_cores_and_no_more(self):
        groups = equal_layers(self.order, 2)
        service = None
        try:
            service = Deployment.build(KEY, groups, etas=[1, 2]).start()
            by_index = {row["index"]: row for row in service.ready}
            self.assertEqual(by_index[0]["cores"], 1)
            self.assertEqual(by_index[1]["cores"], 2)
            # The blocks must be disjoint: two OpenMP teams on one thread read as a
            # slowdown, and the experiment would blame the placement.
            held = [core for row in service.ready for core in
                    service.core_blocks()[row["rank"]]]
            self.assertEqual(len(held), len(set(held)))
        finally:
            _close(service)

    def test_a_slice_that_dies_fails_loudly(self):
        """A slice that cannot finish must be a result, not a hang.

        Killed from outside rather than starved by a quota, because "how small a
        limit still lets a function run" is a property of torch's allocator and not
        of this module -- what has to hold either way is that the parent notices. The
        victim is a *downstream* slice, so the entry is left blocked on a lane and the
        only thing between that and a wedged sweep is reading every pipe each poll.
        """
        service = None
        started = time.monotonic()
        try:
            service = Deployment.build(KEY, equal_layers(self.order, 2)).start()
            victim = max(service.processes, key=lambda rank: rank)
            service.processes[victim].terminate()
            service.processes[victim].join(10)
            with self.assertRaises(RuntimeError) as raised:
                service.invoke(self.inputs)
            self.assertLess(time.monotonic() - started, 120.0,
                            "the failure took long enough to look like a hang")
            self.assertIn("slice", str(raised.exception))
        finally:
            _close(service)


class TestCampaign(unittest.TestCase):
    """The closed loop Fig. 10b's percentile is taken from."""

    @classmethod
    def setUpClass(cls):
        from slicer.faas.program import record
        model, inputs = zoo.load(KEY, device="cpu")
        cls.program = record(model, inputs, name=KEY, device="cpu")
        cls.order = list(cls.program.graph.order)
        cls.inputs = inputs

    def test_the_window_cannot_exceed_what_a_lane_carries(self):
        service = None
        try:
            service = Deployment.build(KEY, equal_layers(self.order, 2),
                                       queue_depth=2).start()
            with self.assertRaises(ValueError):
                service.load(self.inputs, requests=4, concurrency=3)
            # Refusing the request must not have damaged the service.
            self.assertEqual(len(service.load(self.inputs, requests=2,
                                             concurrency=2).latencies_s), 2)
        finally:
            _close(service)

    def test_a_campaign_reports_one_latency_per_request(self):
        service = None
        try:
            service = Deployment.build(KEY, equal_layers(self.order, 2),
                                       queue_depth=3).start()
            load = service.load(self.inputs, requests=7, concurrency=3)
            self.assertEqual(len(load.latencies_s), 7)
            self.assertEqual(sum(row["requests"] for row in load.per_slice), 7 * 2)
            self.assertGreater(load.throughput, 0.0)
            # A percentile over seven is a sorted list, not a mean: p99 >= p50.
            self.assertGreaterEqual(load.percentile(99), load.percentile(50))
            # A request's clock starts when it enters the service, so the entry's own
            # hop is *inside* its latency rather than in front of it.
            entry = next(row for row in load.per_slice if row["entry"])
            self.assertGreaterEqual(min(load.latencies_s), min(entry["span_s"]))
        finally:
            _close(service)

    def test_one_start_answers_several_windows(self):
        """A concurrency curve is one deployment, not one deployment per point.

        Starting costs a whole model load per slice, so :meth:`Deployment.sweep` runs
        the windows back to back and cuts the reports apart afterwards. That only
        holds because every rank serves exactly the requests of a window before the
        next one opens, which is what the split is checked against here.
        """
        service = None
        try:
            service = Deployment.build(KEY, equal_layers(self.order, 2),
                                       queue_depth=4).start()
            with self.assertRaises(ValueError):
                # Refused up front: nothing was admitted, so nothing is half-run.
                service.sweep(self.inputs, [(2, 1), (2, 9)])
            loads = service.sweep(self.inputs, [(3, 1), (5, 2), (2, 4)])
            self.assertEqual([len(load.latencies_s) for load in loads], [3, 5, 2])
            for load, requests in zip(loads, (3, 5, 2)):
                self.assertGreater(load.seconds, 0.0)
                for row in load.per_slice:
                    self.assertEqual(row["requests"], requests)
                    self.assertEqual(len(row["span_s"]), requests)
                self.assertGreater(sum(sum(row["moved_bytes"])
                                       for row in load.per_slice), 0.0,
                                   "a window that moved nothing was not really run")
        finally:
            _close(service)

    def test_bytes_are_billed_where_they_are_written(self):
        """The wire total is the cut's activation bytes, counted once per request."""
        groups = equal_layers(self.order, 2)
        service = None
        try:
            service = Deployment.build(KEY, groups).start()
            per_request = sum(service.program.sizes[key]
                              for keys in service.lanes.values() for key in keys)
            self.assertGreater(per_request, 0.0, "gcn1 has no cut to test with")
            load = service.load(self.inputs, requests=4, concurrency=1)
            moved = sum(sum(row["moved_bytes"]) for row in load.per_slice)
            self.assertGreaterEqual(moved, 4 * per_request)
            # A payload carries a stamp and a header per value, never a second copy.
            self.assertLess(moved, 4 * (per_request + 4096))
        finally:
            _close(service)

    def test_a_closed_service_leaves_nothing_behind(self):
        service = Deployment.build(KEY, equal_layers(self.order, 2)).start()
        service.invoke(self.inputs)
        root = service.lane_root
        service.close()
        self.assertTrue(all(not process.is_alive()
                            for process in service.processes.values()))
        self.assertFalse(os.path.exists(root))


if __name__ == "__main__":
    unittest.main(verbosity=2)
