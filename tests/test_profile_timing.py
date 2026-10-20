"""Tests for the measured timing model: the curve arithmetic, not the measurement.

The measurement itself (``measure_scaling``) runs a whole service eight times and
belongs to an experiment, not to a test suite. What is tested here is the part that
decides a partition: how a measured curve turns into ``t_p(l_i, \\gamma_i)``, and
what happens at the edges of the ladder -- below the smallest thread count, above
the largest, and for a layer that was never run.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - the local editing mirror has no torch
    torch = None

from slicer.partition.cost import GB, PlatformConfig, TimingModel
from slicer.partition.graph import Layer
from slicer.profile.timing import MeasuredTiming, Scaling, group_ratio


def curve(**per_eta: float) -> dict[int, float]:
    return {int(key.split("_")[1]): value for key, value in per_eta.items()}


@unittest.skipIf(torch is None, "torch not installed")
class TestRatio(unittest.TestCase):
    def setUp(self) -> None:
        # A layer that scales well to 16 threads and then gets slower: the shape a
        # launch-bound layer actually has, which a serial fraction cannot express.
        self.scaling = Scaling(per_layer={"a": curve(eta_1=8.0, eta_2=4.4, eta_4=2.6,
                                                     eta_8=2.2, eta_16=2.3)},
                               whole=curve(eta_1=20.0, eta_2=12.0, eta_4=8.0,
                                           eta_8=6.0, eta_16=6.0))

    def test_a_measured_layer_reports_its_own_time_ratio(self) -> None:
        self.assertAlmostEqual(self.scaling.ratio("a", 4)[0], 2.6 / 8.0)
        self.assertEqual(self.scaling.ratio("a", 4)[1], "layer")

    def test_between_measurements_the_curve_is_interpolated_not_extrapolated(self) -> None:
        ratio, source = self.scaling.ratio("a", 3)
        self.assertAlmostEqual(ratio, (4.4 + (2.6 - 4.4) / 2) / 8.0)
        self.assertEqual(source, "layer@interp")

    def test_past_the_top_of_the_ladder_the_last_measured_ratio_is_reused(self) -> None:
        # 128 threads were never tried; claiming a linear speedup there would be the
        # Amdahl assumption this module exists to replace.
        self.assertEqual(self.scaling.ratio("a", 128), (2.3 / 8.0, "layer@max"))

    def test_an_unmeasured_layer_falls_back_to_the_forward(self) -> None:
        ratio, source = self.scaling.ratio("never_measured", 8)
        self.assertAlmostEqual(ratio, 6.0 / 20.0)
        self.assertEqual(source, "forward")

    def test_no_curve_at_all_is_reported_rather_than_assumed_ideal(self) -> None:
        self.assertEqual(Scaling().ratio("a", 8), (1.0, "none"))

    def test_eta_one_is_the_base(self) -> None:
        self.assertEqual(self.scaling.ratio("a", 1), (1.0, "base"))
        self.assertEqual(self.scaling.ratio("a", 0), (1.0, "base"))


@unittest.skipIf(torch is None, "torch not installed")
class TestTiming(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = PlatformConfig()
        self.scaling = Scaling(per_layer={"conv1": {1: 4.0, 4: 1.5, 16: 1.6}},
                               whole={1: 10.0, 4: 5.0, 16: 5.0},
                               copy_bytes_per_s=2.0 * GB)
        self.timing = MeasuredTiming(platform=self.platform, scaling=self.scaling)
        self.layer = Layer(name="conv1", memory=GB, time=4.0, params=2 * GB)

    def test_compute_time_uses_the_measured_shape_not_the_layer_level(self) -> None:
        # The level comes from the service profile (4.0 s here); only the ratio is
        # measured, so re-profiling at a new batch size needs no new scaling run.
        self.assertAlmostEqual(self.timing.compute_time(self.layer, 4), 4.0 * 1.5 / 4.0)

    def test_a_layer_that_scales_backwards_is_not_credited_for_it(self) -> None:
        self.assertAlmostEqual(self.timing.compute_time(self.layer, 16), 4.0 * 1.6 / 4.0)
        self.assertGreater(self.timing.compute_time(self.layer, 16),
                           self.timing.compute_time(self.layer, 4))

    def test_without_measurements_the_analytic_model_still_answers(self) -> None:
        bare = MeasuredTiming(platform=self.platform)
        self.assertEqual(bare.compute_time(self.layer, 4),
                         TimingModel(platform=self.platform).compute_time(self.layer, 4))

    def test_aggregation_is_priced_at_the_measured_copy(self) -> None:
        # 4 workers, 1 result gathered over the wire: 3/4 of the output, at 2 GB/s.
        self.assertAlmostEqual(self.timing.aggregate_time(self.layer, 4, GB),
                               0.75 / 2.0)

    def test_a_slice_that_sends_nothing_aggregates_nothing(self) -> None:
        self.assertEqual(self.timing.aggregate_time(self.layer, 4, 0.0), 0.0)
        self.assertEqual(self.timing.aggregate_time(self.layer, 1, GB), 0.0)

    def test_a_folded_node_averages_its_members_curves(self) -> None:
        group = Layer(name="conv1", memory=GB, time=6.0, members=("conv1", "other"))
        # conv1 -> 1.5/4.0, "other" is unmeasured so it takes the forward's 5.0/10.0.
        self.assertAlmostEqual(group_ratio(self.scaling, group, 4), (0.375 + 0.5) / 2)
        self.assertAlmostEqual(self.timing.compute_time(group, 4), 6.0 * (0.375 + 0.5) / 2)


if __name__ == "__main__":
    unittest.main()
