"""Self-checks for the predictors of Sec. II-B.

RMSLE is the whole evaluation story of this section, and it is easy to get a
number without a meaning: an asymmetric metric summed over a population whose
scales span five orders of magnitude mostly measures the largest rows. So these
tests fix the metric's behaviour by hand (perfect fit, the under/over asymmetry
the paper cites, scale invariance) and then check both protocols on data whose
answer is known -- a footprint exactly proportional to ``s`` is predictable from
``s`` alone, while two operators that differ only in ``p`` are not, which is the
blind spot of the paper's three features.

Run: python3 -m unittest discover -s tests  (needs numpy and sklearn)
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression  # noqa: F401
except ImportError:  # pragma: no cover - the local editing mirror has neither
    np = None

from slicer.profile.regress import (accuracy, cross_fit, design, group,  # noqa: E402
                                   hold_out_sizes, per_operator, rmsle, target)


class Row:
    """Duck-typed ``OpSample``/``LayerSample``: regress.py only reads attributes."""

    def __init__(self, model, layer, family, size, params, memory, ns):
        self.model = model
        self.layer = layer
        self.family = family
        self.size = size
        self.params = params
        self.memory = memory
        self.ns = ns


def series(model="net", layer="layer0", family="conv0", k=1e3, offset=0.0):
    """One operator seen at eight input sizes, with memory ``offset + k * s``."""
    return [Row(model, layer, family, size, 4096.0, offset + k * size, 10.0 * size)
            for size in range(1, 9)]


@unittest.skipIf(np is None, "numpy/sklearn not installed")
class TestMetric(unittest.TestCase):
    def test_perfect_prediction_scores_zero(self) -> None:
        values = [1.0, 50.0, 3e6]
        self.assertEqual(rmsle(values, values), 0.0)
        self.assertEqual(accuracy(values, values), 1.0)

    def test_underestimating_costs_more_than_overestimating(self) -> None:
        # The paper's stated reason for the metric: a shortfall is an OOM, and in
        # log space 90 against 100 is further away than 110 against 100.
        self.assertGreater(rmsle([100.0], [90.0]), rmsle([100.0], [110.0]))

    def test_error_is_relative_not_absolute(self) -> None:
        # A 10% miss must score the same at a megabyte and at a gigabyte, or the
        # aggregate would report only the convolutions.
        self.assertAlmostEqual(rmsle([1e6], [1.1e6]), rmsle([1e9], [1.1e9]), delta=1e-6)

    def test_a_negative_prediction_cannot_score_better_than_zero(self) -> None:
        self.assertEqual(rmsle([1.0], [-1e9]), rmsle([1.0], [0.0]))

    def test_accuracy_ignores_zero_targets(self) -> None:
        # An in-place activation has no bytes to be wrong about; scoring it as a
        # perfect hit would let a predictor farm accuracy on empty rows.
        self.assertTrue(np.isnan(accuracy([0.0], [5.0])))
        self.assertAlmostEqual(accuracy([100.0, 0.0], [90.0, 0.0]), 0.9, places=12)

    def test_empty_input_is_not_a_perfect_score(self) -> None:
        self.assertTrue(np.isnan(rmsle([], [])))


@unittest.skipIf(np is None, "numpy/sklearn not installed")
class TestFeatures(unittest.TestCase):
    def test_design_columns_are_features_then_one_hot(self) -> None:
        rows = [Row("a", "l", "conv", 3, 4096.0, 1.0, 2.0),
                Row("b", "l", "conv", 5, 4096.0, 1.0, 2.0)]
        matrix = design(rows, ("size", "params"), ("a", "b", "c"))
        self.assertEqual(matrix.shape, (2, 5))
        self.assertEqual(matrix[0].tolist(), [3.0, 4096.0, 1.0, 0.0, 0.0])
        self.assertEqual(matrix[1].tolist(), [5.0, 4096.0, 0.0, 1.0, 0.0])

    def test_a_feature_a_table_does_not_have_reads_as_zero(self) -> None:
        # ``in_bytes`` is an operator-only column; the same design() call fits the
        # layer table, which has no such attribute.
        self.assertEqual(design(series()[:1], ("size", "in_bytes")).ravel().tolist(),
                         [1.0, 0.0])

    def test_target_follows_the_rows(self) -> None:
        rows = series()
        self.assertEqual(target(rows, "memory").tolist(), [r.memory for r in rows])

    def test_group_is_by_key_not_by_row(self) -> None:
        rows = series(layer="l", family="a") + series(layer="l", family="b")
        grouped = group(rows, lambda r: (r.model, r.layer))
        self.assertEqual([len(v) for v in grouped.values()], [16])
        self.assertEqual(len(group(rows, lambda r: (r.model, r.layer, r.family))), 2)

    def test_a_size_never_appears_on_both_sides(self) -> None:
        rows = series(family="a") + series(family="b")
        train, test = hold_out_sizes(rows, fraction=0.75, seed=1)
        self.assertFalse({r.size for r in train} & {r.size for r in test})
        self.assertEqual(len(train) + len(test), len(rows))
        self.assertTrue(test)


@unittest.skipIf(np is None, "numpy/sklearn not installed")
class TestProtocols(unittest.TestCase):
    def test_per_operator_recovers_a_proportional_operator(self) -> None:
        scored = per_operator(series(), "memory", lambda r: (r.model, r.layer, r.family))
        self.assertEqual(scored["groups"], 1)
        self.assertEqual(scored["unused_groups"], 0)
        self.assertLess(scored["lr"]["rmsle"], 1e-6)
        self.assertGreater(scored["lr"]["accuracy"], 0.999)

    def test_every_predictor_can_be_asked(self) -> None:
        scored = per_operator(series(k=1e3, offset=1e4), "memory", lambda r: r.family)
        for kind in ("lr", "xgboost", "rf"):
            self.assertIn("rmsle", scored[kind], kind)
            self.assertEqual(scored[kind]["n"], 2, kind)
            self.assertLess(scored[kind]["rmsle"], 0.35, kind)

    def test_a_group_too_small_to_hold_anything_out_is_counted_not_dropped_silently(self) -> None:
        scored = per_operator(series()[:3], "memory", lambda r: r.family)
        self.assertEqual(scored["groups"], 0)
        self.assertEqual(scored["unused_groups"], 1)
        self.assertTrue(scored["lr"]["insufficient"])

    def test_groups_are_fitted_separately(self) -> None:
        # A linear group and a step group together must score no better than the
        # sum of their own errors: one predictor each, so neither absorbs the
        # other's residual.
        straight = series(layer="l0", family="conv")
        step = [Row("net", "l1", "relu", size, 1.0, 1e6 if size > 4 else 1e3, 1.0)
                for size in range(1, 9)]
        keyed = lambda r: (r.model, r.layer, r.family)  # noqa: E731
        both = per_operator(straight + step, "memory", keyed)
        alone = per_operator(straight, "memory", keyed)
        self.assertEqual(both["groups"], 2)
        self.assertEqual(both["lr"]["n"], alone["lr"]["n"]
                         + per_operator(step, "memory", keyed)["lr"]["n"])
        self.assertGreater(both["lr"]["rmsle"], alone["lr"]["rmsle"])

    def test_cross_fit_cannot_see_a_held_out_group(self) -> None:
        # Eight layers, each exactly proportional in s, fitted as one population
        # with whole layers held out. <X,s,p> carries no shape, so the fit cannot
        # know which constant of proportionality a layer it has never measured
        # uses; the gap against per_operator is the price of that thin encoding.
        rows = [row for index in range(8)
                for row in series(layer="layer%d" % index, family="conv%d" % index,
                                  k=1e3 * (index + 1))]
        tight = per_operator(rows, "memory", lambda r: (r.model, r.layer, r.family))
        wide = cross_fit(rows, "memory", lambda r: (r.model, r.layer))
        self.assertLess(tight["lr"]["rmsle"], 1e-6)
        self.assertGreater(wide["lr"]["rmsle"], tight["lr"]["rmsle"] * 100)
        self.assertEqual(wide["groups"] + wide["unused_groups"], 8)

    def test_cross_fit_scores_only_held_out_groups(self) -> None:
        rows = [row for index in range(8) for row in series(layer="l%d" % index)]
        wide = cross_fit(rows, "memory", lambda r: (r.model, r.layer))
        self.assertEqual(wide["lr"]["n"], 2 * 8)  # two unseen layers, eight sizes each


if __name__ == "__main__":
    unittest.main()
