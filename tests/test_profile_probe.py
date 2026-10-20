"""Tests for the measurement pass: what a byte is, and who pays for it.

Needs torch, so these run on v100 via ``python -m unittest
discover -s tests``. The three things worth failing loudly on are the ones no
downstream check catches: a layer whose name does not join to the structure pass
mislabels every profile after it, an operator that allocates nothing but is
charged the buffer it rewrote inflates a whole service, and a probe that observes
its own sizing code reports the profiler as part of the network.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn

    from slicer.models.graphify import build_structure_graph
    from slicer.profile.probe import (LayerRecord, OpRecord, attach_peaks,  # noqa: E402
                                     correct, measure_forward, median_repeats,
                                     op_family, output_bytes, request_size, scale)
except ImportError:  # pragma: no cover - the local editing mirror has no torch
    torch = nn = None


def build_net() -> "nn.Module":
    """A service with a skip edge, an in-place activation, and module-level work.

    Defined in a function because the classes below subclass ``nn.Module``, which
    does not exist on a machine without torch: the module must stay importable so
    its tests skip rather than error out.
    """

    class Gated(nn.Module):
        """The strays the graph has no node for run between two leaf modules."""

        def __init__(self, width: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(width, width)
            self.act = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(width, width)

        def forward(self, x):
            return self.fc2(self.act(torch.sigmoid(self.fc1(x)) * 2.0))

    class Net(nn.Module):
        def __init__(self, width: int = 8) -> None:
            super().__init__()
            self.stem = nn.Linear(width, width)
            self.gated = Gated(width)
            self.head = nn.Linear(width, width)

        def forward(self, x):
            hidden = self.stem(x)
            return self.head(self.gated(hidden)) + hidden

    torch.manual_seed(0)
    return Net().eval()


def build_cnn() -> "nn.Module":
    """A conv service, so a layer has an operator that really allocates."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
                         nn.Flatten(), nn.Linear(8 * 30 * 30, 10)).eval()


CONV_BYTES = 1 * 8 * 30 * 30 * 4
NAMES = ["stem", "gated.fc1", "gated.act", "gated.fc2", "head"]


@unittest.skipIf(torch is None, "torch not installed")
class TestJoin(unittest.TestCase):
    def setUp(self) -> None:
        self.net = build_net()
        self.inputs = (torch.randn(4, 8),)
        self.graph = build_structure_graph(self.net, self.inputs, name="net")
        self.layers, self.records = measure_forward(self.net, self.inputs,
                                                    expect_order=self.graph.order)

    def names(self) -> list:
        return [layer.name for layer in self.layers]

    def test_execution_order_is_the_graphs_order(self) -> None:
        # Both passes drive the same Namer, so a measured row joins to a graph
        # node by name; expect_order would have raised before this could pass.
        self.assertEqual(self.names(), NAMES)
        self.assertEqual(self.graph.order, NAMES)

    def test_a_shorter_sequence_is_reported_as_an_extra_layer(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            measure_forward(self.net, self.inputs, expect_order=self.graph.order[:-1])
        self.assertIn("extra", str(caught.exception))

    def test_a_reordered_sequence_is_reported_as_a_divergence(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            measure_forward(self.net, self.inputs,
                            expect_order=list(reversed(self.graph.order)))
        self.assertIn("diverged", str(caught.exception))

    def test_every_operator_is_listed_in_the_layer_it_names(self) -> None:
        by_name = {layer.name: layer for layer in self.layers}
        for record in self.records:
            self.assertIn(record.layer, by_name, record)
            self.assertIn(record, by_name[record.layer].ops)

    def test_records_appear_in_exactly_one_layer(self) -> None:
        self.assertEqual(sum(len(layer.ops) for layer in self.layers), len(self.records))

    def test_the_probe_does_not_profile_itself(self) -> None:
        # Sizing a result means calling torch on it, which used to record the
        # probe's own numel/element_size/untyped_storage calls as operators.
        for record in self.records:
            for forbidden in ("numel", "element_size", "untyped_storage", "storage",
                              "data_ptr", "grad_enabled", "parameters"):
                self.assertNotIn(forbidden, record.op, record)

    def test_operator_spans_fit_inside_the_layer_that_bills_them(self) -> None:
        for layer in self.layers:
            if layer.billed:
                continue  # a stray's time was spent before this layer opened
            self.assertGreater(layer.ns, 0.0, layer.name)
            self.assertLessEqual(sum(r.ns for r in layer.ops), layer.ns + 1.0,
                                 layer.name)


@unittest.skipIf(torch is None, "torch not installed")
class TestOwnership(unittest.TestCase):
    def setUp(self) -> None:
        self.layers, self.records = measure_forward(build_net(), (torch.randn(4, 8),))
        self.by_name = {layer.name: layer for layer in self.layers}

    def test_work_between_modules_is_billed_to_the_next_module(self) -> None:
        # graphify contracts a non-leaf operator into its consumer, and in
        # execution order that consumer is the leaf that runs next: sigmoid and
        # the scaling finish before ReLU opens, so ReLU pays for them. ``billed``
        # says so, which is what makes the shift visible instead of silent.
        self.assertEqual([self.by_name[name].billed for name in NAMES], [0, 0, 2, 0, 1])
        self.assertEqual([r.op for r in self.by_name["gated.act"].ops if "relu" not in r.op],
                         ["sigmoid", "mul"])

    def test_a_stray_after_the_last_module_is_not_lost(self) -> None:
        self.assertEqual(self.by_name["head"].ops[-1].family, "add")
        self.assertEqual(self.records[-1].layer, "head")


@unittest.skipIf(torch is None, "torch not installed")
class TestBytes(unittest.TestCase):
    def test_a_fresh_result_costs_its_storage(self) -> None:
        self.assertEqual(output_bytes(torch.zeros(4, 8)), 4 * 8 * 4)

    def test_a_view_of_an_operand_costs_nothing(self) -> None:
        # The offset case matters as much as the plain one: a slice is somebody
        # else's allocation, and only the storage pointer, not the tensor
        # pointer, knows that.
        source = torch.zeros(4, 8)
        for derived in (source.transpose(0, 1), source.reshape(8, 4), source[:2],
                        source.narrow(1, 2, 4), source.expand(1, 4, 8).squeeze(0)):
            self.assertEqual(output_bytes(derived, (source,)), 0.0, derived.shape)

    def test_an_in_place_write_costs_nothing(self) -> None:
        # The distinction the whole accounting rests on: F.relu with inplace=True
        # returns the buffer it was handed, and torch.relu returns a new one.
        source = torch.zeros(4, 8)
        self.assertEqual(output_bytes(nn.functional.relu(source, inplace=True),
                                      (source,)), 0.0)
        self.assertEqual(output_bytes(nn.functional.relu(source), (source,)), 4 * 8 * 4)

    def test_a_fresh_copy_of_an_operand_costs_its_storage(self) -> None:
        source = torch.zeros(4, 8)
        self.assertEqual(output_bytes(source.clone(), (source,)), 4 * 8 * 4)

    def test_two_views_of_one_storage_are_one_allocation(self) -> None:
        source = torch.zeros(4, 8)
        self.assertEqual(output_bytes((source, source.transpose(0, 1)), ()), 4 * 8 * 4)

    def test_an_in_place_activation_allocates_nothing(self) -> None:
        layers, _ = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),))
        act = self.layer(layers, "2")
        self.assertEqual([row.out for row in act.ops], [0.0])
        self.assertEqual(act.returned, 0.0)  # the buffer it hands back is the one it got
        self.assertIsNone(act.peak)  # CPU: no allocator high-water mark exists

    def test_a_reshape_that_copies_nothing_allocates_nothing(self) -> None:
        layers, _ = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),))
        flat = self.layer(layers, "3")
        self.assertEqual([row.op for row in flat.ops], ["flatten"])
        self.assertEqual(flat.returned, 0.0)
        self.assertEqual(flat.ops[0].out, 0.0)

    def test_a_conv_bills_exactly_the_image_it_wrote(self) -> None:
        layers, _ = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),))
        conv = self.layer(layers, "0")
        self.assertEqual([row.out for row in conv.ops], [float(CONV_BYTES)])
        self.assertEqual(conv.returned, float(CONV_BYTES))

    def test_a_host_side_query_is_not_an_operator(self) -> None:
        # BatchNorm2d.forward asks its input for a dim; that is a few hundred
        # nanoseconds of host time and no bytes, and it is not part of the
        # service the partitioner is pricing.
        _, records = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),))
        self.assertEqual([r.op for r in records if r.op in ("dim", "size", "numel")], [])
        self.assertEqual(sorted({r.op for r in records}),
                         ["batch_norm", "conv2d", "flatten", "linear", "relu"])

    def test_a_tensor_property_is_not_an_operator(self) -> None:
        # ``x.data`` reaches __torch_function__ as the descriptor's __get__, which
        # keeps that name and loses the property's. The first full CUDA sweep
        # recorded 7032 of them as operators, more than every real operator it
        # measured put together.
        class Read(nn.Module):
            def forward(self, x):
                return torch.relu(x.data.view_as(x).T.contiguous())

        layers, records = measure_forward(Read(), (torch.randn(4, 6),))
        self.assertEqual([r.op for r in records if r.op == "__get__"], [])
        self.assertEqual([layer.name for layer in layers], ["Read"])

    def test_the_timing_pass_reads_no_counters(self) -> None:
        # The passes are separate because a counter costs more than a small layer,
        # so the timing pass must come back with durations and no footprints --
        # and with none on a CPU, where the counters do not exist either.
        layers, records = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),))
        self.assertTrue(any(layer.ns > 0 for layer in layers))
        self.assertEqual({layer.peak for layer in layers}, {None})
        self.assertEqual({row.peak for row in records}, {None})

    def test_the_counter_pass_reports_no_durations(self) -> None:
        layers, records = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),),
                                          watch="memory")
        self.assertEqual({layer.ns for layer in layers}, {0.0})
        self.assertEqual({row.ns for row in records}, {0.0})
        self.assertEqual({layer.peak for layer in layers}, {None})

    def test_the_layer_pass_sees_no_operators_at_all(self) -> None:
        # Eq. (4)'s t_l comes from a forward with no __torch_function__ mode, or
        # the mode's own dispatch cost is inside the span it is measuring. It
        # therefore cannot report operators, which is the point of the pass.
        layers, records = measure_forward(build_cnn(), (torch.randn(1, 3, 32, 32),),
                                          watch="layers")
        self.assertEqual(records, [])
        self.assertEqual([layer.ops for layer in layers], [[] for _ in layers])
        self.assertTrue(any(layer.ns > 0 for layer in layers))
        self.assertEqual(layers[0].returned, CONV_BYTES)

    def test_an_unknown_pass_is_a_typo_not_a_default(self) -> None:
        with self.assertRaises(ValueError):
            measure_forward(build_net(), (torch.randn(4, 8),), watch="layer")

    def test_work_between_two_layers_is_billed_to_the_one_that_ran_first(self) -> None:
        # The pass that times layers sees no operators, so a hand-written service's
        # module-level work would belong to nobody: gcn1's four leaf layers cover
        # 0.16 ms of a 0.54 ms forward. What it can still measure is the interval
        # between two spans, and that is where the work is.
        class Between(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(64, 64)
                self.b = nn.Linear(64, 64)

            def forward(self, x):
                y = self.a(x)
                for _ in range(64):
                    y = torch.relu(y)  # no leaf module owns this
                return self.b(y)

        busy, _ = measure_forward(Between(), (torch.randn(4, 64),), watch="layers")
        quiet, _ = measure_forward(nn.Sequential(Between().a, Between().b),
                                   (torch.randn(4, 64),), watch="layers")
        self.assertEqual(busy[1].gap, 0.0)  # nothing follows the last span
        self.assertGreater(busy[0].ns, busy[0].gap)  # ns keeps its own span too
        self.assertGreater(busy[0].gap, 4 * quiet[0].gap)

    def test_a_layer_that_writes_reports_what_it_handed_back(self) -> None:
        layers, _ = measure_forward(build_net(), (torch.randn(4, 8),))
        fc = self.layer(layers, "gated.fc1")
        self.assertEqual(fc.returned, 4 * 8 * 4)
        self.assertTrue(any(r.out == fc.returned for r in fc.ops))

    def test_one_buffer_is_billed_once_within_a_layer(self) -> None:
        # __torch_function__ can see a function and the aten call under it, and
        # both hand back the same buffer. Only the call that allocated it may
        # bill it, or the operator table would hold every conv twice -- and a
        # layer's operator rows would add up to more than the layer itself.
        layers, _ = measure_forward(build_net(), (torch.randn(4, 8),))
        for name in ("stem", "gated.fc1", "gated.fc2"):
            layer = self.layer(layers, name)
            self.assertEqual(sum(r.out for r in layer.ops), layer.returned, name)

    def test_request_size_counts_elements_not_bytes(self) -> None:
        self.assertEqual(request_size((torch.zeros(4, 8), torch.zeros(2))), 34)
        self.assertEqual(request_size({"a": torch.zeros(4, 8)}), 32)

    @staticmethod
    def layer(layers, name):
        return next(layer for layer in layers if layer.name == name)


@unittest.skipIf(torch is None, "torch not installed")
class TestConsolidation(unittest.TestCase):
    """repeats -> one profile, in pure Python."""

    def record(self, name, ns, op_times=()):
        layer = LayerRecord(name, 0.0, ns, 0.0)
        layer.ops = [OpRecord("conv2d", name, value, 1.0) for value in op_times]
        return layer

    def test_medians_are_taken_per_layer_not_per_run(self) -> None:
        runs = [[self.record("a", 10.0), self.record("b", 1.0)],
                [self.record("a", 30.0), self.record("b", 3.0)],
                [self.record("a", 20.0), self.record("b", 2.0)]]
        medians = median_repeats(runs)
        self.assertEqual((medians["a"].ns, medians["b"].ns), (20.0, 2.0))
        self.assertEqual(list(medians), ["a", "b"])

    def test_a_device_with_no_peak_counter_keeps_reporting_none(self) -> None:
        medians = median_repeats([[self.record("a", 1.0)] for _ in range(2)])
        self.assertIsNone(medians["a"].peak)

    def test_a_billed_gap_survives_the_median(self) -> None:
        # Folding an interval into ns and then mediating over repeats has to keep
        # both numbers, or the report cannot say how much of a layer is its own.
        runs = []
        for value in (10.0, 30.0, 20.0):
            record = self.record("a", value + 5.0)
            record.gap = 5.0
            runs.append([record])
        self.assertEqual(median_repeats(runs)["a"].ns, 25.0)
        self.assertEqual(median_repeats(runs)["a"].gap, 5.0)

    def test_runs_that_disagree_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            median_repeats([[self.record("a", 1.0)], [self.record("b", 1.0)]])

    def test_the_instrumentation_is_subtracted_and_cannot_go_negative(self) -> None:
        medians = {"a": self.record("a", 100.0, [10.0, 10.0, 10.0]),
                   "b": self.record("b", 10.0, [1.0])}
        correct(medians, per_layer=40.0, per_op=20.0)
        self.assertEqual([medians["a"].ns, medians["b"].ns], [0.0, 0.0])
        self.assertEqual([r.ns for r in medians["a"].ops], [0.0] * 3)

    def test_peaks_join_the_two_passes_by_position(self) -> None:
        timed = {"a": self.record("a", 100.0, [10.0, 20.0])}
        measured = median_repeats([[self.record("a", 0.0, [0.0, 0.0])]])
        for layer in measured.values():
            layer.peak = 4096.0
            for operator in layer.ops:
                operator.peak = 2048.0
        merged = attach_peaks(timed, measured)
        self.assertEqual(merged["a"].peak, 4096.0)
        self.assertEqual([r.peak for r in merged["a"].ops], [2048.0, 2048.0])
        self.assertEqual([r.ns for r in merged["a"].ops], [10.0, 20.0])

    def test_a_disagreement_between_the_passes_is_refused(self) -> None:
        timed = {"a": self.record("a", 100.0, [10.0, 20.0])}
        measured = {"a": self.record("a", 0.0, [10.0])}
        with self.assertRaises(RuntimeError):
            attach_peaks(timed, measured)

    def test_scaling_keeps_shares_and_moves_the_sum(self) -> None:
        medians = {"a": self.record("a", 300.0), "b": self.record("b", 100.0)}
        factor = scale(medians, 800.0)
        self.assertAlmostEqual(factor, 2.0)
        self.assertEqual((medians["a"].ns, medians["b"].ns), (600.0, 200.0))

    def test_a_forward_that_ran_nothing_scales_nothing(self) -> None:
        # A divided-by-zero here would report a factor of inf and a profile of
        # NaNs, which is a worse failure than leaving the vector alone.
        medians = {"a": self.record("a", 0.0)}
        self.assertEqual(scale(medians, 500.0), 1.0)
        self.assertEqual(medians["a"].ns, 0.0)

    def test_op_family_is_a_table_lookup_not_a_guess(self) -> None:
        self.assertEqual(op_family("torch.nn.functional.conv2d"), "conv")
        self.assertEqual(op_family("aten::_convolution"), "conv")
        self.assertEqual(op_family("torch.matmul"), "matmul")
        self.assertEqual(op_family("Tensor.add_"), "add")
        self.assertEqual(op_family("aten::_slicedSoftmax"), "softmax")
        self.assertEqual(op_family("aten::expand"), "reshape")  # not an exponent
        self.assertEqual(op_family("Tensor.flatten"), "reshape")  # not a copy
        self.assertEqual(op_family("aten::addmm"), "matmul")  # not an addition
        # Two of the substrings in this one belong to other families, and the
        # first full sweep read it as a multiplication.
        self.assertEqual(op_family("multi_head_attention_forward"), "attention")
        self.assertEqual(op_family("aten::index_select"), "other")


@unittest.skipIf(torch is None, "torch not installed")
@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "no GPU")
class TestMeasuredPeak(unittest.TestCase):
    """The allocator's own reading, which is the only memory target a predictor
    can learn anything from: the byte model is a formula in the request size."""

    def records(self):
        model = nn.Sequential(nn.Conv2d(8, 64, 3, padding=1),
                              nn.Conv2d(64, 64, 3, padding=1),
                              nn.ReLU(inplace=True))
        inputs = (torch.randn(8, 8, 64, 64),)
        return measure_forward(model, inputs, device="cuda", watch="memory")

    def test_an_operator_that_needs_scratch_measures_more_than_it_writes(self) -> None:
        # cuDNN hands a convolution a workspace the byte model cannot see and the
        # high-water mark can. If nothing here ever allocates scratch, the reading
        # is not being taken at all and the tables would silently fall back to the
        # formula.
        _, records = self.records()
        self.assertTrue(all(row.peak >= row.out for row in records))
        self.assertTrue(any(row.peak > row.out for row in records),
                        "no operator measured above its own bytes")

    def test_an_in_place_layer_rises_no_higher_than_its_input(self) -> None:
        layers, _ = self.records()
        self.assertEqual(layers[-1].returned, 0.0)
        self.assertEqual(layers[-1].peak, 0.0)

    def test_the_operators_of_a_layer_add_up_to_no_more_than_the_layer(self) -> None:
        layers, _ = self.records()
        for layer in layers:
            if layer.billed:
                continue  # a stray's rise happened before this layer's entry reset
            self.assertLessEqual(sum(row.peak for row in layer.ops), layer.peak,
                                 layer.name)


if __name__ == "__main__":
    unittest.main()
