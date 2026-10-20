"""Self-checks for the recorded program a slice executes (P5's premise).

The claim under test is narrow and load-bearing: replaying a *subset* of a model's
recorded calls, with the values a cut separates shipped by hand, computes what the
model computes. Everything the end-to-end numbers say is only about the split, so
these tests compare numbers and never timings.

Run: python -m unittest tests.test_faas_program   (needs torch)
"""

from __future__ import annotations

import gc
import sys
import unittest
import weakref
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicer.faas.program import record, run_chain  # noqa: E402
from slicer.models.graphify import _iter_tensors  # noqa: E402
from slicer.models import zoo  # noqa: E402

KEYS = ("gcn1", "resnet50", "lstm_cnn")


def _flat(value):
    return list(_iter_tensors(value))


class TestRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {}
        for key in KEYS:
            model, inputs = zoo.load(key, device="cpu")
            cls.cases[key] = (model, inputs, record(model, inputs, name=key))

    def expected(self, key):
        model, inputs, _ = self.cases[key]
        with torch.no_grad():
            return _flat(model(*inputs))

    def test_the_replayed_program_computes_the_forward(self):
        for key in KEYS:
            with self.subTest(model=key):
                _, inputs, program = self.cases[key]
                back = _flat(program.run(inputs))
                self.assertEqual(len(back), len(self.expected(key)))
                for got, want in zip(back, self.expected(key)):
                    self.assertTrue(torch.allclose(got, want, atol=1e-5),
                                    "a replay of every step is not the forward")

    def test_a_cut_in_the_middle_ships_what_the_next_slice_needs(self):
        for key in KEYS:
            _, inputs, program = self.cases[key]
            order = program.graph.order
            for part in (1, len(order) // 2, len(order) - 1):
                with self.subTest(model=key, cut=part):
                    cut = run_chain(program, [order[:part], order[part:]], inputs)
                    for got, want in zip(_flat(cut), self.expected(key)):
                        self.assertTrue(torch.allclose(got, want, atol=1e-5))

    def test_the_slices_together_run_every_step_once(self):
        _, _, program = self.cases["resnet50"]
        order = program.graph.order
        parts = [order[:20], order[20:60], order[60:]]
        cases = program.slices(parts)
        positions = [p for case in cases for p in case.steps]
        self.assertEqual(sorted(positions), list(range(len(program.steps))))
        self.assertEqual([case.layers for case in cases], [tuple(p) for p in parts])

    def test_a_residual_cut_ships_the_skip_as_well_as_the_block_output(self):
        """ResNet adds the block input around the block, so a cut sees two values."""
        _, _, program = self.cases["resnet50"]
        order = program.graph.order
        middle = len(order) // 2
        cases = program.slices([order[:middle], order[middle:]])
        self.assertEqual([peer for peer, _ in cases[1].inbound], [0])
        self.assertGreater(len(cases[1].inbound[0][1]), 1,
                           "only the frontier value crossed; the skip was dropped")

    def test_every_value_that_crosses_is_shipped_by_its_writer(self):
        for key in KEYS:
            _, _, program = self.cases[key]
            order = program.graph.order
            first, second = program.slices([order[:7], order[7:]])
            self.assertEqual({k for _, keys in second.inbound for k in keys},
                             {k for _, keys in first.outbound for k in keys},
                             "a slice shipped a value nobody asked for")


class TestOwnership(unittest.TestCase):
    """A slice that holds the whole model holds the whole quota, so it must not."""

    def test_release_leaves_the_slice_able_to_run(self):
        model, inputs = zoo.load("resnet50", device="cpu")
        program = record(model, inputs, name="resnet50")
        order = program.graph.order
        case = program.slices([order[:12], order[12:]])[0]
        before = sum(p.numel() * p.element_size() for p in model.parameters())
        freed = program.release(case.keeps)
        after = sum(p.numel() * p.element_size() for p in model.parameters())
        self.assertGreater(freed, 0)
        self.assertLess(after, before / 2, "most of a ResNet is below the first cut")
        env = dict(zip(program.inputs, _iter_tensors(inputs)))
        case.run(env)
        self.assertTrue(all(key in env for _, keys in case.outbound for key in keys),
                        "the slice could not produce its own boundary")

    def test_a_released_weight_is_not_still_held_by_the_capture(self):
        """The freed bytes have to be *unreachable*, or a quota measures nothing.

        ``release()`` clears a parameter table, and a recorded step keeps the tensors it
        read by value -- so while the whole service's steps are alive, dropping a weight
        from the table frees nothing any collector can see, and every slice stays the
        size of the service. This is the order the child uses, asserted on reachability
        because a resident-size reading on a shared box is too noisy to be a test.
        """
        model, inputs = zoo.load("resnet50", device="cpu")
        program = record(model, inputs, name="resnet50")
        order = program.graph.order
        case = program.slices([order[:12], order[12:]])[0]
        victim = next(p for p in model.parameters() if id(p) not in case.keeps)
        held = weakref.ref(victim)
        del victim
        program.steps = ()
        program.release(case.keeps)
        gc.collect()
        self.assertIsNone(held(), "a weight below the cut is still referenced")


if __name__ == "__main__":
    unittest.main(verbosity=2)
