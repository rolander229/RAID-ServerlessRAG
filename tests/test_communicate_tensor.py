"""Self-checks for the activation codec (Sec. II-C: what a cut puts on the wire).

Run: python -m unittest tests.test_communicate_tensor   (needs torch)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicer.communicate.tensor import pack, unpack  # noqa: E402


class TestCodec(unittest.TestCase):
    def test_a_tensor_round_trips_exactly(self):
        for value in [torch.randn(2, 3, 4), torch.randint(0, 8, (5,)),
                      torch.ones(4, 1, dtype=torch.bool),
                      torch.arange(6, dtype=torch.int64).view(2, 3),
                      torch.randn(3, dtype=torch.float16),
                      torch.tensor(1.5), torch.empty(0), torch.randn(784, 3)]:
            with self.subTest(dtype=value.dtype, shape=tuple(value.shape)):
                back = unpack(pack(value))
                self.assertEqual(back.shape, value.shape)
                self.assertEqual(back.dtype, value.dtype)
                self.assertTrue(back.is_cpu)
                self.assertTrue(torch.equal(back, value.to(back.dtype)))

    def test_the_header_is_a_few_tens_of_bytes(self):
        """Fig. 3 compares transports by *bytes moved*, so the codec must not lie."""
        value = torch.randn(8, 3, 224, 224)
        payload = pack(value)
        overhead = len(payload) - value.numel() * value.element_size()
        self.assertLess(overhead, 64)
        self.assertGreater(len(payload), value.numel() // 2,
                           "the activation itself has to be what is shipped")

    def test_a_structure_around_a_tensor_keeps_its_shape(self):
        hidden = torch.randn(2, 16)
        case = (hidden, {"mask": torch.ones(2, 16, dtype=torch.int64),
                         "list": [hidden, 3]})
        back = unpack(pack(case))
        self.assertIsInstance(back, tuple)
        self.assertTrue(torch.equal(back[0], hidden))
        self.assertIsInstance(back[1]["list"], list)
        self.assertEqual(back[1]["list"][1], 3)
        self.assertTrue(torch.equal(back[1]["mask"], case[1]["mask"]))

    def test_a_value_that_is_not_a_tensor_crosses_too(self):
        """A GCN's adjacency index and a tokenizer's offsets arrive as ints."""
        for value in [7, "cora", None, (2, 3)]:
            self.assertEqual(unpack(pack(value)), value)

    def test_a_writable_copy_comes_back(self):
        """A received activation is the consumer's to write in place.

        ``torch.frombuffer`` over an immutable ``bytes`` would hand back a read-only
        tensor, and a residual block that adds into it fails on the first layer.
        """
        back = unpack(pack(torch.randn(128)))
        self.assertFalse(back.requires_grad)
        back.add_(1.0)

    def test_a_truncated_or_trailing_payload_is_refused(self):
        payload = pack(torch.randn(4, 4))
        with self.assertRaises(ValueError):
            unpack(payload + b"\x00")
        with self.assertRaises(Exception):
            unpack(payload[:len(payload) // 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
