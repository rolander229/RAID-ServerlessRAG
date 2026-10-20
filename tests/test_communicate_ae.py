"""Tests for the cross-slice codec (Fig. 7): the axis bookkeeping and the ratio.

``fit`` runs 200 Adam steps, so the fitted cases keep to tiny tensors; what they
assert is the two claims the paper's Fig. 3 rests on -- that the encoded tensor
really is ``1/R`` of the bytes, and that spending more of them buys a better
reconstruction. The arithmetic of viewing an activation as a ``(N, C, H, W)`` map
is tested separately, because the 12 services cut on rank-1 to rank-4 tensors and
a wrong permutation is a silently wrong compression ratio.

Run: python3 tests/test_communicate_ae.py
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

from slicer.communicate.ae import Autoencoder, _plan, _unview, _view, fit  # noqa: E402


@unittest.skipIf(torch is None, "torch not installed")
class TestView(unittest.TestCase):
    def test_every_rank_round_trips_through_the_convolution_view(self):
        for shape, axis in [((64,), 0), ((4, 128), -1), ((2, 16, 96), -1),
                            ((1, 32, 8, 8), 1), ((3, 64, 5, 5), 1)]:
            with self.subTest(shape=shape, axis=axis):
                tensor = torch.randn(*shape)
                perm, added, inverse = _plan(len(shape), axis)
                coded = _view(tensor, perm, added)
                self.assertEqual(coded.dim(), 4)
                self.assertTrue(torch.equal(_unview(coded, shape, perm, added, inverse),
                                            tensor))

    def test_the_channel_axis_lands_where_a_convolution_expects_it(self):
        # A sequence activation (B, S, H) and a feature map (N, C, H, W) put their
        # channels in different places; the view has to know which one it is making.
        sequence = torch.randn(2, 16, 96)
        perm, added, inverse = _plan(sequence.dim(), -1)
        self.assertEqual(_view(sequence, perm, added).shape, (2, 96, 16, 1))
        feature_map = torch.randn(1, 32, 8, 8)
        perm, added, inverse = _plan(feature_map.dim(), 1)
        self.assertEqual(_view(feature_map, perm, added).shape, (1, 32, 8, 8))

    def test_a_lone_feature_vector_is_reshaped_not_permuted(self):
        perm, added, inverse = _plan(1, 0)
        vector = torch.randn(64)
        self.assertEqual(_view(vector, perm, added).shape, (1, 64, 1, 1))
        self.assertEqual(_unview(torch.randn(1, 64, 1, 1), (64,), perm, added, inverse).shape,
                         (64,))

    def test_a_scalar_has_nothing_to_compress(self):
        with self.assertRaises(ValueError):
            _plan(0, 0)


@unittest.skipIf(torch is None, "torch not installed")
class TestFit(unittest.TestCase):
    def test_the_encoded_tensor_is_the_requested_fraction_of_the_bytes(self):
        activation = torch.randn(2, 64, 8, 8)
        for ratio in (2, 4, 8, 16, 64):
            with self.subTest(ratio=ratio):
                codec = fit(activation, ratio, steps=60)
                self.assertLessEqual(abs(codec.raw_bytes / codec.coded_bytes - ratio),
                                     ratio * 0.05 + 1.0,
                                     "R is a channel count divided by a channel count, "
                                     "so it can only be approached in integers")
                self.assertEqual(codec.net.latent, max(1, round(64 / ratio)))

    def test_more_bytes_buy_a_better_reconstruction(self):
        # A rank-deficient activation is the friendly case, and the ordering is the
        # claim: coding harder loses more.
        base = torch.randn(2, 8, 8, 8)
        weight = torch.randn(64, 8, 1, 1)
        activation = torch.nn.functional.conv2d(base, weight)  # (2, 64, 8, 8), rank 8
        errors = [fit(activation, ratio, steps=250).relative_error
                  for ratio in (2, 8, 32)]
        self.assertLess(errors[0], errors[1])
        self.assertLess(errors[1], errors[2])

    def test_ratio_one_can_reproduce_its_input(self):
        activation = torch.randn(1, 16, 6, 6)
        codec = fit(activation, 1, steps=600, lr=3e-3)
        self.assertLess(codec.relative_error, 0.1)
        self.assertEqual(codec.reconstruct(activation).shape, activation.shape)

    def test_reconstruction_comes_back_in_the_original_shape(self):
        sequence = torch.randn(2, 16, 96)  # channels last, like BERT's hidden state
        codec = fit(sequence, 4, channel_axis=-1, steps=40)
        out = codec.reconstruct(sequence)
        self.assertEqual(out.shape, sequence.shape)
        self.assertEqual(codec.coded_bytes, sequence.numel() * 4 // 4)

    def test_a_ratio_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            fit(torch.randn(1, 4, 2, 2), 0.5)

    def test_the_accounting_reports_the_codec_as_time(self):
        codec = fit(torch.randn(1, 32, 8, 8), 8, steps=30)
        booked = codec.accounting()
        self.assertAlmostEqual(booked["codec_ms"],
                               booked["encode_ms"] + booked["decode_ms"])
        self.assertGreater(booked["byte_reduction"], 1.0)


@unittest.skipIf(torch is None, "torch not installed")
class TestAutoencoder(unittest.TestCase):
    def test_the_two_convolutions_are_one_layer_each(self):
        # Fig. 7 is explicit about the size of the codec, and the parameter count is
        # how a reader can check that nothing else was trained.
        net = Autoencoder(channels=64, latent=8)
        convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
        self.assertEqual(len(convs), 2)
        self.assertEqual(sum(p.numel() for p in net.parameters()), (64 * 8 + 8) + (8 * 64 + 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
