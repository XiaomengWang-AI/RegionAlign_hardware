"""Exact checksum compatibility on synthetic tensors and real tiny quantizers."""
import unittest
from unittest import mock

import torch

from regionalign.tests.helpers import TINY_GROUP, tiny_lm
from regionalign.experiments import bounded_checksum as B
from regionalign.harness import core as P, region as R


class BoundedChecksumTests(unittest.TestCase):
    def test_real_bf16_int8_int4_quantizers_match_the_original_checksum(self):
        for fmt in ("bf16", "int8", "int4"):
            for seed in (0, 11, 42):
                with self.subTest(format=fmt, seed=seed):
                    model = tiny_lm(torch.bfloat16, seed)
                    P.quantize(model, fmt, group=TINY_GROUP)
                    reference = R.model_checksum(model)
                    for chunk in (7, 127, B.DEFAULT_CHUNK_ELEMENTS):
                        self.assertEqual(B.model_checksum(model, chunk), reference)

    def test_raw_words_global_offsets_and_period_boundaries(self):
        generator = torch.Generator().manual_seed(79)
        model = torch.nn.Module()
        # Raw BF16 words include NaN payloads and sign bits: a numeric cast
        # instead of reinterpretation would fail this comparison.
        words = torch.randint(-32768, 32768, (131048,), dtype=torch.int16, generator=generator)
        model.register_parameter("raw", torch.nn.Parameter(words.view(torch.bfloat16), requires_grad=False))
        reference = R.model_checksum(model)
        for chunk in (997, 65520, 65521, 65522):
            self.assertEqual(B.model_checksum(model, chunk), reference)

    def test_noncontiguous_parameters_and_scales_preserve_logical_byte_order(self):
        generator = torch.Generator().manual_seed(8)
        for dtype in (torch.bfloat16, torch.float32, torch.float64, torch.int8):
            with self.subTest(dtype=dtype):
                value = torch.randn(16, 34, generator=generator).to(dtype).t()[::2]
                self.assertFalse(value.is_contiguous())
                model = torch.nn.Module()
                model.register_parameter("weight", torch.nn.Parameter(value, requires_grad=False))
                model.ar_quant = {"scales": {"weight": torch.rand(14, 10, generator=generator).t()}}
                reference = R.model_checksum(model)
                for chunk in (7, 32, 999):
                    self.assertEqual(B.model_checksum(model, chunk), reference)

    def test_float_scale_reduction_and_conversions_are_not_partitioned(self):
        generator = torch.Generator().manual_seed(15)
        model = torch.nn.Module()
        model.register_parameter("a", torch.nn.Parameter(torch.tensor([1.0, -0.0], dtype=torch.bfloat16)))
        scales = torch.randn(131047, dtype=torch.float64, generator=generator)
        scales[::7] *= 1e12
        model.ar_quant = {"scales": {"z": scales, "a": scales[:19].flip(0)}}
        self.assertEqual(B.model_checksum(model, 3), R.model_checksum(model))
        original_sum = torch.Tensor.sum
        float_sizes = []
        def observe(tensor, *args, **kwargs):
            if tensor.dtype == torch.float64:
                float_sizes.append(tensor.numel())
            return original_sum(tensor, *args, **kwargs)
        with mock.patch.object(torch.Tensor, "sum", observe):
            B.model_checksum(model, 3)
        self.assertEqual(float_sizes, [19, 131047])

    def test_signed_overflow_accumulation_matches_torch_int64_sum(self):
        values = [(1 << 63) - 1, (1 << 63) - 1, -(1 << 63), -19, 1 << 62]
        reference = int(torch.tensor(values, dtype=torch.int64).sum())
        total = 0
        for value in values:
            total = B._signed_int64(total + value)
        self.assertEqual(total, reference)
        self.assertEqual(sum(values) % B.STATE_MODULUS, total % B.STATE_MODULUS)

    def test_parameter_staging_stays_bounded_for_transposed_inputs(self):
        value = torch.arange(10000, dtype=torch.int16).reshape(100, 100).t().view(torch.bfloat16)
        model = torch.nn.Module()
        model.register_parameter("a", torch.nn.Parameter(value, requires_grad=False))
        reference = R.model_checksum(model)
        arange = torch.arange
        spans = []
        def bounded(*args, **kwargs):
            result = arange(*args, **kwargs)
            spans.append(result.numel())
            return result
        with mock.patch.object(torch, "arange", side_effect=bounded):
            self.assertEqual(B.model_checksum(model, 113), reference)
        self.assertLessEqual(max(spans), 113)

    def test_empty_parameters_and_invalid_chunk_sizes(self):
        model = torch.nn.Module()
        model.register_parameter("a", torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16)))
        self.assertEqual(B.model_checksum(model), R.model_checksum(model))
        for chunk in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                B.model_checksum(model, chunk)

    def test_install_only_patches_region_and_restores_after_nested_exception(self):
        original = R.model_checksum
        sentinel = mock.Mock(return_value=123)
        with mock.patch.object(R, "model_checksum", sentinel):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with B.install():
                    self.assertIs(R.model_checksum, B.model_checksum)
                    with B.install():
                        self.assertIs(R.model_checksum, B.model_checksum)
                    self.assertIs(R.model_checksum, B.model_checksum)
                    raise RuntimeError("intentional")
            self.assertIs(R.model_checksum, sentinel)
        self.assertIs(R.model_checksum, original)


if __name__ == "__main__":
    unittest.main()
