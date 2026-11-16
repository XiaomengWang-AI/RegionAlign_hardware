"""CPU tests of the kernel in ``harness.core`` on a tiny random Llama.

They cover what every experiment relies on: stored-bit flips restore exactly,
legal layout transforms preserve the function and the stored codes, integer
quantization is complete, and the in-range rule is the only value rule.
"""
import random
import unittest

import torch
torch.set_num_threads(1)
from transformers import LlamaConfig, LlamaForCausalLM

from regionalign.harness import core as P
from regionalign.tests.helpers import (TINY_GROUP, TinyRuntime, grouped_perm, random_spec,
                                       tiny_gqa_lm, tiny_layernorm_lm, tiny_lm, tiny_runtime)


class BitTests(unittest.TestCase):
    def test_bit_mask_bounds(self):
        self.assertEqual(P.bit_mask(15), -32768)
        with self.assertRaises(ValueError):
            P.bit_mask(16)

    def test_flips_restore_exactly(self):
        lm = tiny_lm(torch.bfloat16)
        name = "model.layers.1.mlp.down_proj.weight"
        before = lm.get_parameter(name).data.clone()
        flips = [(name, 7, 15), (name, 7, 3), (name, 300, 14)]
        P.apply_flips(lm, flips)
        after = lm.get_parameter(name).data
        self.assertFalse(torch.equal(before.view(torch.int16), after.view(torch.int16)))
        self.assertEqual(int((before.view(torch.int16) != after.view(torch.int16)).sum()), 2)
        P.apply_flips(lm, flips)
        self.assertTrue(torch.equal(before.view(torch.int16), after.view(torch.int16)))


class PermutationTests(unittest.TestCase):
    def test_validation_rejects_illegal_orders(self):
        d = P.dims(tiny_lm())
        ok = list(range(d["mlp"]))
        P.validate_spec({"mlp": {"1": ok}}, d)
        bad = [{"mlp": {"1": ok[:-1]}}, {"mlp": {"1": [0] + ok[:-1]}}, {"mlp": {"2": ok}},
               {"mlp": {"01": ok}}, {"vo": {"0": list(range(16, 32)) + list(range(16))
                                            + list(range(32, 64))}},
               {"query": {}}, {"residual": [float(i) for i in range(d["hidden"])]}]
        for spec in bad:
            with self.assertRaises(ValueError, msg=str(spec)[:60]):
                P.validate_spec(spec, d)
        self.assertTrue(P.is_identity(P.validate_spec({"mlp": {"0": ok}}, d)))

    def test_every_family_preserves_the_function_and_inverts_exactly(self):
        rt = TinyRuntime(tiny_lm(torch.float32), [])
        d = P.dims(rt.lm)
        ids = torch.randint(0, 64, (2, 11))
        reference = rt.lm(ids).logits
        original = {n: p.data.clone() for n, p in rt.lm.named_parameters()}
        spec = P.validate_spec(random_spec(d, random.Random(3)), d)
        P.apply_permutation(rt, spec)
        moved = sum(not torch.equal(original[n], p.data) for n, p in rt.lm.named_parameters())
        self.assertEqual(moved, len(original))               # residual touches every tensor
        torch.testing.assert_close(rt.lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)
        self.assertIsNone(rt.residual)
        for name, param in rt.lm.named_parameters():
            self.assertTrue(torch.equal(original[name], param.data), name)

    def test_residual_order_follows_cached_features(self):
        rt = tiny_runtime(torch.float32)
        rows = rt.rows["dev"]
        d = P.dims(rt.lm)
        reference, nll = rt.prefill_logits(rows), rt.nll(rows)
        spec = P.validate_spec(random_spec(d, random.Random(4)), d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(rt.prefill_logits(rows), reference, atol=1e-5, rtol=1e-5)
        for a, b in zip(rt.nll(rows), nll):
            self.assertAlmostEqual(a, b, places=4)
        P.apply_permutation(rt, spec, inverse=True)

    def test_gqa_mlp_and_residual_permutations_preserve_function(self):
        lm = tiny_gqa_lm(torch.float32)
        rt = TinyRuntime(lm, [])
        d = P.dims(lm)
        self.assertEqual(d["kv_heads"], 2)
        self.assertEqual(d["query_group_size"], 2)
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits
        spec = P.validate_spec({
            "residual": list(range(d["hidden"]))[::-1],
            "mlp": {"0": list(range(d["mlp"]))[::-1]},
        }, d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)

    def test_gqa_attention_group_and_value_permutations_preserve_function(self):
        lm = tiny_gqa_lm(torch.float32)
        rt = TinyRuntime(lm, [])
        d = P.dims(lm)
        hd = d["head_dim"]
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits

        # Swap complete query/KV groups.  Heads inside each group stay in
        # order, so every Q head keeps the K/V group it originally used.
        heads = [2, 3, 0, 1]
        spec = P.validate_spec({"heads": {"0": heads}}, d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)

        # Swap value coordinates identically for the two Q heads sharing each
        # KV head.  A different permutation in only one member is rejected.
        local = list(range(hd))[::-1]
        vo = [h * hd + i for h in range(d["heads"]) for i in local]
        spec = P.validate_spec({"vo": {"0": vo}}, d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)

        with self.assertRaisesRegex(ValueError, "complete query groups"):
            P.validate_spec({"heads": {"0": [0, 2, 1, 3]}}, d)
        bad_vo = list(vo)
        bad_vo[hd], bad_vo[hd + 1] = bad_vo[hd + 1], bad_vo[hd]
        with self.assertRaisesRegex(ValueError, "shared within each query group"):
            P.validate_spec({"vo": {"0": bad_vo}}, d)

    def test_gqa_int4_attention_permutation_preserves_scales_and_codes(self):
        lm = tiny_gqa_lm(torch.float32)
        P.quantize(lm, "int4", group=TINY_GROUP)
        rt = TinyRuntime(lm, [])
        d, hd = P.dims(lm), P.dims(lm)["head_dim"]
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits
        before = {name: P.int_codes(lm, name, torch.arange(param.numel())).clone()
                  for name, param in lm.named_parameters()}
        scales = {name: value.clone() for name, value in lm.ar_quant["scales"].items()}
        local = list(range(hd))[::-1]
        vo = [h * hd + i for h in range(d["heads"]) for i in local]
        spec = P.validate_spec({"heads": {"0": [2, 3, 0, 1]}, "vo": {"0": vo}}, d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)
        for name, param in lm.named_parameters():
            self.assertTrue(torch.equal(before[name], P.int_codes(lm, name, torch.arange(param.numel()))), name)
        for name, value in lm.ar_quant["scales"].items():
            self.assertTrue(torch.equal(scales[name], value), name)

    def test_biased_gqa_projections_follow_their_output_axis(self):
        # Qwen-style q/k/v biases are coordinates of the projection output,
        # hence the same row orders that move their sibling weights must move
        # them as well.  INT4 uses one scale for the whole 1-D vector.
        lm = tiny_gqa_lm(torch.float32, attention_bias=True)
        P.quantize(lm, "int4", group=TINY_GROUP)
        rt, d = TinyRuntime(lm, []), P.dims(lm)
        hd = d["head_dim"]
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits
        local = list(range(hd))[::-1]
        vo = [head * hd + i for head in range(d["heads"]) for i in local]
        spec = P.validate_spec({"heads": {"0": [2, 3, 0, 1]}, "vo": {"0": vo}}, d)
        weight_orders = P.axis_orders(spec, d)
        bias_names = [f"model.layers.0.self_attn.{kind}_proj.bias" for kind in "qkv"]
        before = {name: P.int_codes(lm, name, torch.arange(lm.get_parameter(name).numel())).clone()
                  for name in bias_names}
        scales = {name: lm.ar_quant["scales"][name].clone() for name in bias_names}

        P.apply_permutation(rt, spec)
        for name in bias_names:
            weight = name[:-len("bias")] + "weight"
            order = torch.tensor(weight_orders[(weight, 0)])
            actual = P.int_codes(lm, name, torch.arange(lm.get_parameter(name).numel()))
            self.assertTrue(torch.equal(actual, before[name].index_select(0, order)), name)
            self.assertTrue(torch.equal(scales[name], lm.ar_quant["scales"][name]), name)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)

        P.apply_permutation(rt, spec, inverse=True)
        for name in bias_names:
            actual = P.int_codes(lm, name, torch.arange(lm.get_parameter(name).numel()))
            self.assertTrue(torch.equal(actual, before[name]), name)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)

    def test_mqa_attention_permutation_preserves_function(self):
        # MQA is the one-KV-group endpoint of the same shared-KV rule.
        lm = tiny_gqa_lm(torch.float32, kv_heads=1)
        rt = TinyRuntime(lm, [])
        d, hd = P.dims(lm), P.dims(lm)["head_dim"]
        self.assertEqual((d["kv_heads"], d["query_group_size"]), (1, 4))
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits
        local = list(range(hd))[::-1]
        vo = [h * hd + i for h in range(d["heads"]) for i in local]
        spec = P.validate_spec({"heads": {"0": [3, 0, 2, 1]}, "vo": {"0": vo}}, d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)

    def test_layernorm_decoder_keeps_function_codes_and_scales(self):
        # StableLM-style decoders normalise with LayerNorm, a gain and a bias per
        # residual coordinate, carry q/k/v biases and rotate only part of each
        # head.  Every bias must follow the axis that moves its weight; partial
        # RoPE is unaffected because query/key dimensions never move inside a head.
        for kv_heads in (4, 2):
            for fmt in QUANTIZED:
                with self.subTest(kv_heads=kv_heads, fmt=fmt):
                    lm = tiny_layernorm_lm(torch.float32, kv_heads=kv_heads)
                    P.quantize(lm, fmt, group=TINY_GROUP)
                    rt, d, rng = TinyRuntime(lm, []), P.dims(lm), random.Random(11)
                    hd, q = d["head_dim"], d.get("query_group_size", 1)
                    blocks = rng.sample(range(d["heads"] // q), d["heads"] // q)
                    heads = [b * q + i for b in blocks for i in rng.sample(range(q), q)]
                    local = [rng.sample(range(hd), hd) for _ in range(d["heads"] // q)]
                    vo = [h * hd + i for h in range(d["heads"]) for i in local[h // q]]
                    spec = P.validate_spec({
                        "residual": grouped_perm(d["hidden"], TINY_GROUP, rng),
                        "mlp": {"1": grouped_perm(d["mlp"], TINY_GROUP, rng)},
                        "heads": {"0": heads}, "vo": {"0": vo, "1": vo}}, d)
                    orders = P.parameter_axis_orders(lm, spec, d)
                    for name in ("model.norm.bias", "model.layers.1.input_layernorm.bias",
                                 "model.layers.1.post_attention_layernorm.bias",
                                 "model.layers.0.self_attn.q_proj.bias",
                                 "model.layers.0.self_attn.v_proj.bias"):
                        self.assertIn((name, 0), orders)
                    ids = torch.randint(0, 64, (2, 11))
                    reference = lm(ids).logits
                    weights = {n: p.data.clone() for n, p in lm.named_parameters()}
                    codes = {n: P.int_codes(lm, n, torch.arange(p.numel())).view(p.shape)
                             for n, p in lm.named_parameters()}
                    scales = {n: s.clone() for n, s in lm.ar_quant["scales"].items()}
                    P.apply_permutation(rt, spec)
                    torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
                    expected = dict(codes)
                    for (name, axis), order in orders.items():
                        expected[name] = expected[name].index_select(axis, torch.tensor(order))
                    for name, param in lm.named_parameters():
                        actual = P.int_codes(lm, name, torch.arange(param.numel())).view(param.shape)
                        self.assertTrue(torch.equal(actual, expected[name]), name)
                    P.apply_permutation(rt, spec, inverse=True)
                    for name, param in lm.named_parameters():
                        self.assertTrue(torch.equal(weights[name], param.data), name)
                    for name, s in lm.ar_quant["scales"].items():
                        self.assertTrue(torch.equal(scales[name], s), name)

    def test_tied_embedding_and_lm_head_are_permuted_once(self):
        torch.manual_seed(17)
        config = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=4, max_position_embeddings=128,
                             tie_word_embeddings=True)
        lm = LlamaForCausalLM(config).eval().requires_grad_(False)
        rt = TinyRuntime(lm, [])
        d = P.dims(lm)
        ids = torch.randint(0, 64, (2, 11))
        reference = lm(ids).logits
        spec = P.validate_spec({"residual": list(range(d["hidden"]))[::-1]}, d)
        P.apply_permutation(rt, spec)
        torch.testing.assert_close(lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
        P.apply_permutation(rt, spec, inverse=True)


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.rt = tiny_runtime()
        self.name = "model.layers.0.mlp.down_proj.weight"       # 64 x 128

    def check(self, flips, k=16):
        return P.check_attack(self.rt, flips, k)

    def test_location_rules(self):
        n = self.name
        self.assertTrue(self.check([[n, 0, 3], [n, 2047, 2]])["valid"])
        self.assertFalse(self.check([[n, 0, 3], [n, 2048, 9]])["valid"])     # 4097 bytes
        self.assertEqual(self.check([[n, 0, 3], [n, 2047, 2]])["span_bytes"], 4095)
        other = "model.layers.1.mlp.down_proj.weight"
        self.assertFalse(self.check([[n, 0, 3], [other, 0, 3]])["valid"])
        self.assertFalse(self.check([[n, 0, 3], [n, 0, 3]])["valid"])        # duplicate
        self.assertFalse(self.check([[n, i, 0] for i in range(3)], k=2)["valid"])
        self.assertFalse(self.check([[n, 10 ** 9, 0]])["valid"])
        self.assertFalse(self.check([["model.nope.weight", 0, 0]])["valid"])
        self.assertFalse(self.check([[n, 0, 16]])["valid"])

    def test_in_range_is_the_only_value_rule(self):
        W = self.rt.param(self.name).data
        row_max = W[0].abs().max()
        element = int(W[0].abs().argmin())       # tiny weight: raising its exponent is risky
        blowup = [[self.name, element, 14]]     # x2^128: huge or infinite
        result = self.check(blowup)
        self.assertFalse(result["valid"])
        self.assertTrue(any("violates inrange" in reason for reason in result["reasons"]))
        sign = [[self.name, int(W[0].abs().argmax()), 15]]
        self.assertTrue(self.check(sign)["valid"])          # |w'| == row max
        self.assertLessEqual(abs(self.check(sign)["values"][0]["new"]), float(row_max))
        W.view(-1)[5] = torch.tensor(1.5, dtype=torch.bfloat16)          # exponent 127
        self.assertFalse(self.check([[self.name, 5, 14]])["valid"])  # -> inf/nan

    def test_any_bit_may_flip_in_either_direction(self):
        rt = tiny_runtime(fmt="int8")
        name = "model.layers.0.mlp.down_proj.weight"
        index = torch.arange(rt.param(name).numel())
        stored = P.stored_codes(rt.lm, name, index)
        for bit in range(P.code_bits(rt.lm, name)):      # every bit of the int8 code, sign included
            is_one = (stored >> bit) & 1
            for wanted in (0, 1):                # 0 -> 1 and 1 -> 0
                allowed = [int(e) for e in torch.nonzero(is_one == wanted).flatten()
                           if self.rule_allows(rt, name, int(e), bit)]
                self.assertTrue(allowed, (bit, wanted))
                self.assertTrue(P.check_attack(rt, [[name, allowed[0], bit]], 1)["valid"], (bit, wanted))

    def test_tied_alias_without_stored_codes_is_invalid_not_a_crash(self):
        from regionalign.harness import region as R
        for fmt in ("int8", "int4"):
            with self.subTest(fmt=fmt):
                torch.manual_seed(17)
                config = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     num_key_value_heads=4, max_position_embeddings=128,
                                     tie_word_embeddings=True)
                lm = LlamaForCausalLM(config).eval().requires_grad_(False)
                P.quantize(lm, fmt, group=TINY_GROUP)
                rt = TinyRuntime(lm, [])
                self.assertNotIn("lm_head.weight", lm.ar_quant["scales"])     # the alias owns no codes
                result = P.check_attack(rt, [["lm_head.weight", 0, 0]], 1)
                self.assertFalse(result["valid"])
                self.assertTrue(any("missing from" in reason for reason in result["reasons"]))
                self.assertTrue(P.check_attack(rt, [["model.embed_tokens.weight", 0, 0]], 1)["valid"])
                d = P.dims(lm)
                bits = P.FORMAT_BITS[fmt]
                region = R.make_region("down_proj", 0, 0, d, bits=bits, page_bytes=64)
                verdict = R.check_region_attack(rt, region, [["lm_head.weight", 0, 0]], 1)
                self.assertFalse(verdict["valid"])
                self.assertTrue(any("outside the region tensor" in reason for reason in verdict["reasons"]))

    @staticmethod
    def rule_allows(rt, name, element, bit):
        """The in-range rule recomputed from the stored code, without ``check_attack``."""
        W = rt.param(name).data
        index = torch.tensor([element])
        new = P.coded_values(rt.lm, name, index, P.stored_codes(rt.lm, name, index) ^ (1 << bit)).float()
        return bool(torch.isfinite(new)) and float(new.abs()) <= float(W[element // W.shape[1]].abs().max())


class GenerationTests(unittest.TestCase):
    def test_out_of_memory_is_raised_but_kernel_errors_are_failures(self):
        rt = tiny_runtime()
        rows = rt.rows["dev"][:2]
        original = rt.lm.model.forward
        for error, raises in ((torch.cuda.OutOfMemoryError("CUDA out of memory"), True),
                              (RuntimeError("kernel failed on extreme values"), False)):
            def broken(*args, error=error, **kwargs):
                raise error
            rt.lm.model.forward = broken
            try:
                if raises:
                    with self.assertRaises(torch.cuda.OutOfMemoryError):
                        rt.generate(rows)
                else:
                    records = rt.generate(rows)
                    self.assertTrue(all(r["nonfinite"] and r["finish"] == "error" for r in records))
            finally:
                rt.lm.model.forward = original


QUANTIZED = ("int8", "int4")


class QuantTests(unittest.TestCase):
    def test_code_flips_values_rules_and_restore(self):
        for fmt in QUANTIZED:
            with self.subTest(fmt=fmt):
                rt = tiny_runtime(fmt=fmt)
                name = "model.layers.0.mlp.down_proj.weight"          # 64 x 128
                W, bits = rt.param(name).data, P.FORMAT_BITS[fmt]
                qmax = 2 ** (bits - 1) - 1
                codes = P.int_codes(rt.lm, name, torch.arange(W.numel()))
                top = int(torch.nonzero(codes == qmax).flatten()[0])
                zero = int(torch.nonzero(codes == 0).flatten()[0])
                before = W.clone()
                scale = float(P._scales_at(rt.lm, name, torch.tensor([zero])))
                msb = [[name, zero, bits - 1]]                         # 0 -> -2^(bits-1)
                result = P.check_attack(rt, msb, 1)
                self.assertAlmostEqual(result["values"][0]["new"], -(2 ** (bits - 1)) * scale, places=2)
                P.apply_flips(rt.lm, msb)
                self.assertEqual(int(P.int_codes(rt.lm, name, torch.tensor([zero]))), -(2 ** (bits - 1)))
                P.apply_flips(rt.lm, msb)
                self.assertTrue(torch.equal(before.view(torch.int16), W.view(torch.int16)))
                down = [[name, top, bits - 1]]                         # qmax -> -1
                self.assertEqual(int(round(P.check_attack(rt, down, 1)["values"][0]["new"]
                                           / float(P._scales_at(rt.lm, name, torch.tensor([top]))))), -1)
                self.assertFalse(P.check_attack(rt, [[name, 0, bits]], 1)["valid"])
                row_max = float(W.view(-1)[zero // W.shape[1] * W.shape[1]:][:W.shape[1]].abs().max())
                # -2^(bits-1) * s is allowed exactly when it stays within the clean row maximum.
                self.assertEqual(result["valid"], 2 ** (bits - 1) * scale <= row_max, result["reasons"])
                if fmt != "int4":        # int8 has one scale per row: -128 * s exceeds 127 * s
                    self.assertFalse(result["valid"])

    def test_quantized_gain_codes_and_per_vector_scale(self):
        for fmt in QUANTIZED:
            with self.subTest(fmt=fmt):
                rt = tiny_runtime(fmt=fmt)
                n = "model.layers.1.input_layernorm.weight"          # 64 gains, one scale
                g = rt.param(n).data
                bits = P.FORMAT_BITS[fmt]
                scale = float(rt.lm.ar_quant["scales"][n])
                self.assertEqual(P.code_bits(rt.lm, n), bits)
                self.assertAlmostEqual(scale, float(g.abs().max()) / (2 ** (bits - 1) - 1), places=3)
                codes = P.int_codes(rt.lm, n, torch.arange(g.numel()))
                e = int(codes.abs().argmin())
                flip = [[n, e, bits - 2]]                           # sign-preserving code amplification
                result = P.check_attack(rt, flip, 1)
                self.assertTrue(result["valid"], result["reasons"])
                stored = (int(codes[e]) & ((1 << bits) - 1)) ^ (1 << (bits - 2))
                self.assertAlmostEqual(result["values"][0]["new"],
                                       (stored - (1 << bits) * (stored >= 1 << (bits - 1))) * scale, places=2)
                self.assertEqual(result["span_bytes"], 1)
                self.assertFalse(P.check_attack(rt, [[n, e, bits]], 1)["valid"])
                d = P.dims(rt.lm)
                spec = P.validate_spec({"residual": list(range(d["hidden"]))[::-1]}, d)
                P.apply_permutation(rt, spec)
                moved = P.int_codes(rt.lm, n, torch.arange(g.numel()))
                self.assertTrue(torch.equal(moved, codes.flip(0)))
                self.assertEqual(float(rt.lm.ar_quant["scales"][n]), scale)
                P.apply_permutation(rt, spec, inverse=True)
                self.assertTrue(torch.equal(P.int_codes(rt.lm, n, torch.arange(g.numel())), codes))

    def test_missing_tensor_is_rejected_instead_of_falling_back_to_bf16(self):
        rt = tiny_runtime(fmt="int4")
        name = "model.norm.weight"
        rt.lm.ar_quant["scales"].pop(name)
        with self.assertRaisesRegex(ValueError, "incomplete quantization"):
            P.quantization_audit(rt.lm)
        with self.assertRaisesRegex(ValueError, "missing from"):
            P.code_bits(rt.lm, name)

    def test_flips_restore_embedding_head_and_gain_codes(self):
        for fmt in QUANTIZED:
            with self.subTest(fmt=fmt):
                rt = tiny_runtime(fmt=fmt)
                names = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
                snapshots = {n: rt.param(n).data.clone() for n in names}
                flips = [[n, 2, P.FORMAT_BITS[fmt] - 1] for n in names]
                self.assertFalse(P.check_attack(rt, flips, 4)["valid"])     # more than one tensor
                P.apply_flips(rt.lm, flips)
                for n in names:
                    self.assertFalse(torch.equal(snapshots[n], rt.param(n).data), n)
                P.apply_flips(rt.lm, flips)
                for n in names:
                    self.assertTrue(torch.equal(snapshots[n], rt.param(n).data), n)

    def test_permutations_preserve_function_codes_and_scales(self):
        for fmt in QUANTIZED:
            with self.subTest(fmt=fmt):
                rt = tiny_runtime(torch.float32, fmt=fmt)
                d, rng = P.dims(rt.lm), random.Random(5)
                ids = torch.randint(0, 64, (2, 11))
                reference = rt.lm(ids).logits
                weights = {n: p.data.clone() for n, p in rt.lm.named_parameters()}
                original_codes = {n: P.int_codes(rt.lm, n, torch.arange(p.numel())).view(p.shape)
                                  for n, p in rt.lm.named_parameters()}
                scales = {n: s.clone() for n, s in rt.lm.ar_quant["scales"].items()}
                spec = random_spec(d, rng)
                if fmt == "int4":
                    spec["residual"] = grouped_perm(d["hidden"], TINY_GROUP, rng)
                    spec["mlp"] = {"0": grouped_perm(d["mlp"], TINY_GROUP, rng)}
                spec = P.validate_spec(spec, d)
                P.apply_permutation(rt, spec)
                torch.testing.assert_close(rt.lm(ids).logits, reference, atol=1e-5, rtol=1e-5)
                expected_codes = dict(original_codes)
                for (name, axis), order in P.axis_orders(spec, d).items():
                    expected_codes[name] = expected_codes[name].index_select(axis, torch.tensor(order))
                for name, param in rt.lm.named_parameters():
                    if name in rt.lm.ar_quant["scales"]:
                        index = torch.arange(param.numel())
                        codes = P.int_codes(rt.lm, name, index)
                        self.assertTrue(torch.equal(codes.view(param.shape), expected_codes[name]), name)
                        self.assertLessEqual(int(codes.abs().max()), 2 ** (P.FORMAT_BITS[fmt] - 1) - 1)
                        decoded = P.coded_values(rt.lm, name, index, P.stored_codes(rt.lm, name, index))
                        self.assertTrue(torch.equal(decoded, param.data.view(-1)), name)
                P.apply_permutation(rt, spec, inverse=True)
                for name, param in rt.lm.named_parameters():
                    self.assertTrue(torch.equal(weights[name], param.data), name)
                for name, s in rt.lm.ar_quant["scales"].items():
                    self.assertTrue(torch.equal(scales[name], s), name)

    def test_int4_rejects_orders_that_split_groups(self):
        rt = tiny_runtime(fmt="int4")
        d = P.dims(rt.lm)
        inside = list(range(d["hidden"]))
        inside[1], inside[2] = inside[2], inside[1]
        P.validate_spec({"residual": inside}, d)
        across = list(range(d["hidden"]))
        across[1], across[TINY_GROUP + 1] = across[TINY_GROUP + 1], across[1]
        for spec in ({"residual": across}, {"mlp": {"0": across + list(range(d["hidden"], d["mlp"]))}}):
            with self.assertRaises(ValueError, msg=str(spec)[:40]):
                P.validate_spec(spec, d)
        d8 = P.dims(tiny_runtime(fmt="int8").lm)
        self.assertIsNone(d8["group"])
        P.validate_spec({"residual": across}, d8)            # int8 rows: any order is exact


if __name__ == "__main__":
    unittest.main()
