"""Weight formats, legal layout transforms and fault validation for RegionAlign.

This is the whole shared kernel of the package: nothing here depends on code
outside ``regionalign``.  ``region`` adds the region geometry and the three
layout arms on top of it.

Conventions
- Tensor names are relative to a Llama-compatible decoder backbone (for example
  LlamaForCausalLM, Mistral or Qwen2), e.g. ``model.layers.3.mlp.down_proj.weight``.
- Weight formats: ``int8`` and ``int4`` are the formats the experiments use;
  ``bf16`` is the unquantized checkpoint.  In the int formats every decoder
  parameter is stored as two's complement codes with fixed FP32 scales (see
  ``quantize``); element ``e`` of such a tensor is code ``e`` and bit ``b`` is a
  bit of that code (int8: byte ``e``; int4: byte ``e // 2``, low nibble first).
  This includes embeddings, ``lm_head``, all decoder Linears and every RMSNorm
  gain.  The scales are not a fault target.  Computation uses the decoded
  weights in the parameter dtype.  A BF16 element ``e`` occupies bytes ``2e``
  and ``2e + 1`` (bit 15 is the sign).
- A permutation order is always ``new_index -> old_index``.
- A flip is ``[tensor_name, element_index, bit]`` in the coordinates of the
  rearranged (deployed) model.  Flips are XOR and must be distinct.
- Any stored bit may flip, in either direction.  The only value rule is the
  in-range rule of ``value_allowed``: a faulted weight must stay finite and
  within the clean maximum magnitude of its row.
- MHA, GQA and MQA use the same residual/MLP families.  Attention-head orders
  for GQA/MQA are group constrained because Q heads share K/V projections.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import contextmanager


# ---------------------------------------------------------------- constants
WINDOW_BYTES = 4096            # one continuous region of one weight tensor
PAGE_BYTES = 4096
MAX_NEW_TOKENS = 32
GEN_BATCH = 8
LOGIT_ROWS = 8
LOGIT_ATOL, LOGIT_RTOL = 0.5, 0.02
FORMATS = ("bf16", "int8", "int4")
FORMAT_BITS = {"bf16": 16, "int8": 8, "int4": 4}
GROUP = 128                    # int4: input columns that share one scale


# ---------------------------------------------------------------- reporting
def reportable(value):
    """A measurement as JSON can hold it: non-finite becomes null, not NaN.

    Reports are written with ``allow_nan=False`` on purpose, so one NaN or inf
    anywhere in a report raises instead of saving it. Two things in a report are
    measurements of a *faulted* model and may legitimately come out non-finite:
    a faulted weight value, and the logit distance of a permutation that did not
    survive. Both already make the attack invalid; losing the whole report on
    top of that would discard the evidence of why. Verdicts are decided on the
    raw floats, only the stored copy is sanitised.
    """
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


# ---------------------------------------------------------------- stored bits and the value rule
def bit_mask(bit):
    """int16 XOR mask for one BF16 bit (bit 15 is the sign)."""
    if isinstance(bit, bool) or not isinstance(bit, int) or not 0 <= bit < 16:
        raise ValueError(f"BF16 bit must be an int in [0, 16), got {bit!r}")
    return (1 << bit) - (1 << 16 if bit == 15 else 0)


def row_absmax(W):
    """Clean per-row bound max|w| as float32: one per output row (2-D) or one for a vector."""
    rows = W.reshape(1, -1) if W.dim() == 1 else W
    return rows.abs().amax(dim=1).float()


def value_allowed(new, bound):
    """The in-range rule for faulted values ``new`` (float32).

    A faulted weight must stay finite and satisfy ``|w'| <= bound``, where
    ``bound`` is the clean ``max|w|`` of its row (see ``row_absmax``).  The rule
    looks at the value only: it does not depend on which cell holds the bit or
    on the direction of the flip.
    """
    import torch
    return torch.isfinite(new) & (new.abs() <= bound)


# ---------------------------------------------------------------- weight formats
def quantize(lm, fmt, group=GROUP):
    """Weight-only symmetric round-to-nearest quantization, in place.

    Every Llama parameter is quantized. INT8 uses one FP32 scale per output
    row; INT4 uses one per output row and group of ``group`` input columns,
    including a shorter final group when needed. Both formats use one scale
    for an entire 1-D gain vector, so a residual permutation preserves its
    codes. Codes are two's complement in [-(2^(b-1)-1), 2^(b-1)-1].
    The weights then hold the decoded values ``q * s`` in their own dtype, so
    the compute path is W8A16/W4A16 in BF16, and the codes are recovered exactly
    as ``round(w / s)`` (BF16 rounding moves ``q * s`` by less than half a code
    step for |q| <= 128). Integer codes have no negative zero, so a zero code
    always decodes to +0. The scales live in ``lm.ar_quant`` and move with every
    permutation (a vector's single scale stays put).
    """
    import torch
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    if getattr(lm, "ar_quant", None) is not None:
        raise RuntimeError("model is already quantized")
    if fmt == "bf16":
        return
    if isinstance(group, bool) or not isinstance(group, int) or group < 1:
        raise ValueError("quantization group must be a positive integer")
    params = list(lm.named_parameters())
    unsupported = [name for name, p in params if p.dim() not in (1, 2) or p.numel() == 0]
    if unsupported:
        raise ValueError(f"unsupported parameter shapes: {unsupported}")
    bits = FORMAT_BITS[fmt]
    qmax = 2 ** (bits - 1) - 1
    scales = {}
    for name, param in params:
        rows, cols = param.shape if param.dim() == 2 else (1, param.numel())
        g = min(group, cols) if fmt == "int4" and param.dim() == 2 else cols
        # Padding affects only scale reduction; no padded codes become weights.
        v = param.data.float().view(rows, cols)
        padding = (-cols) % g
        if padding:
            v = torch.nn.functional.pad(v, (0, padding))
        v = v.view(rows, -1, g)
        s = (v.abs().amax(-1) / qmax).clamp_min(1e-12)
        q = torch.round(v / s[..., None]).clamp(-qmax, qmax) + 0.0     # +0.0 turns -0 into +0
        param.data.copy_((q * s[..., None]).view(rows, -1)[:, :cols].reshape(param.shape).to(param.dtype))
        scales[name] = s.contiguous()
    lm.ar_quant = {"format": fmt, "bits": bits, "group": group if fmt == "int4" else None,
                   "scope": "all_llm_parameters", "schema_version": 2, "scales": scales}
    quantization_audit(lm)


def quantization_audit(lm):
    """Verify complete integer-code coverage and report its parameter-level evidence."""
    params = dict(lm.named_parameters())
    quant = getattr(lm, "ar_quant", None)
    total = sum(p.numel() for p in params.values())
    fmt = weight_format(lm)
    names = set(quant["scales"]) if quant else set()
    if quant:
        if quant.get("scope") != "all_llm_parameters" or quant.get("schema_version") != 2:
            raise ValueError("quantization must use schema 2 for all Llama parameters")
        if names != set(params):
            raise ValueError(f"incomplete quantization: missing={sorted(set(params) - names)}, "
                             f"unknown={sorted(names - set(params))}")
        if fmt not in ("int8", "int4") or quant["bits"] != FORMAT_BITS[fmt]:
            raise ValueError("invalid integer quantization format/bit width")
        if fmt == "int4" and (isinstance(quant["group"], bool) or
                              not isinstance(quant["group"], int) or quant["group"] < 1):
            raise ValueError("invalid INT4 scale group")
        for name, param in params.items():
            rows, cols = param.shape if param.dim() == 2 else (1, param.numel())
            g = quant["group"] if quant["group"] and param.dim() == 2 else cols
            expected = (rows, math.ceil(cols / g))
            if tuple(quant["scales"][name].shape) != expected:
                raise ValueError(f"{name}: invalid scales shape; expected {expected}")
    encoded = sum(params[name].numel() for name in names)
    return {"format": fmt, "scope": "all_llm_parameters", "schema_version": 2,
            "bits": FORMAT_BITS[fmt], "total_tensors": len(params), "total_parameters": total,
            "quantized_tensors": len(names), "quantized_parameters": encoded,
            "quantized_fraction": encoded / total if total else 0.0,
            "unquantized_tensors": sorted(set(params) - names),
            "int4_group_columns": quant["group"] if quant else None,
            "gain_scale": "per_vector", "scales_faultable": False,
            "compute": "decoded_weights_in_parameter_dtype"}


def weight_format(lm):
    return (getattr(lm, "ar_quant", None) or {}).get("format", "bf16")


def code_bits(lm, name):
    """Bits per stored element of tensor ``name``: 16 for BF16 tensors, else 8 or 4."""
    quant = getattr(lm, "ar_quant", None)
    if quant:
        if name not in quant["scales"]:
            raise ValueError(f"{name} is missing from the full-parameter quantization")
        return quant["bits"]
    return 16


def _scales_at(lm, name, index):
    quant = lm.ar_quant
    param = lm.get_parameter(name)
    cols = param.shape[-1]                         # a vector is one row
    group = quant["group"] if param.dim() == 2 and quant["group"] else cols
    return quant["scales"][name][index // cols, (index % cols) // group]


def int_codes(lm, name, index):
    """Signed codes of a quantized tensor at flat ``index`` (int64)."""
    import torch
    W = lm.get_parameter(name).data.view(-1)
    return torch.round(W[index].float() / _scales_at(lm, name, index)).long()


def stored_codes(lm, name, index):
    """Unsigned stored code per element: the BF16 word or the int8/int4 two's complement code."""
    import torch
    bits = code_bits(lm, name)
    if bits == 16:
        return lm.get_parameter(name).data.view(-1).view(torch.int16)[index].long() & 0xFFFF
    return int_codes(lm, name, index) & ((1 << bits) - 1)


def decode_constants(lm, name, index):
    """Decode constants ``(bits, scales, dtype)`` at flat ``index`` -- the parts of
    ``coded_values`` that depend only on ``name`` and ``index``, never on the codes.
    A per-bit sweep over one fixed ``index`` computes these once and passes them to
    ``coded_values`` instead of having each call re-derive them; the result is the
    same either way. ``scales`` and ``dtype`` are ``None`` for an unquantized tensor."""
    bits = code_bits(lm, name)
    if bits == 16:
        return bits, None, None
    return bits, _scales_at(lm, name, index), lm.get_parameter(name).dtype


def coded_values(lm, name, index, codes, bits=None, scales=None, dtype=None):
    """Decoded weights (tensor dtype) for unsigned stored ``codes`` at flat ``index``.

    ``bits``, ``scales`` and ``dtype`` are the index-dependent decode constants
    (see ``decode_constants``); when omitted they are looked up here, so passing
    them in is only a speed-up for repeated decodes at one ``index`` and never
    changes the result."""
    import torch
    if bits is None:
        bits = code_bits(lm, name)
    signed = codes - (codes >= 1 << (bits - 1)).long() * (1 << bits)
    if bits == 16:
        return signed.to(torch.int16).view(torch.bfloat16)
    if scales is None:
        scales = _scales_at(lm, name, index)
    if dtype is None:
        dtype = lm.get_parameter(name).dtype
    return (signed.float() * scales).to(dtype)


def group_preserving(order, group):
    """True if every destination group of ``group`` columns comes from one source group."""
    return all(len({i // group for i in order[start:start + group]}) == 1 and
               len(order[start:start + group]) == min(group, len(order) - order[start] // group * group)
               for start in range(0, len(order), group))


# ---------------------------------------------------------------- legal permutations
def dims_from_config(config, quant_group=None):
    """Return the dimension contract used by legal layout transforms.

    ``num_key_value_heads`` is absent on older MHA checkpoints, where it is
    exactly ``num_attention_heads``.  For GQA/MQA, attention-head transforms
    are restricted to whole query-head groups; the MLP and residual families
    remain unchanged.  Keeping this description in one place lets planning
    code validate a checkpoint without loading its weights.
    """
    layers = getattr(config, "num_hidden_layers", None)
    hidden = getattr(config, "hidden_size", None)
    mlp = getattr(config, "intermediate_size", None)
    heads = getattr(config, "num_attention_heads", None)
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is None:
        # Older MHA configs omit the field; an explicit null has the same
        # meaning. A present zero is still rejected below instead of being
        # silently treated as MHA.
        kv_heads = heads
    if any(value is None for value in (layers, hidden, mlp, heads, kv_heads)):
        raise ValueError("checkpoint lacks decoder dimensions required by the layout transform")
    if not all(isinstance(value, int) and value > 0 for value in
               (layers, hidden, mlp, heads, kv_heads)):
        raise ValueError("decoder dimensions must be positive integers")
    if hidden % heads or heads % kv_heads:
        raise ValueError("hidden_size must divide attention heads and query groups must divide heads")
    out = {"layers": layers, "hidden": hidden, "mlp": mlp, "heads": heads,
           "head_dim": hidden // heads, "group": quant_group}
    if kv_heads != heads:
        out["kv_heads"] = kv_heads
        out["query_group_size"] = heads // kv_heads
    return out


def dims(lm):
    """Return layout dimensions for MHA, GQA, and MQA decoder checkpoints.

    This function deliberately does not claim that every Transformer is
    supported.  ``axis_orders`` still requires the Llama-compatible parameter
    names used by the runtime (``model.layers.*.self_attn.*`` and
    ``model.layers.*.mlp.*``); an unsupported naming scheme fails explicitly
    when its parameter map is materialized.
    """
    config = lm.config
    quant_group = (getattr(lm, "ar_quant", None) or {}).get("group")
    return dims_from_config(config, quant_group=quant_group)


def validate_parameter_layout(lm, d=None):
    """Check that a loaded checkpoint has the Llama-compatible tensor schema.

    The attention equations are shared by many decoder families, but parameter
    names are not.  We support models that expose the standard Hugging Face
    ``model.layers.*`` schema (including common Llama/Mistral/Qwen2 variants)
    and fail early for other schemas instead of silently permuting the wrong
    axes.
    """
    d = d or dims(lm)
    required = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for layer in range(d["layers"]):
        prefix = f"model.layers.{layer}."
        required.extend(prefix + name + ".weight" for name in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "input_layernorm", "post_attention_layernorm"))
    try:
        params = {name: lm.get_parameter(name) for name in required}
    except (AttributeError, KeyError) as error:
        raise ValueError(
            "unsupported decoder parameter layout; expected model.layers.* "
            "with separate q_proj/k_proj/v_proj/o_proj and gate/up/down projections"
        ) from error
    hd, kv_heads = d["head_dim"], d.get("kv_heads", d["heads"])
    expected = {
        "model.embed_tokens.weight": (None, d["hidden"]),
        "lm_head.weight": (None, d["hidden"]),
        "model.norm.weight": (d["hidden"],),
    }
    for layer in range(d["layers"]):
        prefix = f"model.layers.{layer}."
        expected.update({
            prefix + "self_attn.q_proj.weight": (d["heads"] * hd, d["hidden"]),
            prefix + "self_attn.k_proj.weight": (kv_heads * hd, d["hidden"]),
            prefix + "self_attn.v_proj.weight": (kv_heads * hd, d["hidden"]),
            prefix + "self_attn.o_proj.weight": (d["hidden"], d["heads"] * hd),
            prefix + "mlp.gate_proj.weight": (d["mlp"], d["hidden"]),
            prefix + "mlp.up_proj.weight": (d["mlp"], d["hidden"]),
            prefix + "mlp.down_proj.weight": (d["hidden"], d["mlp"]),
            prefix + "input_layernorm.weight": (d["hidden"],),
            prefix + "post_attention_layernorm.weight": (d["hidden"],),
        })
    for name, shape in expected.items():
        actual = tuple(params[name].shape)
        if shape[0] is None:
            if len(actual) != 2 or actual[1] != shape[1]:
                raise ValueError(f"unexpected shape for {name}: {actual}")
        elif actual != shape:
            raise ValueError(f"unexpected shape for {name}: {actual}, expected {shape}")
    return d


def _order(values, size, what):
    if hasattr(values, "tolist"):
        values = values.tolist()
    values = list(values)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError(f"{what} must contain integers")
    if len(values) != size or sorted(values) != list(range(size)):
        raise ValueError(f"{what} must be a permutation of range({size})")
    return values


def _layer(key, layers):
    if isinstance(key, bool):
        raise ValueError(f"bad layer key {key!r}")
    layer = int(key)
    if str(layer) != str(key) or not 0 <= layer < layers:
        raise ValueError(f"bad layer key {key!r}")
    return layer


def validate_spec(spec, d):
    """Normalize a permutation spec; raise ValueError if it is not a legal layout.

    Families (each optional): ``residual`` (one global order of the hidden
    dimension), ``mlp`` {layer: order of the MLP channels}, ``heads`` {layer:
    order of attention heads}, ``vo`` {layer: order of value dimensions that
    keeps every dimension inside its head}. Query/key dimensions inside a head
    cannot move because RoPE ties them to fixed frequencies. Under int4
    (``d["group"]``) every order applied to the input columns of a quantized
    tensor must keep each scale group together (move whole groups, or columns
    inside their group); otherwise the stored codes would change.
    """
    spec = spec or {}
    if not isinstance(spec, dict) or set(spec) - {"residual", "mlp", "heads", "vo"}:
        raise ValueError("permutation keys must be among residual, mlp, heads, vo")
    out = {"residual": None, "mlp": {}, "heads": {}, "vo": {}}
    if spec.get("residual") is not None:
        out["residual"] = _order(spec["residual"], d["hidden"], "residual")
    for family, size in (("mlp", d["mlp"]), ("heads", d["heads"]), ("vo", d["hidden"])):
        orders = spec.get(family) or {}
        if not isinstance(orders, dict):
            raise ValueError(f"{family} must map layer -> order")
        for key, values in orders.items():
            layer = _layer(key, d["layers"])
            out[family][layer] = _order(values, size, f"{family}[{layer}]")
    head_dim = d["head_dim"]
    query_groups = d.get("query_group_size", 1)
    for layer, order in out["heads"].items():
        if query_groups > 1:
            # A GQA KV head is shared by a contiguous block of query heads.
            # A legal head permutation may reorder whole blocks, and may
            # reorder heads inside a block only when the complete block is
            # moved.  Mixing two old KV groups in one new block would change
            # which K/V states a query uses.
            block = query_groups
            old_groups = [old // block for old in order]
            if any(len(set(old_groups[i:i + block])) != 1
                   for i in range(0, len(old_groups), block)):
                raise ValueError("GQA head order must preserve complete query groups")
            if sorted(old_groups[::block]) != list(range(d["kv_heads"])):
                raise ValueError("GQA head order must permute KV groups exactly once")
    for layer, order in out["vo"].items():
        if any(old // head_dim != new // head_dim for new, old in enumerate(order)):
            raise ValueError(f"vo[{layer}] moves a value dimension across heads; use heads")
        if query_groups > 1:
            # All query heads sharing one KV head see the same value vector;
            # their local value-coordinate permutation must therefore agree.
            for start in range(0, d["heads"], query_groups):
                local = [order[(start + i) * head_dim:(start + i + 1) * head_dim]
                         for i in range(query_groups)]
                if any([x % head_dim for x in part] !=
                       [x % head_dim for x in local[0]] for part in local[1:]):
                    raise ValueError("GQA value order must be shared within each query group")

    def identity(order):
        return all(new == old for new, old in enumerate(order))
    if out["residual"] is not None and identity(out["residual"]):
        out["residual"] = None
    for family in ("mlp", "heads", "vo"):
        out[family] = {k: v for k, v in out[family].items() if not identity(v)}
    if d.get("group"):
        checked = set()
        for (name, axis), order in axis_orders(out, d).items():
            if axis != 1 or id(order) in checked:
                continue
            checked.add(id(order))
            if not group_preserving(order, d["group"]):
                raise ValueError(f"order on {name} splits int4 scale groups of {d['group']}")
    return out


def is_identity(spec):
    return spec["residual"] is None and not (spec["mlp"] or spec["heads"] or spec["vo"])


def spec_to_json(spec):
    return {"residual": spec["residual"],
            **{f: {str(k): v for k, v in spec[f].items()} for f in ("mlp", "heads", "vo")}}


def axis_orders(spec, d):
    """{(tensor_name, axis): new->old order} for every tensor axis the spec moves."""
    orders = {}

    def put(name, axis, order):
        if (name, axis) in orders:
            raise AssertionError(f"axis moved twice: {name}:{axis}")
        orders[(name, axis)] = order
    residual = spec["residual"]
    if residual is not None:
        put("model.embed_tokens.weight", 1, residual)
        put("lm_head.weight", 1, residual)
        put("model.norm.weight", 0, residual)
        for layer in range(d["layers"]):
            prefix = f"model.layers.{layer}."
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "mlp.gate_proj", "mlp.up_proj"):
                put(prefix + name + ".weight", 1, residual)
            for name in ("self_attn.o_proj", "mlp.down_proj", "input_layernorm",
                         "post_attention_layernorm"):
                put(prefix + name + ".weight", 0, residual)
    for layer, order in spec["mlp"].items():
        prefix = f"model.layers.{layer}.mlp."
        put(prefix + "gate_proj.weight", 0, order)
        put(prefix + "up_proj.weight", 0, order)
        put(prefix + "down_proj.weight", 1, order)
    hd = d["head_dim"]
    kv_heads = d.get("kv_heads", d["heads"])
    query_group_size = d.get("query_group_size", 1)
    for layer in sorted(set(spec["heads"]) | set(spec["vo"])):
        prefix = f"model.layers.{layer}.self_attn."
        heads = spec["heads"].get(layer)
        qk = [h * hd + i for h in heads for i in range(hd)] if heads else None
        values = spec["vo"].get(layer) or list(range(d["hidden"]))
        vo = [values[j] for j in qk] if qk else values    # value dims first, then heads
        if qk:
            put(prefix + "q_proj.weight", 0, qk)
            if query_group_size == 1:
                put(prefix + "k_proj.weight", 0, qk)
                put(prefix + "v_proj.weight", 0, vo)
            else:
                group_order = [h // query_group_size
                               for h in heads[::query_group_size]]
                kv_order = [g * hd + i for g in group_order for i in range(hd)]
                put(prefix + "k_proj.weight", 0, kv_order)
                # ``values`` is indexed by old query-head coordinates.  It is
                # shared within each old group, so one representative head
                # supplies the row order for that group's V projection.
                v_order = []
                for g in group_order:
                    base = g * query_group_size * hd
                    v_order.extend(values[base + i] % hd + g * hd for i in range(hd))
                put(prefix + "v_proj.weight", 0, v_order)
        elif spec["vo"].get(layer) is not None and query_group_size > 1:
            values = spec["vo"][layer]
            v_order = []
            for g in range(kv_heads):
                base = g * query_group_size * hd
                v_order.extend(values[base + i] % hd + g * hd for i in range(hd))
            put(prefix + "v_proj.weight", 0, v_order)
        elif spec["vo"].get(layer) is not None:
            put(prefix + "v_proj.weight", 0, vo)
        put(prefix + "o_proj.weight", 1, vo)
    return orders


def parameter_axis_orders(lm, spec, d):
    """Axis orders for stored parameters, including optional module biases.

    ``axis_orders`` describes the algebraic weight transform independently of a
    checkpoint.  Some compatible decoders (notably Qwen2) also store biases on
    q/k/v projections.  A Linear bias is indexed by its output coordinate, so it
    must follow exactly the same order whenever axis 0 of the sibling weight is
    moved.  Input-axis permutations leave it unchanged.  Expanding the orders
    from the loaded parameter map keeps bias-free Llama checkpoints byte-for-byte
    on their historical path while making biased compatible decoders exact.
    """
    orders = axis_orders(spec, d)
    expanded = dict(orders)
    for (name, axis), order in orders.items():
        if axis != 0 or not name.endswith(".weight"):
            continue
        bias_name = name[:-len("weight")] + "bias"
        try:
            bias = lm.get_parameter(bias_name)
        except (AttributeError, KeyError):
            continue
        if bias.dim() != 1 or bias.numel() != len(order):
            raise ValueError(
                f"unexpected shape for {bias_name}: {tuple(bias.shape)}, expected ({len(order)},)")
        expanded[(bias_name, 0)] = order
    return expanded


def apply_permutation(rt, spec, inverse=False):
    """Rearrange ``rt.lm`` in place and track the image-feature order in ``rt.residual``.

    Function-preserving up to floating-point summation order. Calling it again
    with ``inverse=True`` restores the original codes exactly.
    """
    import torch
    d = validate_parameter_layout(rt.lm)
    quant = getattr(rt.lm, "ar_quant", None)
    seen = {}
    for (name, axis), order in parameter_axis_orders(rt.lm, spec, d).items():
        param = rt.lm.get_parameter(name)
        # Some decoder checkpoints tie ``lm_head.weight`` to
        # ``model.embed_tokens.weight``. Both logical uses need the same
        # hidden-axis permutation, but the shared Parameter must be moved only
        # once; applying it twice would undo the layout transform. Keep this
        # explicit rather than relying on ``named_parameters`` de-duplication,
        # because axis_orders intentionally contains both logical names.
        key = (id(param), axis)
        previous = seen.get(key)
        if previous is not None:
            if previous != order:
                raise ValueError(f"aliased parameter {name} has conflicting axis orders")
            continue
        seen[key] = order
        index = torch.as_tensor(order, dtype=torch.long, device=param.device)
        if inverse:
            index = torch.argsort(index)
        param.data = param.data.index_select(axis, index).contiguous()
        if quant and name in quant["scales"] and param.dim() == 2:   # scales follow rows and groups
            scales = quant["scales"][name]
            if axis == 0:
                quant["scales"][name] = scales.index_select(0, index).contiguous()
            elif quant["group"]:
                g = quant["group"]
                assert group_preserving(index.tolist(), g), name
                quant["scales"][name] = scales.index_select(1, index[::g] // g).contiguous()
    if spec["residual"] is not None:
        index = torch.as_tensor(spec["residual"], dtype=torch.long)
        if inverse:
            index = torch.argsort(index)
        current = rt.residual if rt.residual is not None else torch.arange(d["hidden"])
        moved = current[index]
        rt.residual = None if torch.equal(moved, torch.arange(d["hidden"])) else moved


# ---------------------------------------------------------------- flips
def flip_byte(element, bit, bits=16):
    """Byte offset of a stored bit: BF16 words, int8 bytes, int4 nibbles (low nibble first)."""
    return {16: 2 * element + bit // 8, 8: element, 4: element // 2}[bits]


def normalize_flips(flips):
    """Type-check flips and return them as (tensor, element, bit) tuples."""
    if not isinstance(flips, (list, tuple)):
        raise ValueError("flips must be a list of [tensor, element, bit]")
    out = []
    for flip in flips:
        if not isinstance(flip, (list, tuple)) or len(flip) != 3:
            raise ValueError(f"bad flip {flip!r}; expected [tensor, element, bit]")
        name, element, bit = flip
        if not isinstance(name, str):
            raise ValueError(f"bad tensor name {name!r}")
        if isinstance(element, bool) or not isinstance(element, int) or element < 0:
            raise ValueError(f"bad element index {element!r}")
        bit_mask(bit)
        out.append((name, element, bit))
    if len(set(out)) != len(out):
        raise ValueError("flips must be distinct")
    return out


def _element_masks(flips):
    """{tensor: {element: unsigned XOR mask of its stored code}}."""
    masks = defaultdict(dict)
    for name, element, bit in flips:
        masks[name][element] = masks[name].get(element, 0) ^ (1 << bit)
    return masks


def apply_flips(lm, flips):
    """XOR the bits of the stored codes in place; applying the same flips again restores them.

    BF16 tensors flip their own words; quantized tensors flip the int8/int4 code
    and store the decoded value, which is the same computation an integer
    kernel performs on the faulted code.
    """
    import torch
    for name, masks in _element_masks(flips).items():
        W = lm.get_parameter(name).data.view(-1)
        index = torch.tensor(list(masks), dtype=torch.long, device=W.device)
        mask = torch.tensor(list(masks.values()), dtype=torch.long, device=W.device)
        W[index] = coded_values(lm, name, index, stored_codes(lm, name, index) ^ mask)


def check_attack(rt, flips, k):
    """Validate one attack against the deployed model (rearranged, unfaulted).

    Returns a dict with ``valid``, ``reasons`` and location statistics. The
    rules are the generic part of the threat model: at most ``k`` distinct XOR
    bits, all in one tensor within ``WINDOW_BYTES`` bytes, and every faulted
    weight passing the in-range rule of ``value_allowed``.  Membership in the
    preregistered region is checked by ``region.check_region_attack``.
    """
    import torch
    info = {"k": k, "n_flips": 0, "valid": False, "reasons": [], "tensors": [],
            "span_bytes": None, "single_page": None}
    try:
        flips = normalize_flips(flips)
    except ValueError as error:
        info["reasons"].append(str(error))
        return info
    info["n_flips"], info["flips"] = len(flips), [list(f) for f in flips]
    reasons = info["reasons"]
    if not flips:
        reasons.append("no flips")
    if len(flips) > k:
        reasons.append(f"{len(flips)} flips exceed budget k={k}")
    names = sorted({f[0] for f in flips})
    info["tensors"] = names
    params = {}
    for name in names:
        try:
            params[name] = rt.lm.get_parameter(name)
            code_bits(rt.lm, name)
        except AttributeError:
            reasons.append(f"unknown tensor {name}")
        except ValueError as error:     # a name that owns no stored codes, e.g. a tied alias
            del params[name]
            reasons.append(str(error))
    for name, element, bit in flips:
        if name in params and element >= params[name].numel():
            reasons.append(f"element {element} outside {name}")
        if name in params and bit >= code_bits(rt.lm, name):
            reasons.append(f"bit {bit} outside the {code_bits(rt.lm, name)}-bit code of {name}")
    if reasons:
        return info
    if len(names) > 1:
        reasons.append("local attacks must stay inside one tensor")
    by_tensor = defaultdict(list)
    for name, element, bit in flips:
        by_tensor[name].append(flip_byte(element, bit, code_bits(rt.lm, name)))
    spans = {n: max(b) - min(b) + 1 for n, b in by_tensor.items()}
    pages = {n: len({b // PAGE_BYTES for b in v}) for n, v in by_tensor.items()}
    info["span_bytes"] = max(spans.values())
    info["single_page"] = all(p == 1 for p in pages.values())
    if info["span_bytes"] > WINDOW_BYTES:
        reasons.append(f"span {info['span_bytes']}B exceeds {WINDOW_BYTES}B")
    values = []
    for name, masks in _element_masks(flips).items():
        param = params[name].data
        index = torch.tensor(list(masks), dtype=torch.long, device=param.device)
        stored = stored_codes(rt.lm, name, index)
        old = param.view(-1)[index].float()
        new = coded_values(rt.lm, name, index, stored ^ torch.tensor(
            list(masks.values()), dtype=torch.long, device=param.device)).float()
        bounds = row_absmax(param)
        bound = bounds[index // param.shape[1]] if param.dim() == 2 else bounds.expand_as(new)
        ok = value_allowed(new, bound)
        for i, element in enumerate(masks):
            values.append({"tensor": name, "element": element,
                           "old": reportable(float(old[i])),
                           "new": reportable(float(new[i])),
                           "bound": reportable(float(bound[i]))})
            if not bool(ok[i]):
                reasons.append(f"{name}[{element}] -> {float(new[i]):.4g} violates inrange")
    info["values"] = values
    info["valid"] = not reasons
    return info


# ---------------------------------------------------------------- loss
def distinct_rows(rows):
    """``(first, expand)``: the first position of each distinct row and every row's index
    into ``first``, or None when a row is not plain JSON data.

    Rows are distinct by content, so the repeats of a with-replacement bootstrap
    panel collapse onto one row.
    """
    first, expand, seen = [], [], {}
    for index, row in enumerate(rows):
        try:
            key = json.dumps(row, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return None
        position = seen.setdefault(key, len(first))
        if position == len(first):
            first.append(index)
        expand.append(position)
    return first, expand


def mean_nll(rt, rows, bs=4, distinct=None):
    """Mean teacher-forced NLL over ``rows``; nan as soon as one row is not finite.

    With ``distinct`` (from ``distinct_rows``; batch size one only) each distinct row
    is evaluated once and its value is reused at every repeat.  A row's value does not
    depend on the other rows at batch size one, and the sum keeps the original row
    order, so the mean is bit-identical to evaluating every row.
    """
    if not rows:
        return 0.0
    if distinct is None:
        values = rt.nll(rows, bs=bs)
    else:
        if bs != 1:
            raise ValueError("distinct-row reuse requires batch size one")
        first, expand = distinct
        unique = rt.nll([rows[i] for i in first], bs=1)
        values = [unique[i] for i in expand]
    return sum(values) / len(values) if all(math.isfinite(v) for v in values) else float("nan")


class _StopForward(Exception):
    """Ends a capture pass once the residual stream at the cut layer is recorded."""


# ---------------------------------------------------------------- runtime
class BaseRuntime:
    """Model-agnostic NLL, gradients and greedy generation over embedded rows.

    Subclasses provide ``lm``, ``device``, ``eos``, ``residual`` and
    ``seq_embeds/answer_ids/decode/score``. ``task`` names what ``score`` means;
    ``max_new_tokens`` is the decode protocol of that task.
    """
    residual = None
    task = "vqa"
    max_new_tokens = MAX_NEW_TOKENS
    _prefix = None              # the reuse ``reusing`` activates; see ``prefix_states``

    def param(self, name):
        return self.lm.get_parameter(name)

    def _pad(self, seqs, left):
        import torch
        length = max(len(s) for s in seqs)
        x = seqs[0].new_zeros((len(seqs), length, seqs[0].shape[-1]))
        mask = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        for i, s in enumerate(seqs):
            if left:
                x[i, length - len(s):], mask[i, length - len(s):] = s, 1
            else:
                x[i, :len(s)], mask[i, :len(s)] = s, 1
        return x, mask

    def token_weights(self, row, n):
        """Per-token weights of one row's answer loss; None is the uniform mean.

        This is the task's loss function, and the only place a task may change
        what ``nll`` and ``grads`` optimise. The default leaves it uniform.
        """
        return None

    def _batch(self, rows):
        """Right-padded input embeddings, mask and per-row answer spans of ``rows``."""
        seqs, spans = [], []
        for row in rows:
            answer = self.answer_ids(row)
            embeds, n_ctx = self.seq_embeds(row, answer)
            seqs.append(embeds)
            spans.append((n_ctx, answer.to(self.device), self.token_weights(row, len(answer))))
        x, mask = self._pad(seqs, left=False)
        return x, mask, spans

    def _hidden(self, rows, x, mask):
        """Final decoder states; one row with an active prefix state starts at its layer."""
        prefix = self._prefix
        state = (prefix["states"].get(id(rows[0]))
                 if prefix is not None and len(rows) == 1 else None)
        if state is None:
            return self.lm.model(inputs_embeds=x, attention_mask=mask, use_cache=False)[0]
        if state.shape != x.shape or state.dtype != x.dtype:
            raise RuntimeError("a reused prefix state does not match its row")
        model = self.lm.model
        layers = model.layers
        model.layers = layers[prefix["layer"]:]
        try:
            return model(inputs_embeds=state, attention_mask=mask, use_cache=False)[0]
        finally:
            model.layers = layers

    def prefix_states(self, rows, layer):
        """Record the residual stream entering decoder layer ``layer`` for single-row passes.

        A fault confined to layer ``layer`` or later leaves every earlier layer's output
        unchanged, so a one-row pass may start at ``layer`` from the recorded state and
        run the same kernels on the same inputs as the full pass.  The states are
        recorded once on the current model; they are accepted only if every row's reused
        NLL equals its full-pass NLL bit for bit.  They stay valid only while the
        parameters before ``layer`` and the row objects are unchanged.

        Returns ``(prefix, status)``: ``prefix`` maps ``id(row)`` to its state under
        ``"states"`` (activate it with ``reusing``), or is None when ``status`` is
        ``"unavailable"`` (layer 0 or no decoder stack: nothing to save) or
        ``"mismatch"``.
        """
        import torch
        layers = getattr(getattr(self.lm, "model", None), "layers", None)
        if layers is None or not 0 < layer < len(layers):
            return None, "unavailable"
        recorded = []

        def capture(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            recorded.append(hidden.detach().clone())
            raise _StopForward

        states = {}
        with torch.no_grad():
            handle = layers[layer].register_forward_pre_hook(capture, with_kwargs=True)
            try:
                for row in rows:
                    x, mask, _ = self._batch([row])
                    try:
                        self.lm.model(inputs_embeds=x, attention_mask=mask, use_cache=False)
                    except _StopForward:
                        pass
                    if len(recorded) != 1:
                        return None, "unavailable"
                    states[id(row)] = recorded.pop()
            finally:
                handle.remove()
            prefix = {"layer": layer, "states": states}
            for row in rows:
                full = self._nll_batch([row])
                with self.reusing(prefix):
                    reused = self._nll_batch([row])
                if not torch.equal(full, reused):
                    return None, "mismatch"
        return prefix, "verified"

    @contextmanager
    def reusing(self, prefix):
        """Let one-row passes over the rows of ``prefix`` start at its layer."""
        previous, self._prefix = self._prefix, prefix
        try:
            yield
        finally:
            self._prefix = previous

    def _nll_batch(self, rows):
        import torch
        import torch.nn.functional as F
        x, mask, spans = self._batch(rows)
        hidden = self._hidden(rows, x, mask)
        losses = []
        for i, (n, a, w) in enumerate(spans):
            logits = self.lm.lm_head(hidden[i, n - 1:n - 1 + len(a)]).float()
            if w is None:
                losses.append(F.cross_entropy(logits, a))
            else:
                w = w.to(logits.device)
                losses.append((F.cross_entropy(logits, a, reduction="none") * w).sum() / w.sum())
        return torch.stack(losses)

    def nll(self, rows, bs=4):
        """Per-row teacher-forced NLL of the majority answer plus EOS (nan if non-finite)."""
        import torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(rows), bs):
                out.extend(float(v) for v in self._nll_batch(rows[start:start + bs]))
        return out

    def grads(self, names, rows, bs=1):
        """Gradients of the mean answer NLL over ``rows`` w.r.t. the named weights."""
        import torch
        params = [self.param(n) for n in names]
        for p in params:
            p.grad = None
            p.requires_grad_(True)
        try:
            with torch.enable_grad():
                for start in range(0, len(rows), bs):
                    (self._nll_batch(rows[start:start + bs]).sum() / len(rows)).backward()
            return {n: p.grad for n, p in zip(names, params)}
        finally:
            for p in params:
                p.requires_grad_(False)
                p.grad = None

    def generate(self, rows, max_new=None, bs=GEN_BATCH):
        """Greedy decoding; flags rows whose logits become non-finite before EOS."""
        import torch
        max_new = self.max_new_tokens if max_new is None else max_new
        embed = self.lm.get_input_embeddings()
        records = []
        for start in range(0, len(rows), bs):
            batch = rows[start:start + bs]
            try:
                with torch.inference_mode():
                    x, mask = self._pad([self.seq_embeds(r)[0] for r in batch], left=True)
                    position = (mask.cumsum(-1) - 1).clamp_min(0)
                    out = self.lm.model(inputs_embeds=x, attention_mask=mask,
                                        position_ids=position, use_cache=True)
                    past = out.past_key_values
                    logits = self.lm.lm_head(out[0][:, -1]).float()
                    bad = ~torch.isfinite(logits).all(-1)
                    nxt, last = logits.argmax(-1), position[:, -1]
                    tokens = [[] for _ in batch]
                    done = torch.zeros(len(batch), dtype=torch.bool, device=x.device)
                    del out, x
                    for step in range(max_new):
                        for i in range(len(batch)):
                            if not done[i]:
                                tokens[i].append(int(nxt[i]))
                        done |= nxt == self.eos
                        if bool(done.all()) or step == max_new - 1:
                            break
                        mask = torch.cat([mask, mask.new_ones((len(batch), 1))], 1)
                        last = last + 1
                        out = self.lm.model(inputs_embeds=embed(nxt[:, None]), attention_mask=mask,
                                            position_ids=last[:, None], past_key_values=past,
                                            use_cache=True)
                        past = out.past_key_values
                        logits = self.lm.lm_head(out[0][:, -1]).float()
                        bad |= ~torch.isfinite(logits).all(-1) & ~done
                        nxt = logits.argmax(-1)
                for i, row in enumerate(batch):
                    records.append({"sid": row["sid"], "prediction": self.decode(tokens[i]),
                                    "token_ids": tokens[i], "nonfinite": bool(bad[i]),
                                    "finish": "eos" if bool(done[i]) else "length"})
            except RuntimeError as error:   # e.g. a kernel failing on extreme values
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    raise                   # a resource failure is never attack damage
                for row in batch:
                    records.append({"sid": row["sid"], "prediction": "", "token_ids": [],
                                    "nonfinite": True, "finish": "error", "error": str(error)[:200]})
        return records

    def extra_metrics(self, rows, records):
        """Task-specific diagnostics to report beside ``acc`` (none by default)."""
        return {}

    def accuracy(self, rows):
        """The task's panel score with every non-finite or failed generation scored 0."""
        records = self.generate(rows)
        for row, record in zip(rows, records):
            record["score"] = 0.0 if record["nonfinite"] else self.score(record["prediction"],
                                                                         row["answers"])
        n = len(records)
        return {"acc": sum(r["score"] for r in records) / n,
                "nonfinite_rate": sum(r["nonfinite"] for r in records) / n,
                "eos_rate": sum(r["finish"] == "eos" for r in records) / n,
                "length_rate": sum(r["finish"] == "length" for r in records) / n,
                **self.extra_metrics(rows, records), "records": records}

    def prefill_logits(self, rows):
        import torch
        with torch.inference_mode():
            x, mask = self._pad([self.seq_embeds(r)[0] for r in rows], left=True)
            position = (mask.cumsum(-1) - 1).clamp_min(0)
            hidden = self.lm.model(inputs_embeds=x, attention_mask=mask, position_ids=position,
                                   use_cache=False)[0]
            return self.lm.lm_head(hidden[:, -1]).float()
