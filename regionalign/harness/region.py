"""Shared RegionAlign geometry, legal placement, evaluator rules and search context.

Regions are predefined aligned pages of decoder Linear code storage. The kernel
supports the existing INT8/INT4 geometry, legal MHA/GQA/MQA permutations and
per-arm forward-equivalent accounting. Task scoring, model loading, panel
selection and experiment-specific artifact audits belong to their drivers.

The attacker is given the region first and nothing else: any stored bit on the
page may flip, in either direction.  There is no map of flippable cells.
"""
from __future__ import annotations

import hashlib
import math
import random
from contextlib import nullcontext

from . import core as P

LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
PAGE_BYTES = P.PAGE_BYTES
K_SCHEDULE = tuple(range(1, 17))
K_MAX = max(K_SCHEDULE)
ARMS = ("noperm", "local", "random")
AUDIT_KEYS = ("format", "bits", "scope", "schema_version", "total_tensors", "total_parameters",
              "quantized_tensors", "quantized_parameters", "quantized_fraction",
              "unquantized_tensors", "int4_group_columns", "gain_scale")
COST = {"eval": 1, "grad": 3}
BUDGET = 300
GATE_NLL = 0.01
KINDS = tuple(name.split(".")[-1] for name in LINEARS)


class BudgetExceeded(RuntimeError):
    """A search asked for more compute than ``BUDGET``."""


def tensor_name(kind, layer):
    prefix = "self_attn" if kind.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
    return f"model.layers.{layer}.{prefix}.{kind}.weight"


def tensor_shape(kind, d):
    """(rows, cols) of a decoder Linear of ``kind``."""
    if kind in ("gate_proj", "up_proj"):
        return d["mlp"], d["hidden"]
    if kind == "down_proj":
        return d["hidden"], d["mlp"]
    if kind in ("k_proj", "v_proj"):
        kv_heads = d.get("kv_heads", d["heads"])
        return kv_heads * d["head_dim"], d["hidden"]
    return d["hidden"], d["hidden"]


def elements_per_page(bits=8, page_bytes=PAGE_BYTES):
    """Logical weight codes in a physical page of ``page_bytes`` bytes."""
    if bits not in P.FORMAT_BITS.values() or isinstance(page_bytes, bool) or page_bytes < 1:
        raise ValueError("page geometry needs a supported code width and positive byte size")
    return page_bytes * 8 // bits


def page_count(kind, d, bits=8, page_bytes=PAGE_BYTES):
    rows, cols = tensor_shape(kind, d)
    return -(-rows * cols // elements_per_page(bits, page_bytes))


def make_region(kind, layer, page, d, bits=8, page_bytes=PAGE_BYTES):
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a decoder Linear; gains, embed and lm_head are out")
    if not 0 <= layer < d["layers"] or not 0 <= page < page_count(kind, d, bits, page_bytes):
        raise ValueError(f"no page {page} in layer {layer} {kind}")
    return {"id": f"L{layer:02d}.{kind}.p{page}", "kind": kind, "layer": layer, "page": page,
            "tensor": tensor_name(kind, layer), "page_bytes": page_bytes, "bits": bits}


def page_segments(region, d):
    """[(row, first column, end column)] of the elements the page holds, in storage order."""
    rows, cols = tensor_shape(region["kind"], d)
    per_page = elements_per_page(region["bits"], region["page_bytes"])
    start = region["page"] * per_page
    end = min(start + per_page, rows * cols)
    segments, element = [], start
    while element < end:
        row, col = divmod(element, cols)
        stop = min(cols, col + end - element)
        segments.append((row, col, stop))
        element += stop - col
    return segments


def primary_segment(region, d):
    """The page's widest segment: where a placed row lands."""
    segments = page_segments(region, d)
    return max(segments, key=lambda s: (s[2] - s[1], -segments.index(s)))


def columns_movable(region):
    """Only down_proj rows are wider than a page, so only there do columns need placing."""
    return region["kind"] == "down_proj"


def _transposed(n, a, b):
    order = list(range(n))           # new -> old
    order[a], order[b] = b, a
    return order


def _grouped_transposed(n, target, source, group=None):
    """Map ``source`` to ``target`` without splitting an INT4 scale group.

    A column order may freely permute values inside one scale group and may move
    complete equal-sized groups.  With no group this deliberately reduces to
    the historical two-element transposition used by INT8.
    """
    if not 0 <= target < n or not 0 <= source < n:
        raise ValueError("permutation coordinate is outside its axis")
    if not group:
        return _transposed(n, target, source)
    target_start, source_start = target // group * group, source // group * group
    target_stop, source_stop = min(n, target_start + group), min(n, source_start + group)
    if target_start == source_start:
        return _transposed(n, target, source)
    if target_stop - target_start != source_stop - source_start:
        raise ValueError("a full INT4 scale group cannot move into a shorter final group")
    order = list(range(n))
    target_group = order[target_start:target_stop]
    source_group = order[source_start:source_stop]
    order[target_start:target_stop] = source_group
    order[source_start:source_stop] = target_group
    source_at = target_start + source - source_start
    order[target], order[source_at] = order[source_at], order[target]
    return order


def _kv_group_head_order(d, target_group, source_group):
    """Query-head order that maps one complete old KV group to a new KV group."""
    qg = d.get("query_group_size", 1)
    kv_heads = d.get("kv_heads", d["heads"])
    if not 0 <= target_group < kv_heads or not 0 <= source_group < kv_heads:
        raise ValueError("KV group is outside the attention layout")
    order = list(range(d["heads"]))
    if target_group != source_group:
        a, b = target_group * qg, source_group * qg
        order[a:a + qg], order[b:b + qg] = order[b:b + qg], order[a:a + qg]
    return order


def _attention_spec(kind, layer, target_row, source_row, d):
    """Unvalidated attention spec mapping one source row to one target row."""
    hd = d["head_dim"]
    qg = d.get("query_group_size", 1)
    target_head, target_index = divmod(target_row, hd)
    source_head, source_index = divmod(source_row, hd)
    if kind in ("q_proj", "k_proj") and target_index != source_index:
        raise ValueError("RoPE permits q/k rows to move only at a fixed intra-head coordinate")
    if kind == "q_proj":
        target_group, source_group = target_head // qg, source_head // qg
        heads = _kv_group_head_order(d, target_group, source_group)
        a = target_group * qg + target_head % qg
        b = target_group * qg + source_head % qg
        heads[a], heads[b] = heads[b], heads[a]
        return {"heads": {str(layer): heads}}
    if kind == "k_proj":
        return {"heads": {str(layer): _kv_group_head_order(d, target_head, source_head)}}

    # A V row is a KV group plus one value coordinate.  The group move is
    # represented by `heads`; the coordinate move is repeated for every query
    # head sharing the old source KV group, as required by GQA/MQA.
    heads = _kv_group_head_order(d, target_head, source_head)
    values = list(range(d["hidden"]))
    for query_head in range(source_head * qg, (source_head + 1) * qg):
        a = query_head * hd + target_index
        b = query_head * hd + source_index
        values[a], values[b] = values[b], values[a]
    return {"heads": {str(layer): heads}, "vo": {str(layer): values}}


def _raw_reachable_rows(region, d):
    row = primary_segment(region, d)[0]
    kind, hd = region["kind"], d["head_dim"]
    if kind == "q_proj":
        return [head * hd + row % hd for head in range(d["heads"])]
    if kind == "k_proj":
        return [head * hd + row % hd for head in range(d.get("kv_heads", d["heads"]))]
    return list(range(tensor_shape(kind, d)[0]))


def reachable_rows(region, d):
    """Original rows that a legal stored-code permutation can move into the page."""
    row, kind, group = primary_segment(region, d)[0], region["kind"], d.get("group")
    candidates = _raw_reachable_rows(region, d)
    if not group:
        return candidates
    if kind in ("gate_proj", "up_proj", "o_proj", "down_proj"):
        size = d["mlp"] if kind in ("gate_proj", "up_proj") else d["hidden"]
        target_size = min(group, size - row // group * group)
        return [source for source in candidates
                if min(group, size - source // group * group) == target_size]
    reachable = []
    for source in candidates:
        try:
            P.validate_spec(_attention_spec(kind, region["layer"], row, source, d), d)
        except ValueError:
            continue
        reachable.append(source)
    return reachable


def _column_order(size, first, stop, wanted, group=None):
    """Place wanted old columns in a window, moving full INT4 groups when needed."""
    if any(isinstance(col, bool) or not isinstance(col, int) or not 0 <= col < size
           for col in wanted):
        raise ValueError("a requested column is outside the tensor")
    if not group:
        order, where = list(range(size)), list(range(size))
        free = [position for position in range(first, stop) if order[position] not in wanted]
        for col in wanted:
            if first <= where[col] < stop:
                continue
            if not free:
                raise ValueError("more columns than the window holds")
            position, old_position = free.pop(0), where[col]
            a, b = order[position], order[old_position]
            order[position], order[old_position] = b, a
            where[a], where[b] = old_position, position
        return order
    if first % group or stop % group:
        raise ValueError("the page window cuts an INT4 scale group")
    starts = list(range(0, size, group))
    lengths = [min(group, size - start) for start in starts]
    wanted_groups = list(dict.fromkeys(col // group for col in wanted))
    group_order = list(range(len(starts)))            # destination group -> old group
    where = list(range(len(starts)))                  # old group -> destination group
    destination = [g for g, start in enumerate(starts)
                   if first <= start and start + lengths[g] <= stop]
    for old_group in wanted_groups:
        if where[old_group] in destination:
            continue
        free = [g for g in destination if group_order[g] not in wanted_groups
                and lengths[g] == lengths[old_group]]
        if not free:
            raise ValueError("the page has too few compatible INT4 scale groups")
        target_group, old_position = free[0], where[old_group]
        displaced = group_order[target_group]
        group_order[target_group], group_order[old_position] = old_group, displaced
        where[old_group], where[displaced] = target_group, old_position
    order = []
    for old_group in group_order:
        order.extend(range(starts[old_group], starts[old_group] + lengths[old_group]))
    return order


def placement(region, d, source_row, columns=()):
    """Normalized spec moving original ``source_row`` into the primary segment.

    ``columns`` are original columns of that row that must land inside the
    segment's window (down_proj only; elsewhere the segment is the whole row).
    The result is checked by ``harness.validate_spec``.
    """
    if source_row not in reachable_rows(region, d):
        raise ValueError(f"row {source_row} cannot reach {region['id']}")
    kind, layer = region["kind"], str(region["layer"])
    row, first, stop = primary_segment(region, d)
    spec = {}
    if source_row != row:
        if kind in ("gate_proj", "up_proj"):
            spec["mlp"] = {layer: _grouped_transposed(d["mlp"], row, source_row,
                                                       d.get("group"))}
        elif kind in ("o_proj", "down_proj"):
            spec["residual"] = _grouped_transposed(d["hidden"], row, source_row,
                                                    d.get("group"))
        else:
            spec = _attention_spec(kind, region["layer"], row, source_row, d)
    wanted = list(dict.fromkeys(columns))
    if wanted and not columns_movable(region):
        if any(not first <= c < stop for c in wanted):
            raise ValueError(f"columns of {kind} cannot move into a partial window")
    elif wanted:
        order = _column_order(d["mlp"], first, stop, wanted, d.get("group"))
        spec.setdefault("mlp", {})[layer] = order
    return P.validate_spec(spec, d)


def deploy(region, d, spec, flips):
    """Original-coordinate flips of the region tensor -> coordinates in the deployed model."""
    rows, cols = tensor_shape(region["kind"], d)
    orders = P.axis_orders(spec, d)

    def inverse(order, n):
        if order is None:
            return range(n)
        out = [0] * n
        for new, old in enumerate(order):
            out[old] = new
        return out
    row_of = inverse(orders.get((region["tensor"], 0)), rows)
    col_of = inverse(orders.get((region["tensor"], 1)), cols)
    out = []
    for name, element, bit in flips:
        if name != region["tensor"]:
            raise ValueError(f"{name} is not the region tensor")
        i, j = divmod(element, cols)
        out.append([name, row_of[i] * cols + col_of[j], bit])
    return out


def random_row(region, d, seed):
    """The `random` arm's row: drawn from the seed, never chosen by the attacker."""
    digest = hashlib.sha256(f"{seed}|{region['id']}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16)).choice(reachable_rows(region, d))


def check_region_attack(rt, region, flips, k):
    """Validate one attack on the deployed (rearranged, unfaulted) model.

    ``core.check_attack`` supplies the generic rules -- at most ``k`` distinct
    bits, one tensor, a span of at most 4096 bytes and the in-range value rule
    -- and this adds the region's own: every flip in the region tensor, on its page.
    """
    info = P.check_attack(rt, flips, k)
    try:
        actual_bits = P.code_bits(rt.lm, region["tensor"])
    except (AttributeError, KeyError, ValueError) as error:
        info["reasons"].append(str(error))
        info["valid"] = False
        return info
    if region.get("bits") != actual_bits:
        info["reasons"].append(
            f"region geometry declares {region.get('bits')}-bit codes but the model stores {actual_bits}-bit codes")
    try:
        flips = P.normalize_flips(flips)
    except ValueError:
        return info
    for name, element, bit in flips:
        if name != region["tensor"]:
            info["reasons"].append(f"{name} is outside the region tensor {region['tensor']}")
        elif P.flip_byte(element, bit, P.code_bits(rt.lm, name)) // region["page_bytes"] != region["page"]:
            info["reasons"].append(f"{name}[{element}] bit {bit} is outside page {region['page']}")
    info["valid"] = not info["reasons"]
    return info


def arm_reasons(region, arm, spec, d, seed):
    """Why ``spec`` is not a layout this arm may deploy (empty when it is)."""
    if arm not in ARMS:
        return [f"unknown arm {arm!r}"]
    if arm == "noperm" and not P.is_identity(spec):
        return ["noperm keeps the original layout"]
    if arm == "random":
        expected = placement(region, d, random_row(region, d, seed))
        if P.spec_to_json(expected) != P.spec_to_json(spec):
            return ["random deploys the seeded row's placement and nothing else"]
    return []


def artifact_metadata(rt):
    """What every artifact declares about the model it was searched on."""
    return {"format": P.weight_format(rt.lm), "task": getattr(rt, "task", None),
            "quantization": P.quantization_audit(rt.lm), "quantization_schema_version": 2}


def budget_reasons(artifact, budget=BUDGET):
    """Why the artifact's recorded compute is not a search within ``budget``.

    ``RegionContext`` enforces the budget only inside the search's own process, where
    the search could raise ``ctx.budget`` or reset ``ctx.spent``; the per-kind call
    counts still record every charge, so the evaluator recomputes the spend from them.
    """
    calls, spent = artifact.get("calls"), artifact.get("spent")
    counts_ok = (isinstance(calls, dict) and set(calls) == set(COST)
                 and all(type(n) is int and n >= 0 for n in calls.values()))
    if not counts_ok:
        return ["artifact carries no valid call accounting"]
    used = sum(COST[kind] * n for kind, n in calls.items())
    reasons = []
    if type(spent) is not int or spent != used:
        reasons.append(f"recorded spend {spent!r} differs from its calls ({used})")
    if used > budget:
        reasons.append(f"search used {used} forward-equivalents, over the budget of {budget}")
    return reasons


def envelope(values):
    """Monotone envelope: the best value of any budget at or below each k."""
    best, out = 0.0, []
    for value in values:
        best = max(best, value)
        out.append(best)
    return out


class RegionContext:
    """Everything ``region_search.search`` may use for one (region, arm).

    Exact losses and gradients are measured on the clean model in **original**
    coordinates: flipping a weight and then permuting gives the same function as
    permuting and then flipping its image, so the search needs no deployed model.
    The final layout and flips are validated in deployed coordinates by
    ``check``, which is the evaluator's own rule.

    Two reuses make these measurements cheaper without changing a value
    (``reuse=False`` turns both off; ``self.reuse`` reports what applied).  At
    batch size one, repeated rows of the bootstrap panel are evaluated once
    (``core.mean_nll``).  And because every flip stays inside the region tensor,
    no layer before the region's layer ever changes, so one-row passes start at
    that layer from states recorded on the clean model (``prefix_states``,
    checked bit for bit before use).  Budget charges are unchanged, and ``check``
    always runs full passes.
    """

    def __init__(self, rt, region, arm, seed, budget=BUDGET, log=print, bs=8, grad_bs=4,
                 reuse=True):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.rt, self.region, self.arm, self.seed, self.log = rt, region, arm, seed, log
        self.d = P.dims(rt.lm)
        actual_bits = P.code_bits(rt.lm, region["tensor"])
        if region.get("bits") != actual_bits:
            raise ValueError(
                f"region geometry declares {region.get('bits')}-bit codes but the model stores {actual_bits}-bit codes")
        self.rows = rt.rows["search"]
        self.name = region["tensor"]
        self.shape = tensor_shape(region["kind"], self.d)
        self.segments = page_segments(region, self.d)
        self.primary = primary_segment(region, self.d)
        self.budget, self.spent, self.bs, self.grad_bs = budget, 0, bs, grad_bs
        self.forced_row = random_row(region, self.d, seed) if arm == "random" else None
        self.calls = {"eval": 0, "grad": 0}
        self._reuse_allowed, self.reuse = reuse, None
        self._distinct = self._prefix = None

    # -- exact reuse, prepared once before the first measurement
    def _prepare_reuse(self):
        if self.reuse is not None:
            return
        rows, n = self.rows, len(self.rows)
        groups = P.distinct_rows(rows) if self._reuse_allowed else None
        first, expand = groups or (list(range(n)), list(range(n)))
        if groups is not None and self.bs == 1 and len(first) < n:
            self._distinct = groups
        layer, status = self.region.get("layer"), "off"
        if (self._reuse_allowed and 1 in (self.bs, self.grad_bs)
                and hasattr(self.rt, "prefix_states") and isinstance(layer, int)
                and self.name.startswith(f"model.layers.{layer}.")):
            prefix, status = self.rt.prefix_states([rows[i] for i in first], layer)
            if prefix is not None:
                for index, position in enumerate(expand):
                    prefix["states"][id(rows[index])] = prefix["states"][id(rows[first[position]])]
                self._prefix = prefix
            elif status == "mismatch":
                self.log(f"{self.region.get('id')}: reused layer-{layer} states differ from "
                         "full passes; this search runs full passes")
        self.reuse = {"rows": n, "distinct_rows": len(first),
                      "evaluated_rows": len(first) if self._distinct else n,
                      "prefix_layer": layer if self._prefix is not None else None,
                      "prefix": status}

    def _reusing(self, bs):
        return self.rt.reusing(self._prefix) if self._prefix is not None and bs == 1 else nullcontext()

    # -- compute, charged against the budget
    def remaining(self):
        return self.budget - self.spent

    def can_afford(self, kind, n=1):
        return self.spent + n * COST[kind] <= self.budget

    def _charge(self, kind):
        if not self.can_afford(kind):
            raise BudgetExceeded(f"{kind} would exceed the budget of {self.budget}")
        self.spent += COST[kind]
        self.calls[kind] += 1

    def _flips(self, flips):
        """Checked before anything touches the model, so a bad flip changes nothing."""
        flips = P.normalize_flips(flips)
        if any(name != self.name for name, _, _ in flips):
            raise ValueError("a search may only flip bits of the region tensor")
        size = self.shape[0] * self.shape[1]
        if any(element >= size for _, element, _ in flips):
            raise ValueError(f"a flip lies outside the region tensor's {size} elements")
        bits = P.code_bits(self.rt.lm, self.name)       # a wider bit would not XOR back
        if any(bit >= bits for _, _, bit in flips):
            raise ValueError(f"a flip names a bit outside the {bits}-bit code")
        return flips

    def nll(self, flips=()):
        """Mean search-panel NLL with ``flips`` (original coordinates) applied."""
        flips = self._flips(flips)
        self._prepare_reuse()
        self._charge("eval")
        P.apply_flips(self.rt.lm, flips)
        try:
            with self._reusing(self.bs):
                return P.mean_nll(self.rt, self.rows, bs=self.bs, distinct=self._distinct)
        finally:
            P.apply_flips(self.rt.lm, flips)

    def grad(self, flips=()):
        """Gradient of the mean search-panel NLL w.r.t. the region tensor (float32)."""
        flips = self._flips(flips)
        self._prepare_reuse()
        self._charge("grad")
        P.apply_flips(self.rt.lm, flips)
        try:
            with self._reusing(self.grad_bs):
                return self.rt.grads([self.name], self.rows, bs=self.grad_bs)[self.name].float()
        finally:
            P.apply_flips(self.rt.lm, flips)

    # -- free: geometry and the rules
    def profile(self, flips=()):
        """Max |input| per input column of the region tensor's Linear on the search panel,
        with ``flips`` (original coordinates) applied -- the faulted model's activations."""
        import torch
        flips = self._flips(flips)
        self._prepare_reuse()
        self._charge("eval")
        module = self.rt.lm.get_submodule(self.name[:-len(".weight")])
        stats = []

        def hook(_, args):
            stats.append(args[0].detach().abs().float().amax(dim=(0, 1)))
        # A maximum is unchanged by repeated rows, so each distinct row is passed once.
        rows = self.rows if self._distinct is None else [self.rows[i] for i in self._distinct[0]]
        P.apply_flips(self.rt.lm, flips)
        try:
            handle = module.register_forward_pre_hook(hook)
            try:
                with self._reusing(self.bs):
                    self.rt.nll(rows, bs=self.bs)
            finally:
                handle.remove()
        finally:
            P.apply_flips(self.rt.lm, flips)
        return torch.stack(stats).amax(0)

    def weight(self):
        return self.rt.param(self.name).data

    def candidate_rows(self):
        """Original rows this arm may bring into the page (``[]``: the page stays as it is)."""
        if self.arm == "noperm":
            return []
        if self.arm == "random":
            return [self.forced_row]
        return reachable_rows(self.region, self.d)

    def placement(self, source_row, columns=()):
        if self.arm == "noperm":
            raise ValueError("noperm keeps the original layout")
        if self.arm == "random" and source_row != self.forced_row:
            raise ValueError("random deploys the seeded row only")
        return placement(self.region, self.d, source_row, columns if self.arm == "local" else ())

    def deploy(self, spec, flips):
        return deploy(self.region, self.d, spec, flips)

    def check(self, spec, attacks):
        """The evaluator's verdict on deployed ``attacks`` ({k: flips}) under layout ``spec``."""
        spec = P.validate_spec(spec, self.d)
        layout = arm_reasons(self.region, self.arm, spec, self.d, self.seed)
        if not P.is_identity(spec):
            P.apply_permutation(self.rt, spec)
        try:
            verdicts = {k: check_region_attack(self.rt, self.region, flips, k)
                        for k, flips in attacks.items()}
        finally:
            if not P.is_identity(spec):
                P.apply_permutation(self.rt, spec, inverse=True)
        for info in verdicts.values():
            info["reasons"].extend(layout)
            info["valid"] = not info["reasons"]
        return verdicts


def model_checksum(lm):
    """Position-sensitive fingerprint of every stored code and scale (restore check)."""
    import torch
    total = 0
    for name, param in sorted(lm.named_parameters()):
        flat = param.data.reshape(-1).view(torch.int16).to(torch.int64)
        weights = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64) % 65521 + 1
        total = (total * 1000003 + int((flat * weights).sum())) % (1 << 61)
    for name, scales in sorted(getattr(lm, "ar_quant", {}).get("scales", {}).items()):
        flat = scales.reshape(-1).double()
        weights = torch.arange(flat.numel(), device=flat.device, dtype=torch.float64) % 65521 + 1
        total = (total * 1000003 + int(float((flat * weights).sum()) * 1e6)) % (1 << 61)
    return total
