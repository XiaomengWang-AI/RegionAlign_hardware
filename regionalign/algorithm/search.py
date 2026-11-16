"""Region-first search: pick what the page should hold, then the bits inside it.

Contract: ``search(ctx) -> {"permutation": spec, "attacks": {k: flips}}`` for
k = 1..16, where ``ctx`` is a ``region.RegionContext``, flips are in the
**deployed** model's coordinates and every search-panel loss, gradient and
activation profile is charged against ``ctx``'s budget. Whether the result is
allowed is the evaluator's call (``ctx.check``), never this module's.

A *group* is one way to fill the page: the page as it stands (``noperm`` and
``local``), one original row selected for placement (``local``), or every
logical weight occupying the page after the seeded legal layout (``random``).
The random arm fixes only the layout; bit search still sees the complete fixed
page, exactly as it does for the other arms.

1. **Candidates** -- one gradient pass and one activation profile. Each weight of
   a group gets its best allowed single-bit move under two scores: the
   activation-weighted size ``|dw| * max|x_in|`` (GenBFA's magnitude term) and the
   first-order loss rise ``grad * dw`` (BFA's ranking). An INT8 sign or top bit
   moves a weight by up to its row's maximum, far outside the range where a
   linearisation ranks reliably: on the tuning regions, on every page where a
   flip did measurable damage the most damaging flip came from the size ranking,
   and on the two down_proj pages the size-best flip beat the first-order-best
   in every screened row (62/62, 68/68). Both rankings are kept, size leading
   (``SIZE_LEAD`` size candidates per first-order one).
2. **Groups by successive halving** (``local``): candidate layouts are the
   ``ROWS_SIZE`` best rows by size reach and the ``ROWS_FO`` best by first-order
   reach, where a row's reach is the sum of its top-``K_MAX`` allowed single-bit
   scores -- a free surrogate of multi-bit potential, not just its single best
   flip. The first halving round is seeded by that same free surrogate
   (``Group.surrogate``), so no budget is spent measuring one flip of every
   group; only the kept survivors are ever measured exactly, and the budget then
   concentrates on them round by round (``HALVING``) -- fixed-budget best-arm
   identification (Karnin et al.).
3. **Growth by lazy greedy** inside a group: single flips are measured exactly
   once, and each budget step accepts the candidate with the largest *current*
   marginal rise, re-measuring a candidate only when its stale value would win
   (Minoux). When no screened candidate still helps, the next ones in the ranking
   are screened. After the first accepted flip the activations (and, with
   ``REFRESH_GRAD``, the gradient) are re-measured on the faulted model and the
   new top candidates join, as the progressive bit search re-ranks after a flip.
"""
from __future__ import annotations

import torch

from regionalign.harness import core as P
from regionalign.harness import region as R

BLOCK = 1 << 21                      # weights per block of the candidate pass (move_table peak-memory knob)
ROWS_SIZE = 24                       # local: best rows by top-K_MAX size reach
ROWS_FO = 8                          # ... and by top-K_MAX first-order reach
SIZE_LEAD = 3                        # size candidates per first-order candidate in a ranking
HALVING = ((4, 2, 0.35), (2, 6, 0.5), (1, R.K_MAX, 1.0))   # (survivors, grow to k, share left)
FIRST_SCREEN = 0.4                   # share of a group's allowance screened before growing
EXPAND = 8                           # candidates screened when the screened ones stop helping
REFRESH = 8                          # candidates screened after the refresh
REFRESH_GRAD = True                  # re-measure the gradient as well as the activations


def finite(value):
    return value == value and value not in (float("inf"), float("-inf"))


def row_potential(scores, k=R.K_MAX):
    """Free multi-bit reach surrogate: along the last axis, the sum of the top-``k``
    allowed, attack-positive candidate scores (disallowed or loss-reducing moves
    count as zero). On a 2-D score table this is per row; on a 1-D one it is the
    whole group's reach. Uses only the already-measured score tables, so it costs
    no budget."""
    usable = scores.masked_fill(~torch.isfinite(scores), 0.0).clamp(min=0.0)
    return usable.topk(min(k, usable.shape[-1]), dim=-1).values.sum(dim=-1)


def afford(ctx, kind, limit, n=1):
    return ctx.can_afford(kind, n) and ctx.spent + n * R.COST[kind] <= limit


def move_table(ctx, W, g, act, rows, cols=None):
    """Per weight of ``rows``: (fo_gain, fo_bit, size, size_bit, cols), shaped [rows, cols]."""
    lm, name = ctx.rt.lm, ctx.name
    n_cols = W.shape[1]
    cols = torch.arange(n_cols, device=W.device) if cols is None else cols
    rows = torch.as_tensor(rows, device=W.device)
    bound = P.row_absmax(W)
    ninf = torch.tensor(float("-inf"), device=W.device)
    out = [[], [], [], []]
    step = max(1, BLOCK // len(cols))
    for lo in range(0, len(rows), step):
        r = rows[lo:lo + step]
        index = (r[:, None] * n_cols + cols[None, :]).reshape(-1)
        codes, w = P.stored_codes(lm, name, index), W.view(-1)[index].float()
        grad, a, rb = g.view(-1)[index], act[index % n_cols], bound[index // n_cols]
        bits, scales, dtype = P.decode_constants(lm, name, index)   # fixed per block; out of the bit sweep
        fo, size = torch.full_like(w, float("-inf")), torch.full_like(w, float("-inf"))
        fo_bit = torch.zeros(w.shape, dtype=torch.int8, device=W.device)
        size_bit = torch.zeros_like(fo_bit)
        for b in range(bits):
            new = P.coded_values(lm, name, index, codes ^ (1 << b),
                                 bits=bits, scales=scales, dtype=dtype).float()
            ok = P.value_allowed(new, rb)
            delta = new - w
            gain = torch.where(ok, grad * delta, ninf)
            moved = torch.where(ok, delta.abs() * a, ninf)
            better = gain > fo
            fo, fo_bit = torch.where(better, gain, fo), torch.where(better, b, fo_bit)
            better = moved > size
            size, size_bit = torch.where(better, moved, size), torch.where(better, b, size_bit)
        for store, value in zip(out, (fo, fo_bit, size, size_bit)):
            store.append(value.view(len(r), len(cols)))
    return (*[torch.cat(v) for v in out], cols)


def ranking(elements, fo, fo_bit, size, size_bit, limit=4096):
    """Both rankings merged, size leading, one bit per weight, disallowed moves dropped."""
    n = min(limit, fo.numel())
    queues = ((size.topk(n).indices.tolist(), size, size_bit),
              (fo.topk(n).indices.tolist(), fo, fo_bit))
    position, picks, seen, turn = [0, 0], [], set(), 0
    while position[0] < n or position[1] < n:
        which = 0 if turn % (SIZE_LEAD + 1) < SIZE_LEAD else 1
        turn += 1
        if position[which] >= n:
            which = 1 - which
        order, score, bits = queues[which]
        k = order[position[which]]
        position[which] += 1
        if float(score[k]) == float("-inf"):
            continue
        e = int(elements[k])
        if e not in seen:
            seen.add(e)
            picks.append((e, int(bits[k])))
    return picks


class Group:
    """One way to fill the page, its candidate moves and the attack grown inside it."""

    def __init__(self, row, cols, table, base, fixed_spec=None, ranking_limit=4096):
        self.row, self.cols, self.base, self.fixed_spec = row, cols, base, fixed_spec
        self.queue = ranking(*table, limit=ranking_limit)
        self.surrogate = float(row_potential(table[3]))   # free top-K_MAX size reach
        self.cursor = 0                    # next unscreened position of ``queue``
        self.marginal = {}                 # (element, bit) -> (rise, len(chosen) when measured)
        self.chosen, self.rise, self.attacks = [], 0.0, []
        self.refreshed = False

    def auc(self):
        rises = [r for _, r in self.attacks] + [self.rise] * (R.K_MAX - len(self.attacks))
        return sum(R.envelope(rises[:R.K_MAX]))


def measure(ctx, group, flip, limit):
    """Exact marginal rise of ``flip`` on top of the group's current set."""
    if not afford(ctx, "eval", limit):
        return False
    value = ctx.nll(group.chosen + [[ctx.name, *flip]])
    rise = value - group.base - group.rise if finite(value) else float("-inf")
    group.marginal[flip] = (rise, len(group.chosen))
    return True


def screen_next(ctx, group, n, limit):
    """Measure the next ``n`` unscreened candidates of the ranking; False if none was."""
    done, used = 0, {e for _, e, _ in group.chosen}
    while done < n and group.cursor < len(group.queue):
        flip = group.queue[group.cursor]
        if flip[0] in used or flip in group.marginal:
            group.cursor += 1
            continue
        if not measure(ctx, group, flip, limit):
            break
        group.cursor += 1
        done += 1
    return done > 0


def grow(ctx, W, state, group, until, limit):
    """Lazy greedy to ``until`` flips while ``ctx.spent`` stays within ``limit``."""
    while len(group.attacks) < until:
        used = {e for _, e, _ in group.chosen}
        accepted = False
        while True:
            live = [(rise, flip) for flip, (rise, _) in group.marginal.items()
                    if flip[0] not in used and finite(rise) and rise > 0]
            if live:
                rise, flip = max(live)
                if group.marginal[flip][1] == len(group.chosen):
                    group.chosen = group.chosen + [[ctx.name, *flip]]
                    group.rise += rise
                    accepted = True
                    break
                if measure(ctx, group, flip, limit):
                    continue
                break
            if not screen_next(ctx, group, EXPAND, limit):
                break
        group.attacks.append((list(group.chosen), group.rise))
        if not accepted:
            break
        if not group.refreshed:
            refresh(ctx, W, state, group, limit)
    while len(group.attacks) < until:                  # nothing more helps (or no budget left)
        group.attacks.append((list(group.chosen), group.rise))


def refresh(ctx, W, state, group, limit):
    """Re-rank on the faulted model, as progressive bit search does after a flip."""
    group.refreshed = True
    cost = R.COST["eval"] * (1 + REFRESH) + (R.COST["grad"] if REFRESH_GRAD else 0)
    if ctx.spent + cost > limit or ctx.remaining() < cost:
        return
    g = ctx.grad(group.chosen) if REFRESH_GRAD else state["g"]
    act = ctx.profile(group.chosen)
    if group.fixed_spec is not None:
        table = layout_page_table(ctx, W, g, act, group.fixed_spec)
    elif group.row is None:
        table = page_table(ctx, W, g, act)
    else:
        fo, fo_bit, size, size_bit, cols = move_table(ctx, W, g, act, [group.row], group.cols)
        table = (group.row * W.shape[1] + cols, fo[0], fo_bit[0], size[0], size_bit[0])
    used = {e for _, e, _ in group.chosen}
    fresh = [f for f in ranking(*table, limit=REFRESH * 4)
             if f[0] not in used and f not in group.marginal][:REFRESH]
    for flip in fresh:
        if not measure(ctx, group, flip, limit):
            break


def page_table(ctx, W, g, act):
    parts = [[], [], [], [], []]
    for row, first, stop in ctx.segments:
        cols = torch.arange(first, stop, device=W.device)
        fo, fo_bit, size, size_bit, _ = move_table(ctx, W, g, act, [row], cols)
        for store, value in zip(parts, (row * W.shape[1] + cols, fo[0], fo_bit[0], size[0], size_bit[0])):
            store.append(value)
    return tuple(torch.cat(p) for p in parts)


def layout_page_table(ctx, W, g, act, spec):
    """Candidates whose images occupy the fixed page under ``spec``.

    ``axis_orders`` uses deployed-position -> original-position orders.  Taking
    the preimage of every deployed page segment therefore gives the complete
    set of logical weights that a fixed layout exposes, including rows moved as
    side effects of the seeded placement and any input-axis order.  Scoring and
    exact evaluation remain in original coordinates; ``ctx.deploy`` maps the
    chosen flips back to their deployed coordinates at the boundary.
    """
    orders = P.axis_orders(spec, ctx.d)
    row_order = orders.get((ctx.name, 0))
    col_order = orders.get((ctx.name, 1))
    parts = [[], [], [], [], []]
    for deployed_row, first, stop in ctx.segments:
        original_row = row_order[deployed_row] if row_order is not None else deployed_row
        deployed_cols = range(first, stop)
        original_cols = ([col_order[col] for col in deployed_cols]
                         if col_order is not None else list(deployed_cols))
        cols = torch.as_tensor(original_cols, device=W.device)
        fo, fo_bit, size, size_bit, _ = move_table(ctx, W, g, act, [original_row], cols)
        values = (original_row * W.shape[1] + cols, fo[0], fo_bit[0], size[0], size_bit[0])
        for store, value in zip(parts, values):
            store.append(value)
    return tuple(torch.cat(part) for part in parts)


def groups_of(ctx, W, state, base):
    """The current page, a fixed random-layout page, or the best placeable rows."""
    g, act = state["g"], state["act"]
    if ctx.arm == "random":
        spec = ctx.placement(ctx.forced_row)
        table = layout_page_table(ctx, W, g, act, spec)
        return [Group(ctx.forced_row, None, table, base, fixed_spec=spec,
                      ranking_limit=table[0].numel())]
    groups = [Group(None, None, page_table(ctx, W, g, act), base)]
    rows = ctx.candidate_rows()
    if rows:
        _, first, stop = ctx.primary
        movable = ctx.arm == "local" and R.columns_movable(ctx.region)
        cols = None if movable else torch.arange(first, stop, device=W.device)
        fo, fo_bit, size, size_bit, cols = move_table(ctx, W, g, act, rows, cols)
        pick = list(dict.fromkeys(
            row_potential(size).topk(min(ROWS_SIZE, len(rows))).indices.tolist()
            + row_potential(fo).topk(min(ROWS_FO, len(rows))).indices.tolist()))
        for i in pick:
            table = (rows[i] * W.shape[1] + cols, fo[i], fo_bit[i], size[i], size_bit[i])
            groups.append(Group(rows[i], None if movable else cols, table, base))
    return groups


def search(ctx):
    W = ctx.weight()
    base = ctx.nll()
    state = {"g": ctx.grad(), "act": ctx.profile()}
    groups = groups_of(ctx, W, state, base)
    if REFRESH_GRAD:
        state["g"] = None            # unused once the candidate tables exist; refresh
                                     # re-measures the gradient, so free the region tensor
    if ctx.arm == "local":
        survivors = groups                           # seeded by the free reach surrogate; only
        for count, until, share in HALVING:          # the kept survivors are ever measured exactly
            survivors = sorted(survivors, key=lambda gr: gr.auc() if gr.attacks else gr.surrogate,
                               reverse=True)[:count]
            allowance = int(ctx.remaining() * share) // len(survivors)
            for group in survivors:
                limit = ctx.spent + allowance
                if not group.attacks:
                    screen_next(ctx, group, int(allowance * FIRST_SCREEN), limit)
                grow(ctx, W, state, group, until, limit)
        best = max(survivors, key=Group.auc)
    else:
        best = groups[0]
        screen_next(ctx, best, int(ctx.remaining() * FIRST_SCREEN), ctx.budget)
        grow(ctx, W, state, best, R.K_MAX, ctx.budget)
    return finish(ctx, W, best)


def finish(ctx, W, group):
    attacks = [flips for flips, _ in group.attacks]
    rises = [rise for _, rise in group.attacks]
    if group.fixed_spec is not None:
        spec = group.fixed_spec
    elif group.row is None:
        spec = P.validate_spec({}, ctx.d)
    else:
        spec = ctx.placement(group.row, [e % W.shape[1] for _, e, _ in attacks[-1]])
    deployed = {k: ctx.deploy(spec, flips) for k, flips in zip(R.K_SCHEDULE, attacks)}
    verdicts = ctx.check(spec, {k: f for k, f in deployed.items() if f})
    bad = {k: v["reasons"] for k, v in verdicts.items() if not v["valid"]}
    if bad:
        raise RuntimeError(f"placement failed the evaluator check: {bad}")
    return {"permutation": P.spec_to_json(spec),
            "attacks": {str(k): f for k, f in deployed.items()},
            "source_row": group.row,
            "search_nll": {"base": group.base, "per_k": [group.base + r for r in rises]},
            "spent": ctx.spent, "calls": dict(ctx.calls)}
