"""Memory-bounded parameter staging for the existing RegionAlign checksum.

Only the integer parameter contribution is partitioned. Its words are the
original logical row-major parameter bytes reinterpreted as signed INT16,
then converted to INT64. Position weights retain the global word offset and
the original modulo-65521 rule. Partial reductions are added in the INT64
ring, reproducing torch.sum's signed overflow before the outer modulo-2**61.

Scale tensors deliberately retain the original float64 expression and single
torch.sum. Reassociating their floating reduction could change rounding and
the subsequent int(value * 1e6); those smaller allocations are not chunked.
Parameter numeric staging is limited to DEFAULT_CHUNK_ELEMENTS words. A
noncontiguous parameter is gathered in bounded logical-order blocks rather
than making a contiguous copy of the entire parameter. No model state,
quantization, evidence coverage, or checksum arithmetic is changed.

Call install() only inside a separately identified execution policy. It patches
regionalign.harness.region.model_checksum for that context and restores the
previous callable on exit, including exceptional and nested exits.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

DEFAULT_CHUNK_ELEMENTS = 1048576
POSITION_MODULUS = 65521
STATE_MODULUS = 1 << 61


def _signed_int64(value):
    """Interpret an integer modulo 2**64 as torch's signed INT64 result."""
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)


def _word_blocks(value, chunk_elements):
    import torch
    if value.is_contiguous():
        words = value.reshape(-1).view(torch.int16)
        for start in range(0, words.numel(), chunk_elements):
            yield start, words[start:start + chunk_elements]
        return
    # reshape(-1) can copy a whole transposed embedding. Advanced indexing
    # instead copies only this block, preserving logical C-order bytes.
    count = value.numel()
    item_bytes = value.element_size()
    if count * item_bytes % 2:
        raise RuntimeError("parameter byte count cannot be reinterpreted as INT16")
    values_per_block = max(1, chunk_elements * 2 // item_bytes)
    if item_bytes == 1:
        values_per_block -= values_per_block % 2
    offset = 0
    for start in range(0, count, values_per_block):
        flat_indices = torch.arange(start, min(start + values_per_block, count),
                                    device=value.device, dtype=torch.int64)
        coordinates = []
        for size in reversed(value.shape):
            coordinates.append(flat_indices % size)
            flat_indices = flat_indices // size
        block = value[tuple(reversed(coordinates))].reshape(-1).view(torch.int16)
        yield offset, block
        offset += block.numel()


def _parameter_sum(value, chunk_elements):
    import torch
    total = 0
    for start, words in _word_blocks(value, chunk_elements):
        flat = words.to(torch.int64)
        weights = (torch.arange(start, start + flat.numel(), device=flat.device,
                                dtype=torch.int64) % POSITION_MODULUS + 1)
        # Every reduction has exactly the original integer dtype. Addition in
        # Z/(2**64) is associative, including when an individual chunk wraps.
        total = _signed_int64(total + int((flat * weights).sum()))
    return total


def model_checksum(lm, chunk_elements=DEFAULT_CHUNK_ELEMENTS):
    """Return the existing integer checksum with bounded parameter temporaries."""
    import torch
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    total = 0
    for name, param in sorted(lm.named_parameters()):
        total = (total * 1000003 + _parameter_sum(param.data, chunk_elements)) % STATE_MODULUS
    # Keep this path identical to region.model_checksum, including one float64
    # reduction and Python float/int conversions. Do not sum scale chunks.
    for name, scales in sorted(getattr(lm, "ar_quant", {}).get("scales", {}).items()):
        flat = scales.reshape(-1).double()
        weights = torch.arange(flat.numel(), device=flat.device, dtype=torch.float64) % 65521 + 1
        total = (total * 1000003 + int(float((flat * weights).sum()) * 1e6)) % STATE_MODULUS
    return total


@contextmanager
def install():
    """Temporarily replace only the shared region module's checksum callable."""
    from regionalign.harness import region
    with patch.object(region, "model_checksum", model_checksum):
        yield
