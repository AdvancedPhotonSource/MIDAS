"""Per-seed best-tuple reduction (score packing + argmax).

Mirrors the 64-bit packed atomic-CAS reduction used by the GPU kernel at
`FF_HEDM/src/IndexerGPU.cu:516-519`. In the Python port no atomic is needed
because each seed's evaluation tuples form a single contiguous batch — we
just argmax over the packed score key.

Packing: upper 32 bits hold `frac_matches` (maximize), lower 32 bits hold
`~ia_bits` (minimize ia). Reinterpret-cast on the float bits gives a
monotone integer ordering for positive floats, matching `__float_as_int`
semantics from the CUDA code.
"""

from __future__ import annotations

import torch


def pack_score(frac_matches: torch.Tensor, avg_ia: torch.Tensor) -> torch.Tensor:
    """Pack (frac_matches, avg_ia) into a uint64-comparable key.

    Reinterprets the float32 bits to int32 — this is monotone increasing for
    non-negative floats. For `avg_ia` we want smaller is better, so we
    invert the bit pattern (~) before placing it in the low bits.

    Returns int64 (since torch lacks a native unsigned 64-bit type).
    """
    frac_f32 = frac_matches.to(torch.float32).contiguous()
    ia_f32 = avg_ia.to(torch.float32).contiguous()
    frac_bits = (frac_f32.view(torch.int32).to(torch.int64)) & 0xFFFFFFFF
    ia_bits = ((~ia_f32.view(torch.int32)).to(torch.int64)) & 0xFFFFFFFF
    return (frac_bits << 32) | ia_bits


def best_tuple(score_keys: torch.Tensor) -> torch.Tensor:
    """argmax over the packed key vector. Returns int64 index."""
    return torch.argmax(score_keys, dim=-1).to(torch.int64)
