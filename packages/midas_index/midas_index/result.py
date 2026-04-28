"""Result dataclasses for the indexer.

`SeedResult`: one record per indexed seed spot — the best (orientation,
position) tuple found, plus matched-spot bookkeeping.

`IndexerResult`: collection wrapper over a list of SeedResult, with
serializers to BestPos_<block_nr>.csv and the consolidated binary format.

Implementation lands in P5 (pipeline + reduce).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class SeedResult:
    """Best evaluation tuple for a single seed spot."""

    spot_id: int
    best_or_mat: "torch.Tensor"      # (3, 3)
    best_pos: "torch.Tensor"         # (3,) sample-frame (ga, gb, gc)
    n_matches: int
    n_t_spots: int
    n_t_frac_calc: int
    frac_matches: float
    avg_ia: float
    matched_ids: "torch.Tensor"      # (n_matches,) int


@dataclass
class IndexerResult:
    """Block-level collection of SeedResult records."""

    block_nr: int
    n_blocks: int
    seeds: list[SeedResult] = field(default_factory=list)
