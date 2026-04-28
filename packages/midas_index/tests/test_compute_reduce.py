"""Tests for compute.reduce."""

import torch

from midas_index.compute.reduce import best_tuple, pack_score


def test_pack_score_higher_frac_wins():
    frac = torch.tensor([0.5, 0.7, 0.6])
    ia = torch.tensor([1.0, 1.0, 1.0])  # equal IAs
    keys = pack_score(frac, ia)
    assert int(best_tuple(keys).item()) == 1   # frac=0.7 wins


def test_pack_score_lower_ia_wins_on_frac_tie():
    frac = torch.tensor([0.5, 0.5, 0.5])
    ia = torch.tensor([2.0, 1.0, 3.0])  # smallest IA at idx 1
    keys = pack_score(frac, ia)
    assert int(best_tuple(keys).item()) == 1


def test_pack_score_frac_takes_precedence_over_ia():
    frac = torch.tensor([0.6, 0.7])
    ia = torch.tensor([0.1, 5.0])  # idx 0 has lower IA, but lower frac
    keys = pack_score(frac, ia)
    assert int(best_tuple(keys).item()) == 1   # frac wins regardless of IA


def test_pack_score_handles_zero_frac():
    frac = torch.tensor([0.0, 0.0])
    ia = torch.tensor([1.0, 0.5])
    keys = pack_score(frac, ia)
    assert int(best_tuple(keys).item()) == 1
