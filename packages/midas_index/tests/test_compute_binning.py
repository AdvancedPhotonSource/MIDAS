"""Tests for compute.binning."""

import numpy as np
import torch

from midas_index.compute.binning import get_bin_indices, lookup_bin_counts


def test_get_bin_indices_matches_c_formula():
    # `pos = (ring-1) * (n_eta * n_ome) + iEta * n_ome + iOme`
    n_eta_bins, n_ome_bins = 3600, 3600
    eta_bin_size = 0.1
    ome_bin_size = 0.1
    ring_nr = torch.tensor([1, 2, 5], dtype=torch.int64)
    eta_deg = torch.tensor([0.0, 0.05, -90.0], dtype=torch.float64)
    omega = torch.tensor([0.0, 90.0, 180.0], dtype=torch.float64)

    pos = get_bin_indices(
        ring_nr, eta_deg, omega, eta_bin_size, ome_bin_size, n_eta_bins, n_ome_bins
    )
    # iEta = floor((180+eta)/0.1); iOme = floor((180+omega)/0.1)
    # pos = (ring-1)*N_eta*N_ome + iEta*N_ome + iOme
    expected = (
        torch.tensor([0, 1, 4]) * n_eta_bins * n_ome_bins
        + torch.tensor([1800, 1800, 900]) * n_ome_bins
        + torch.tensor([1800, 2700, 3600])
    )
    np.testing.assert_array_equal(pos.numpy(), expected.numpy())


def test_lookup_bin_counts_returns_count_and_offset():
    # ndata layout: [count_0, offset_0, count_1, offset_1, ...]
    ndata = torch.tensor([3, 0, 1, 3, 5, 4, 0, 9], dtype=torch.int32)
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    n_per, offset = lookup_bin_counts(pos, ndata)
    np.testing.assert_array_equal(n_per.numpy(), [3, 1, 5, 0])
    np.testing.assert_array_equal(offset.numpy(), [0, 3, 4, 9])
