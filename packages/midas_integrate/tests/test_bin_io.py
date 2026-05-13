"""Round-trip Map.bin / nMap.bin in v3 format, with and without header."""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate.bin_io import (
    MAP_HEADER_MAGIC,
    PXLIST_DTYPE,
    MapHeader,
    compute_param_hash,
    load_map,
    write_map,
    write_synthetic_map,
)


def test_pxlist_dtype_size():
    assert PXLIST_DTYPE.itemsize == 24


def test_round_trip_no_header(tmp_path):
    pxList = np.array([
        (10.5, 20.5, 0.7, -0.1, 0.5),
        (11.0, 21.0, 0.3, +0.2, 0.4),
        (12.0, 22.0, 0.8, +0.0, 0.7),
    ], dtype=PXLIST_DTYPE)
    counts = np.array([2, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)

    map_path = tmp_path / "Map.bin"
    nmap_path = tmp_path / "nMap.bin"
    write_map(map_path, nmap_path, pxList=pxList, counts=counts, offsets=offsets)
    pm = load_map(map_path, nmap_path)

    assert pm.n_bins == 2
    assert pm.n_entries == 3
    assert pm.map_header is None
    np.testing.assert_array_equal(pm.counts, counts)
    np.testing.assert_array_equal(pm.offsets, offsets)
    np.testing.assert_array_equal(pm.pxList["y"], pxList["y"])
    np.testing.assert_array_equal(pm.pxList["frac"], pxList["frac"])


def test_round_trip_with_header(tmp_path):
    pxList = np.array([(0.0, 0.0, 1.0, 0.0, 1.0)], dtype=PXLIST_DTYPE)
    counts = np.array([1], dtype=np.int32)
    offsets = np.array([0], dtype=np.int32)
    hdr = MapHeader(q_mode=1, gradient_mode=0, wavelength=0.7293)

    map_path = tmp_path / "Map.bin"
    nmap_path = tmp_path / "nMap.bin"
    write_map(map_path, nmap_path,
              pxList=pxList, counts=counts, offsets=offsets, header=hdr)
    pm = load_map(map_path, nmap_path)
    assert pm.map_header is not None
    assert pm.map_header.magic == MAP_HEADER_MAGIC
    assert pm.map_header.q_mode == 1
    assert abs(pm.map_header.wavelength - 0.7293) < 1e-12


def test_param_hash_deterministic():
    common = dict(
        Lsd=580550.5, Ycen=700.0, Zcen=865.0, pxY=172.0, pxZ=172.0,
        tx=0.0, ty=0.18, tz=0.53,
        p0=0.0, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p6=0.0,
        RhoD=224100.4,
        RBinSize=1.0, EtaBinSize=5.0,
        RMin=10, RMax=1000, EtaMin=-180, EtaMax=180,
        NrPixelsY=1475, NrPixelsZ=1679,
        TransOpt=(2,),
    )
    h1 = compute_param_hash(**common)
    h2 = compute_param_hash(**common)
    assert h1 == h2
    h3 = compute_param_hash(**{**common, "tx": 0.001})
    assert h1 != h3


def test_param_hash_includes_all_15_distortion_coefficients():
    """After the v2 calibration refactor every one of p0..p14 must invalidate
    the hash.  v1's MapHeader.h only hashed p0,p1,p2,p3,p4,p6 — silently
    producing stale Map.bin when any other coefficient changed."""
    common = dict(
        Lsd=580550.5, Ycen=700.0, Zcen=865.0, pxY=172.0, pxZ=172.0,
        tx=0.0, ty=0.18, tz=0.53,
        RhoD=224100.4,
        RBinSize=1.0, EtaBinSize=5.0,
        RMin=10, RMax=1000, EtaMin=-180, EtaMax=180,
        NrPixelsY=1475, NrPixelsZ=1679,
        TransOpt=(2,),
    )
    h0 = compute_param_hash(**common)
    for k in range(15):
        h_perturbed = compute_param_hash(**{**common, f"p{k}": 1.234e-3})
        assert h0 != h_perturbed, (
            f"p{k} did not change the hash — stale-Map.bin bug "
            f"reintroduced (v1 bug fingerprint)"
        )


def test_param_hash_includes_parallax_and_wavelength_unconditionally():
    """v1 only hashed Wavelength when qMode=1.  After v2 the Parallax and
    Wavelength can be refined for any analysis mode; both must invalidate
    the hash."""
    common = dict(
        Lsd=580550.5, Ycen=700.0, Zcen=865.0, pxY=172.0, pxZ=172.0,
        tx=0.0, ty=0.18, tz=0.53,
        RhoD=224100.4,
        RBinSize=1.0, EtaBinSize=5.0,
        RMin=10, RMax=1000, EtaMin=-180, EtaMax=180,
        NrPixelsY=1475, NrPixelsZ=1679,
        TransOpt=(2,),
        qMode=0,                            # IMPORTANT: not in q-mode
    )
    h0 = compute_param_hash(**common)
    h_par = compute_param_hash(**{**common, "Parallax": 1.0})
    assert h0 != h_par, "Parallax changes did not invalidate the hash"
    h_lam = compute_param_hash(**{**common, "Wavelength": 0.5})
    assert h0 != h_lam, ("Wavelength changes did not invalidate the hash "
                          "outside qMode (v2 refines wavelength under priors)")


def test_synthetic_helper_round_trip(tmp_path):
    recs = [
        (1.0, 2.0, 0.5, 0.0, 0.5),
        (2.0, 3.0, 0.3, 0.0, 0.3),
        (3.0, 4.0, 0.2, 0.0, 0.2),
    ]
    bin_lists = [[0, 1], [], [2]]
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=recs,
                        bin_pixel_lists=bin_lists,
                        write_header=True)
    pm = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    assert pm.n_bins == 3
    assert pm.n_entries == 3
    np.testing.assert_array_equal(pm.counts, [2, 0, 1])
    np.testing.assert_array_equal(pm.offsets, [0, 2, 2])
