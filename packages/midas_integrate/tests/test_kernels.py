"""CSR integration kernels — three modes vs explicit reference loop."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.bin_io import (
    PXLIST_DTYPE,
    PixelMap,
    write_synthetic_map,
    load_map,
)
from midas_integrate.kernels import (
    AREA_THRESHOLD,
    build_csr,
    integrate,
    profile_1d,
)


def _make_random_pixmap(rng, NY, NZ, n_r, n_eta,
                        avg_pix_per_bin=6, oob_fraction=0.05):
    n_bins = n_r * n_eta
    rec_list = []
    bin_lists = []
    cur = 0
    n_oob = int(n_bins * oob_fraction)
    oob_bins = set(
        rng.choice(n_bins, size=n_oob, replace=False).tolist()
        if n_oob > 0 else []
    )
    for b in range(n_bins):
        if rng.random() < 0.05:
            bin_lists.append([])
            continue
        n = int(rng.poisson(avg_pix_per_bin)) + 1
        idxs = []
        for _ in range(n):
            y = float(rng.integers(0, NY))
            z = float(rng.integers(0, NZ))
            frac = float(rng.uniform(0.05, 1.0))
            dr = float(rng.uniform(-0.4, 0.4))
            aw = float(rng.uniform(0.1, 1.0))
            rec_list.append((y, z, frac, dr, aw))
            idxs.append(cur)
            cur += 1
        if b in oob_bins and idxs:
            y0, z0, fr, dr, aw = rec_list[idxs[-1]]
            rec_list[idxs[-1]] = (float(NY + 5), z0, fr, dr, aw)
        bin_lists.append(idxs)
    return rec_list, bin_lists


def _reference_floor(image, pxList, counts, offsets, NY, NZ, normalize=True):
    n_bins = counts.shape[0]
    out = np.zeros(n_bins, dtype=np.float64)
    flat = image.reshape(-1)
    for b in range(n_bins):
        n = int(counts[b])
        if n == 0:
            continue
        s = int(offsets[b])
        I = 0.0
        A = 0.0
        for k in range(n):
            e = pxList[s + k]
            y = int(e["y"])
            z = int(e["z"])
            if y < 0 or y >= NY or z < 0 or z >= NZ:
                continue
            offset = z * NY + y
            I += float(flat[offset]) * float(e["frac"])
            A += float(e["areaWeight"])
        if A > AREA_THRESHOLD:
            out[b] = I / A if normalize else I
    return out


def _reference_bilinear(image, pxList, counts, offsets, NY, NZ, normalize=True):
    n_bins = counts.shape[0]
    out = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        n = int(counts[b])
        if n == 0:
            continue
        s = int(offsets[b])
        I = 0.0
        A = 0.0
        for k in range(n):
            e = pxList[s + k]
            y = float(e["y"])
            z = float(e["z"])
            iy = int(np.floor(y))
            iz = int(np.floor(z))
            fy = y - iy
            fz = z - iz
            if iy < 0:
                iy = 0; fy = 0.0
            if iy >= NY - 1:
                iy = NY - 2; fy = 1.0
            if iz < 0:
                iz = 0; fz = 0.0
            if iz >= NZ - 1:
                iz = NZ - 2; fz = 1.0
            v = (image[iz, iy] * (1 - fy) * (1 - fz)
                 + image[iz, iy + 1] * fy * (1 - fz)
                 + image[iz + 1, iy] * (1 - fy) * fz
                 + image[iz + 1, iy + 1] * fy * fz)
            I += float(v) * float(e["frac"])
            A += float(e["areaWeight"])
        if A > AREA_THRESHOLD:
            out[b] = I / A if normalize else I
    return out


def test_floor_mode_matches_reference(tmp_path):
    rng = np.random.default_rng(0xC0FFEE)
    NY, NZ = 64, 48
    n_r, n_eta = 8, 12
    recs, bins = _make_random_pixmap(rng, NY, NZ, n_r, n_eta, avg_pix_per_bin=5)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=recs, bin_pixel_lists=bins,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    image = rng.uniform(0, 1000, size=(NZ, NY)).astype(np.float64)
    ref = _reference_floor(image, pixmap.pxList, pixmap.counts, pixmap.offsets,
                           NY, NZ, normalize=True).reshape(n_r, n_eta)
    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta,
                     n_pixels_y=NY, n_pixels_z=NZ, dtype=torch.float64)
    out = integrate(torch.from_numpy(image), geom, mode="floor",
                    normalize=True).numpy()
    assert np.max(np.abs(out - ref)) < 1e-9


def test_bilinear_mode_matches_reference(tmp_path):
    rng = np.random.default_rng(0xBEE)
    NY, NZ = 64, 48
    n_r, n_eta = 8, 12
    recs, bins = _make_random_pixmap(rng, NY, NZ, n_r, n_eta, avg_pix_per_bin=5,
                                     oob_fraction=0.0)  # bilinear bounds-checks differently
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=recs, bin_pixel_lists=bins,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    image = rng.uniform(0, 1000, size=(NZ, NY)).astype(np.float64)
    ref = _reference_bilinear(image, pixmap.pxList, pixmap.counts,
                              pixmap.offsets, NY, NZ, normalize=True
                              ).reshape(n_r, n_eta)
    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta,
                     n_pixels_y=NY, n_pixels_z=NZ, dtype=torch.float64)
    out = integrate(torch.from_numpy(image), geom, mode="bilinear",
                    normalize=True).numpy()
    assert np.max(np.abs(out - ref)) < 1e-9


def test_profile_1d_area_weighted_matches_dense(tmp_path):
    rng = np.random.default_rng(0xDEAD)
    NY, NZ = 32, 24
    n_r, n_eta = 6, 10
    recs, bins = _make_random_pixmap(rng, NY, NZ, n_r, n_eta, oob_fraction=0.0)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=recs, bin_pixel_lists=bins,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    image = rng.uniform(0, 100, size=(NZ, NY)).astype(np.float64)
    geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta,
                     n_pixels_y=NY, n_pixels_z=NZ, dtype=torch.float64)
    int2d = integrate(torch.from_numpy(image), geom, mode="floor",
                      normalize=True)
    prof = profile_1d(int2d, geom, mode="area_weighted").numpy()
    # Manual: Σ(I*A) / Σ(A) per row
    area_2d = geom.area_per_bin.reshape(n_r, n_eta).numpy()
    int_np = int2d.numpy()
    valid = area_2d > AREA_THRESHOLD
    num = (int_np * area_2d * valid).sum(axis=1)
    den = (area_2d * valid).sum(axis=1)
    expected = np.where(den > AREA_THRESHOLD, num / np.maximum(den, AREA_THRESHOLD), 0.0)
    np.testing.assert_array_almost_equal(prof, expected, decimal=12)


def test_device_cpu_works_for_all_modes(tmp_path):
    """Smoke test: all three modes run on CPU without crashing."""
    rng = np.random.default_rng(0)
    NY, NZ = 16, 16
    n_r, n_eta = 4, 6
    recs, bins = _make_random_pixmap(rng, NY, NZ, n_r, n_eta, oob_fraction=0)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=recs, bin_pixel_lists=bins,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    image = rng.uniform(0, 10, size=(NZ, NY)).astype(np.float64)
    for dtype in (torch.float32, torch.float64):
        geom = build_csr(pixmap, n_r=n_r, n_eta=n_eta,
                         n_pixels_y=NY, n_pixels_z=NZ, dtype=dtype,
                         bc_y=NY / 2, bc_z=NZ / 2)
        for mode in ("floor", "bilinear", "gradient"):
            out = integrate(torch.from_numpy(image), geom, mode=mode)
            assert out.shape == (n_r, n_eta)
            prof = profile_1d(out, geom)
            assert prof.shape == (n_r,)


@pytest.mark.gpu
def test_cuda_matches_cpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    rng = np.random.default_rng(123)
    NY, NZ = 64, 48
    n_r, n_eta = 8, 12
    recs, bins = _make_random_pixmap(rng, NY, NZ, n_r, n_eta, oob_fraction=0)
    write_synthetic_map(tmp_path / "Map.bin", tmp_path / "nMap.bin",
                        pxList_records=recs, bin_pixel_lists=bins,
                        write_header=False)
    pixmap = load_map(tmp_path / "Map.bin", tmp_path / "nMap.bin")
    image = rng.uniform(0, 1000, size=(NZ, NY)).astype(np.float64)
    geom_cpu = build_csr(pixmap, n_r=n_r, n_eta=n_eta,
                         n_pixels_y=NY, n_pixels_z=NZ, dtype=torch.float64)
    geom_gpu = build_csr(pixmap, n_r=n_r, n_eta=n_eta,
                         n_pixels_y=NY, n_pixels_z=NZ, dtype=torch.float64,
                         device="cuda")
    img_cpu = torch.from_numpy(image)
    img_gpu = img_cpu.to("cuda")
    for mode in ("floor", "bilinear"):
        a = integrate(img_cpu, geom_cpu, mode=mode).numpy()
        b = integrate(img_gpu, geom_gpu, mode=mode).cpu().numpy()
        np.testing.assert_array_almost_equal(a, b, decimal=10)
