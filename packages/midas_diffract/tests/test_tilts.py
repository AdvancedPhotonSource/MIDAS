"""Cross-code validation of detector tilts in NF-HEDM mode.

The differentiable forward model applies detector tilts only in NF mode
(flip_y=False). FF and pf-HEDM use tilt-free projection because the
experimental SpotMatrix.csv centroids are already detector-tilt and
distortion corrected by MIDAS (YOrig(DetCor)/ZOrig(DetCor) in the
InputAllExtraInfoFittingAll.csv header).

This test cross-validates the NF tilt port against simulateNF with
non-zero tx, ty, tz. Companion to test_c_comparison.py which uses
tilts=0 for both FF and NF.

Run:
    cd packages/midas_diffract
    python -m pytest tests/test_tilts.py -v -s

Override the MIDAS install location with ``MIDAS_HOME=/path/to/MIDAS``.
"""
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_diffract.forward import HEDMForwardModel, HEDMGeometry, TriVoxelConfig

MIDAS_HOME = Path(os.environ.get("MIDAS_HOME", "/Users/hsharma/opt/MIDAS"))
BUILD_BIN = MIDAS_HOME / "build" / "bin"

pytestmark = pytest.mark.skipif(
    not (BUILD_BIN / "simulateNF").exists()
    or not (BUILD_BIN / "GetHKLList").exists(),
    reason="simulateNF not built",
)

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def setup_nf_with_tilts(work: Path, tx: float, ty: float, tz: float):
    """Single-voxel NF simulation config with arbitrary tilts."""
    euler_deg = np.array([45.0, 30.0, 60.0])
    euler_rad = euler_deg * DEG2RAD
    vx, vy = 0.0, 0.0
    edge_len = 5.0
    ud = 1.0
    latc = [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]
    wl = 0.172979
    Lsd = 5000.0      # 5 mm for NF
    px_size = 1.48
    y_bc, z_bc = 1024.0, 12.0
    n_pix = 2048
    omega_start, omega_step = 0.0, 0.25
    n_frames = 1440

    mic_file = work / "test.mic"
    with open(mic_file, "w") as f:
        f.write("%TriEdgeSize 0.500000\n")
        f.write("%NumPhases 1\n")
        f.write("%GlobalPosition 0.000000\n")
        f.write("%OrientationRowNr OrientationID RunTime X Y TriEdgeSize "
                "UpDown Eul1 Eul2 Eul3 Confidence PhaseNr\n")
        f.write(f"1 1 0 {vx:.6f} {vy:.6f} {edge_len:.6f} "
                f"{ud:.6f} {euler_rad[0]:.17e} {euler_rad[1]:.17e} "
                f"{euler_rad[2]:.17e} 1.000000 1\n")

    param_file = work / "params_nf.txt"
    with open(param_file, "w") as f:
        f.write(f"LatticeParameter {' '.join(str(v) for v in latc)}\n")
        f.write(f"Wavelength {wl}\n")
        f.write("SpaceGroup 225\n")
        f.write("nDistances 1\n")
        f.write(f"Lsd {Lsd}\n")
        f.write(f"BC {y_bc} {z_bc}\n")
        f.write(f"tx {tx}\nty {ty}\ntz {tz}\nWedge 0\n")
        f.write(f"px {px_size}\nNrPixels {n_pix}\n")
        f.write(f"MaxRingRad {Lsd * 0.5}\n")
        f.write(f"OmegaStart {omega_start}\nOmegaStep {omega_step}\n")
        f.write(f"OmegaRange {omega_start} {omega_start + omega_step * n_frames}\n")
        f.write(f"BoxSize -100000 100000 -100000 100000\n")
        f.write(f"StartNr 0\nEndNr {n_frames - 1}\n")
        f.write("ExcludePoleAngle 6\n")
        f.write(f"NrFilesPerDistance {n_frames}\nSaveReducedOutput 0\n")
        f.write("WriteImage 0\nOnlySpotsInfo 0\n")

    return dict(
        work=work, param_file=param_file, mic_file=mic_file,
        euler_rad=euler_rad, voxel_pos=np.array([vx, vy, 0.0]),
        edge_len=edge_len, ud=ud, latc=latc, wl=wl, Lsd=Lsd, px=px_size,
        y_bc=y_bc, z_bc=z_bc, n_pix=n_pix,
        omega_start=omega_start, omega_step=omega_step, n_frames=n_frames,
        tx=tx, ty=ty, tz=tz,
    )


def parse_hkls_csv(path):
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    hkls_int = data[:, 0:3]
    g_cart = data[:, 5:8]
    theta_deg = data[:, 8]
    return (
        torch.tensor(g_cart, dtype=torch.float64),
        torch.tensor(theta_deg * DEG2RAD, dtype=torch.float64),
        torch.tensor(hkls_int, dtype=torch.float64),
    )


@pytest.mark.parametrize("tx,ty,tz", [
    (0.5, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, 0.0, 0.2),
    (-0.08, -0.26, 0.01),
    (1.0, -0.5, 0.3),
])
def test_nf_tilts_match_c(tx, ty, tz, tmp_path):
    work = tmp_path / "nf_tilt_test"
    work.mkdir()
    cfg = setup_nf_with_tilts(work, tx, ty, tz)

    # Run simulateNF
    out_fn = str(work / "sim_out")
    subprocess.run(
        [str(BUILD_BIN / "simulateNF"), str(cfg["param_file"]),
         str(cfg["mic_file"]), out_fn],
        cwd=str(work), capture_output=True, text=True, timeout=120, check=True,
    )
    # Output is {out_fn}_SimulatedSpots.csv
    spots_file = work / "SimulatedSpots.csv"
    assert spots_file.exists(), f"SimulatedSpots.csv not produced at {spots_file}"
    # header: VoxRowNr DistanceNr FrameNr HorPx VerPx OmegaRaw YRaw ZRaw
    c_data = np.loadtxt(spots_file, skiprows=1)
    if c_data.ndim == 1:
        c_data = c_data.reshape(1, -1)
    c_dist = c_data[:, 1].astype(int)
    c_frame = c_data[:, 2].astype(int)
    c_hor = c_data[:, 3]
    c_ver = c_data[:, 4]
    c_set = set((int(d), int(fr), int(round(y)), int(round(z)))
                for d, fr, y, z in zip(c_dist, c_frame, c_hor, c_ver))

    # Run PyTorch NF forward with same tilts
    hkls_file = work / "hkls.csv"
    assert hkls_file.exists()
    hkls_cart, thetas, hkls_int = parse_hkls_csv(hkls_file)

    geom = HEDMGeometry(
        Lsd=cfg["Lsd"], y_BC=cfg["y_bc"], z_BC=cfg["z_bc"], px=cfg["px"],
        omega_start=cfg["omega_start"], omega_step=cfg["omega_step"],
        n_frames=cfg["n_frames"], n_pixels_y=cfg["n_pix"],
        n_pixels_z=cfg["n_pix"], min_eta=6.0, wavelength=cfg["wl"],
        flip_y=False,  # NF convention
        tx=cfg["tx"], ty=cfg["ty"], tz=cfg["tz"],
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geom,
        hkls_int=hkls_int, device=torch.device("cpu"),
    )

    # Read Euler angles from .mic to match precision
    with open(cfg["mic_file"]) as mf:
        for line in mf:
            if line.startswith("%"):
                continue
            parts = line.split()
            euler_from_mic = [float(parts[7]), float(parts[8]), float(parts[9])]
            break
    euler = torch.tensor([euler_from_mic], dtype=torch.float64)
    centers = torch.tensor(cfg["voxel_pos"][:2], dtype=torch.float64).unsqueeze(0)
    tri = TriVoxelConfig(
        edge_lengths=torch.tensor([cfg["edge_len"]], dtype=torch.float64),
        ud=torch.tensor([cfg["ud"]], dtype=torch.float64),
    )
    py_hits = model.forward_nf_triangles(euler, centers, tri)
    py_set = set((d, fr, y, z) for (vox, d, fr, y, z, ome) in py_hits)

    common = c_set & py_set
    only_c = c_set - py_set
    only_py = py_set - c_set

    n_c = len(c_set)
    n_py = len(py_set)
    match_frac = len(common) / max(1, n_c)
    print(f"  tx={tx:+.2f} ty={ty:+.2f} tz={tz:+.2f} "
          f"| C={n_c} Py={n_py} common={len(common)} ({100*match_frac:.1f}%)")
    if only_c and only_py:
        print(f"    only C sample: {list(only_c)[:3]}")
        print(f"    only Py sample: {list(only_py)[:3]}")

    assert match_frac >= 0.95, \
        f"NF tilt match only {match_frac:.1%} — expected ≥95% pixel agreement"
