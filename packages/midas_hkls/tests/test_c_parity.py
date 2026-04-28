"""Byte-level parity test: midas-hkls ring list vs the C GetHKLList tool.

We aggregate the C output (one row per equivalent reflection) up to ring level
and compare ring count, ring d-spacing, ring multiplicity, and ring 2θ against
midas-hkls's `generate_hkls` output for the same SG / lattice / wavelength.
"""
from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import List, NamedTuple

import pytest

from midas_hkls import Lattice, SpaceGroup, generate_hkls

GETHKL = Path("/Users/hsharma/opt/MIDAS/build/bin/GetHKLList")

pytestmark = pytest.mark.skipif(not GETHKL.exists(), reason="C GetHKLList not built")


class RingC(NamedTuple):
    ring_nr: int
    d: float
    two_theta: float
    multiplicity: int


def _c_rings(sg: int, lat: tuple[float, ...], wl: float, lsd: float = 1_000_000.0,
             max_r: float = 100_000.0) -> List[RingC]:
    out = subprocess.run(
        [str(GETHKL), "--sg", str(sg),
         "--lp", *[str(v) for v in lat],
         "--wl", str(wl),
         "--lsd", str(lsd),
         "--maxR", str(max_r),
         "--stdout"],
        capture_output=True, text=True, check=True,
    ).stdout

    rings: dict[int, dict] = {}
    for line in out.splitlines():
        toks = line.split()
        if len(toks) < 11:
            continue
        # First numeric token is h; first column we want is RingNr at index 4.
        try:
            ring_nr = int(toks[4])
            d = float(toks[3])
            two_theta = float(toks[9])
        except ValueError:
            continue
        info = rings.setdefault(ring_nr, {"d": d, "tt": two_theta, "count": 0})
        info["count"] += 1
        # Sanity — all rows in a ring must agree on d-spacing
        assert abs(info["d"] - d) < 1e-9, f"d-spacing inconsistency in C ring {ring_nr}"
    return [RingC(n, info["d"], info["tt"], info["count"]) for n, info in sorted(rings.items())]


CASES = [
    # (sg, lattice_constants, wavelength, label)
    (225, (5.411, 5.411, 5.411, 90, 90, 90), 0.173, "CeO2 Fm-3m"),
    (221, (4.156, 4.156, 4.156, 90, 90, 90), 0.173, "LaB6 Pm-3m"),
    (227, (5.431, 5.431, 5.431, 90, 90, 90), 0.173, "Si Fd-3m"),
    (229, (2.866, 2.866, 2.866, 90, 90, 90), 0.173, "Fe-α Im-3m"),
    (194, (2.95, 2.95, 4.686, 90, 90, 120), 0.173, "Ti-α P63/mmc"),
    (167, (4.99, 4.99, 17.06, 90, 90, 120), 0.173, "Calcite R-3c"),
    (62,  (5.41, 5.95, 6.78, 90, 90, 90),  0.173, "Pnma"),
    (14,  (5.5,  6.0,  7.5,  90, 95, 90),  0.173, "P21/c monoclinic"),
]


@pytest.mark.parametrize("sg_num, lat_args, wl, label", CASES, ids=[c[3] for c in CASES])
def test_ring_parity(sg_num, lat_args, wl, label):
    sg = SpaceGroup.from_number(sg_num)
    lat = Lattice(*lat_args)

    c_rings = _c_rings(sg_num, lat_args, wl)
    if not c_rings:
        pytest.skip("C GetHKLList produced no rings")
    d_min = min(r.d for r in c_rings) * (1 - 1e-6)
    refs = generate_hkls(sg, lat, wavelength_A=wl, d_min=d_min)

    assert len(refs) >= len(c_rings), (
        f"{label}: midas-hkls has fewer rings ({len(refs)}) than C ({len(c_rings)})"
    )

    # midas-hkls may include a few additional very-near-d_min rings; align by d.
    refs_in_range = [r for r in refs if r.d_spacing >= c_rings[-1].d - 1e-6]
    assert len(refs_in_range) == len(c_rings), (
        f"{label}: ring count mismatch  midas={len(refs_in_range)}  C={len(c_rings)}"
    )

    for c, p in zip(c_rings, refs_in_range):
        assert abs(c.d - p.d_spacing) < 1e-6, (
            f"{label} ring {c.ring_nr}: d  C={c.d!r}  midas={p.d_spacing!r}"
        )
        assert c.multiplicity == p.multiplicity, (
            f"{label} ring {c.ring_nr}: multiplicity  C={c.multiplicity}  midas={p.multiplicity}"
        )
        assert abs(c.two_theta - p.two_theta_deg) < 1e-6, (
            f"{label} ring {c.ring_nr}: 2θ  C={c.two_theta!r}  midas={p.two_theta_deg!r}"
        )
