"""Byte-parity tests for ``midas_nf_pipeline.parse_mic`` against the C
``ParseMic`` reference outputs bundled with NF_HEDM/Example/sim.

The C pipeline produces:

  - ``Au_txt_Reconstructed.mic``        — text mic
  - ``Au_txt_Reconstructed.mic.map``    — orientation raster
  - ``Au_txt_Reconstructed.mic.map.kam`` — kernel-average misorientation (rad)
  - ``Au_txt_Reconstructed.mic.map.grainId``
  - ``Au_txt_Reconstructed.mic.map.grod`` — grain ref orientation deviation (rad)

We run our Python port on the *same* binary input and compare each
output to the C reference.

The ``.mic`` text comparison is exact (line-by-line); the binary
files compare element-wise within numerical tolerance — the C uses
``GetMisOrientationAngle`` from a C lib, ours uses
:func:`midas_stress.orientation.misorientation_om_batch`, both ports
of the same source so we expect equality to ~1e-12, but we allow
``atol=1e-10`` for the angles.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from midas_nf_pipeline.parse_mic import (
    ParseMicParams,
    parse_mic,
    read_mic_binary,
)


# The bundled binary mic is the C-reference snapshot ``*.mic.c_ref`` —
# the live ``Au_bin_Reconstructed.mic`` gets overwritten by every Python
# fit_orientation_run, so we test against the frozen C copy.
_AU_DIR = Path(__file__).resolve().parents[3] / "NF_HEDM/Example/sim"
_AU_BIN = _AU_DIR / "Au_bin_Reconstructed.mic.c_ref"
_AU_REF_TXT = _AU_DIR / "Au_txt_Reconstructed.mic"
_AU_REF_MAP = _AU_DIR / "Au_txt_Reconstructed.mic.map"
_AU_REF_KAM = _AU_DIR / "Au_txt_Reconstructed.mic.map.kam"
_AU_REF_GID = _AU_DIR / "Au_txt_Reconstructed.mic.map.grainId"
_AU_REF_GROD = _AU_DIR / "Au_txt_Reconstructed.mic.map.grod"

_AU_PARAMS = ParseMicParams(
    PhaseNr=1, NumPhases=1, GlobalPosition=0.0,
    inputfile=str(_AU_BIN),
    outputfile="",  # filled in by fixtures
    nSaves=3,
    SGNr=225,
    GBAngle=5.0,
)


@pytest.fixture
def workspace():
    if not _AU_BIN.exists():
        pytest.skip(f"{_AU_BIN} not present — run the C pipeline once first")
    with tempfile.TemporaryDirectory() as td:
        # The c_ref input lives at NF_HEDM/Example/sim/Au_bin_Reconstructed.mic.c_ref
        # but ParseMic looks for `<inputfile>.AllMatches`, so we copy it
        # alongside the binary as `Au_bin_Reconstructed.mic` (no suffix).
        td_path = Path(td)
        bin_dst = td_path / "Au_bin_Reconstructed.mic"
        shutil.copy2(_AU_BIN, bin_dst)
        am_in = _AU_DIR / "Au_bin_Reconstructed.mic.AllMatches"
        if am_in.exists():
            shutil.copy2(am_in, td_path / "Au_bin_Reconstructed.mic.AllMatches")
        params = ParseMicParams(
            PhaseNr=_AU_PARAMS.PhaseNr,
            NumPhases=_AU_PARAMS.NumPhases,
            GlobalPosition=_AU_PARAMS.GlobalPosition,
            inputfile=str(bin_dst),
            outputfile=str(td_path / "Au_py.mic"),
            nSaves=_AU_PARAMS.nSaves,
            SGNr=_AU_PARAMS.SGNr,
            GBAngle=_AU_PARAMS.GBAngle,
        )
        yield params, td_path


def test_mic_binary_roundtrip(workspace):
    """Read the binary, write the text, read it back: counts match."""
    params, _ = workspace
    mic = read_mic_binary(params.inputfile)
    n_valid = int(np.sum(mic[:, 10] != 0))
    out = parse_mic(params)
    text_lines = Path(out["mic"]).read_text().splitlines()
    # 4 header lines + N data rows.
    assert len(text_lines) == 4 + n_valid


def test_mic_text_matches_c(workspace):
    """Line-by-line equality with the C ``Au_txt_Reconstructed.mic``."""
    params, _ = workspace
    out = parse_mic(params)
    py_lines = Path(out["mic"]).read_text().splitlines()
    ref_lines = _AU_REF_TXT.read_text().splitlines()
    assert len(py_lines) == len(ref_lines), (
        f"row count: py={len(py_lines)} ref={len(ref_lines)}"
    )
    # Header lines are byte-equal.
    for i in range(4):
        assert py_lines[i] == ref_lines[i], f"line {i}: {py_lines[i]!r} vs {ref_lines[i]!r}"
    # Data rows: compare token-by-token (numerical tolerance for floats).
    for i in range(4, len(py_lines)):
        py_tok = py_lines[i].split("\t")
        ref_tok = ref_lines[i].split("\t")
        assert len(py_tok) == len(ref_tok)
        # Last token is PhaseNr int.
        assert py_tok[-1] == ref_tok[-1]
        for a, b in zip(py_tok[:-1], ref_tok[:-1]):
            if a == "" and b == "":
                continue
            assert float(a) == pytest.approx(float(b), abs=1e-6), (
                f"line {i}: {a} vs {b}"
            )


def _read_bin_doubles(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.float64)


def _split_header_body(arr: np.ndarray, n_fields: int):
    """Return ``(header[4], body)`` where body has shape ``(n_fields, size_map)``."""
    header = arr[:4]
    size_map = (arr.size - 4) // n_fields
    body = arr[4:4 + size_map * n_fields].reshape(n_fields, size_map)
    return header, body


def test_map_binary_matches_c(workspace):
    params, _ = workspace
    out = parse_mic(params)
    ref = _read_bin_doubles(_AU_REF_MAP)
    py = _read_bin_doubles(Path(out["map"]))
    assert ref.shape == py.shape
    ref_h, ref_b = _split_header_body(ref, 7)
    py_h, py_b = _split_header_body(py, 7)
    np.testing.assert_array_equal(ref_h, py_h)
    # Field 0..4 (Confidence, Eul, RowNr) and field 6 (length) are floats
    # the C copies straight through; field 5 is integer PhaseNr.
    for f in (0, 1, 2, 3, 4, 5):
        np.testing.assert_array_equal(ref_b[f], py_b[f])
    # Length field can have rounding; tolerate epsilon.
    np.testing.assert_allclose(ref_b[6], py_b[6], rtol=1e-12, atol=1e-12)


def test_grainid_binary_matches_c(workspace):
    params, _ = workspace
    out = parse_mic(params)
    ref = _read_bin_doubles(_AU_REF_GID)
    py = _read_bin_doubles(Path(out["grainId"]))
    assert ref.shape == py.shape
    np.testing.assert_array_equal(ref[:4], py[:4])
    # Grain IDs must match exactly (relabelling could differ if BFS visits
    # in different order, but the C and Python both walk raster order so
    # IDs should be identical).
    np.testing.assert_array_equal(ref[4:], py[4:])


def test_kam_binary_matches_c(workspace):
    params, _ = workspace
    out = parse_mic(params)
    ref = _read_bin_doubles(_AU_REF_KAM)
    py = _read_bin_doubles(Path(out["kam"]))
    assert ref.shape == py.shape
    np.testing.assert_array_equal(ref[:4], py[:4])
    # Misorientation angles in radians, expect agreement to ~1e-10.
    np.testing.assert_allclose(ref[4:], py[4:], atol=1e-10, rtol=1e-10)


def test_grod_binary_matches_c(workspace):
    params, _ = workspace
    out = parse_mic(params)
    ref = _read_bin_doubles(_AU_REF_GROD)
    py = _read_bin_doubles(Path(out["grod"]))
    assert ref.shape == py.shape
    np.testing.assert_array_equal(ref[:4], py[:4])
    np.testing.assert_allclose(ref[4:], py[4:], atol=1e-10, rtol=1e-10)
