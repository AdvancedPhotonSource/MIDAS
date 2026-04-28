"""CLI smoke tests."""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from midas_hkls import cli


def test_list_runs():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["list"])
    assert rc == 0
    out = buf.getvalue()
    # 530 entries in the table
    assert out.count("\n") == 530


def test_info_with_ops():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["info", "--sg", "225", "--ops"])
    assert rc == 0
    out = buf.getvalue()
    assert "Hall symbol" in out
    assert "Order" in out
    assert "Centric      : True" in out


def test_gen_to_stdout():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main([
            "gen", "--sg", "225",
            "--lat", "5.411", "5.411", "5.411", "90", "90", "90",
            "--wavelength", "0.173",
            "--two-theta-max", "10.0",
        ])
    assert rc == 0
    rows = buf.getvalue().strip().split("\n")
    assert rows[0].startswith("ring_nr,h,k,l,d_spacing,two_theta_deg,multiplicity")
    assert len(rows) > 5  # several rings within 10° 2θ
