"""Unit tests for ``midas_nf_pipeline.params``."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from midas_nf_pipeline.params import (
    collect_multiline,
    parse_parameters,
    update_param_file,
)


def _write(td, content):
    p = Path(td) / "test.txt"
    p.write_text(content)
    return p


def test_basic_keys_strings_first_token():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "Foo bar\nBaz qux\n")
        result = parse_parameters(p)
        assert result["Foo"] == "bar"
        assert result["Baz"] == "qux"


def test_lattice_parameter_six_floats():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "LatticeParameter 4.08 4.08 4.08 90 90 90\n")
        result = parse_parameters(p)
        assert result["LatticeParameter"] == [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]


def test_grid_refactor_three_floats():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "GridRefactor 5.0 2.0 3\n")
        result = parse_parameters(p)
        assert result["GridRefactor"] == [5.0, 2.0, 3.0]


def test_nDistances_int():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "nDistances 4\n")
        result = parse_parameters(p)
        assert result["nDistances"] == 4
        assert isinstance(result["nDistances"], int)


def test_skip_comments_blanks():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "# comment\n\n  \nFoo bar\n")
        result = parse_parameters(p)
        assert result == {"Foo": "bar"}


def test_duplicate_keys_last_wins():
    """Match C MIDAS_ParamParser overwrite behaviour."""
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "Lsd 100\nLsd 200\nLsd 300\n")
        result = parse_parameters(p)
        assert result["Lsd"] == "300"


def test_collect_multiline_returns_all():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "Lsd 100\nLsd 200\n")
        assert collect_multiline(p, "Lsd") == ["100", "200"]


def test_update_param_file_inplace():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "Foo bar\nBaz qux\n")
        update_param_file(p, {"Foo": "newval"})
        text = p.read_text()
        assert "Foo newval\n" in text
        assert "Baz qux\n" in text
        # Foo should appear exactly once.
        assert text.count("Foo ") == 1


def test_update_param_file_appends_unknown():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "Foo bar\n")
        update_param_file(p, {"NewKey": "newval"})
        text = p.read_text()
        assert "Foo bar\n" in text
        assert "NewKey newval\n" in text


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_parameters("/no/such/file.txt")


def test_too_few_values_raises():
    with tempfile.TemporaryDirectory() as td:
        p = _write(td, "LatticeParameter 4.08 4.08\n")
        with pytest.raises(ValueError, match="LatticeParameter"):
            parse_parameters(p)
