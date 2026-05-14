"""Unit tests for the provenance ledger."""

from __future__ import annotations

import time

import pytest

from midas_pipeline.provenance import ProvenanceStore, file_sha256


def test_roundtrip(tmp_path):
    store = ProvenanceStore(tmp_path / "run")
    store.record(
        "hkl",
        status="complete",
        started_at=1000.0, finished_at=1001.5,
        outputs={"/tmp/hkls.csv": "deadbeef"},
        metrics={"n_hkls": 42},
    )
    rec = store.read("hkl")
    assert rec is not None
    assert rec["status"] == "complete"
    assert rec["duration_s"] == 1.5
    assert rec["outputs"] == {"/tmp/hkls.csv": "deadbeef"}
    assert rec["metrics"] == {"n_hkls": 42}


def test_missing_returns_none(tmp_path):
    store = ProvenanceStore(tmp_path / "run")
    assert store.read("nope") is None
    assert store.all_stages() == {}


def test_invalidate(tmp_path):
    store = ProvenanceStore(tmp_path / "run")
    store.record("hkl", status="complete")
    assert store.read("hkl") is not None
    store.invalidate("hkl")
    assert store.read("hkl") is None


def test_is_complete_requires_hash_match(tmp_path):
    """is_complete must verify recorded output hashes still match disk."""
    out = tmp_path / "x.csv"
    out.write_text("hello")
    store = ProvenanceStore(tmp_path / "run")
    store.record("hkl", status="complete",
                 outputs={str(out): file_sha256(out)})
    assert store.is_complete("hkl") is True

    # Mutate the file; hash now mismatches → not complete.
    out.write_text("world")
    assert store.is_complete("hkl") is False


def test_is_complete_status_not_complete(tmp_path):
    store = ProvenanceStore(tmp_path / "run")
    store.record("hkl", status="incomplete")
    assert store.is_complete("hkl") is False


def test_all_stages(tmp_path):
    store = ProvenanceStore(tmp_path / "run")
    store.record("hkl", status="complete")
    store.record("peakfit", status="complete")
    everything = store.all_stages()
    assert set(everything) == {"hkl", "peakfit"}


def test_file_sha256_missing():
    assert file_sha256("/nonexistent/path/xxxx") == "missing"


def test_file_sha256_content_hash(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    # echo -n abc | sha256sum → ba7816bf...
    assert file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
