"""ProvenanceStore unit tests."""
from __future__ import annotations

import time
from pathlib import Path

from midas_ff_pipeline.provenance import (
    PROVENANCE_FILENAME,
    ProvenanceStore,
    file_sha256,
)


def test_record_and_read_roundtrip(tmp_path: Path):
    store = ProvenanceStore(tmp_path)
    now = time.time()
    store.record(
        "indexing",
        status="complete", started_at=now, finished_at=now + 5,
        outputs={str(tmp_path / "x"): "abc"},
        metrics={"n_grains": 42},
    )
    rec = store.read("indexing")
    assert rec is not None
    assert rec["status"] == "complete"
    assert rec["duration_s"] == 5.0
    assert rec["metrics"] == {"n_grains": 42}


def test_all_stages_roundtrip(tmp_path: Path):
    store = ProvenanceStore(tmp_path)
    store.record("hkl", duration_s=1.0)
    store.record("indexing", duration_s=2.0, metrics={"n_seeds_indexed": 280})
    stages = store.all_stages()
    assert set(stages) == {"hkl", "indexing"}
    assert stages["indexing"]["metrics"]["n_seeds_indexed"] == 280


def test_invalidate_drops_entry(tmp_path: Path):
    store = ProvenanceStore(tmp_path)
    store.record("indexing", duration_s=1.0)
    assert store.read("indexing") is not None
    store.invalidate("indexing")
    assert store.read("indexing") is None


def test_is_complete_checks_outputs(tmp_path: Path):
    store = ProvenanceStore(tmp_path)
    out = tmp_path / "Grains.csv"
    out.write_text("hello\n")
    h = file_sha256(out)
    store.record("process_grains", outputs={str(out): h})

    assert store.is_complete("process_grains") is True

    # Tampering invalidates.
    out.write_text("modified\n")
    assert store.is_complete("process_grains") is False


def test_is_complete_missing_file(tmp_path: Path):
    store = ProvenanceStore(tmp_path)
    out = tmp_path / "absent.bin"
    store.record("indexing", outputs={str(out): "deadbeef"})
    assert store.is_complete("indexing") is False


def test_provenance_path_in_layer_dir(tmp_path: Path):
    store = ProvenanceStore(tmp_path / "LayerNr_1")
    assert store.path.name == PROVENANCE_FILENAME
    assert store.path.parent.name == "LayerNr_1"
