"""Consolidated binary writer / reader (scanning indexer format).

The format defined in `FF_HEDM/src/IndexerConsolidatedIO.h` is used by the
**scanning** indexer (`IndexerScanningOMP`/`IndexerScanningGPU`). v0.1.0 of
midas-index targets the **non-scanning** indexer, whose output is the
pwrite-addressed binaries in `output.py` — not these consolidated files.

This module is a placeholder for a future scanning-indexer port.
"""
