"""Print a compact end-to-end summary of a benchmark run.

Reads the persisted ``BENCHMARK_REPORT.md`` from a workdir and pretty-prints
the timings + tolerances side-by-side.

Usage:
    python print_summary.py /scratch/s1iduser/peakfit_bench
"""
from __future__ import annotations

import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: print_summary.py <workdir>")
        sys.exit(1)
    work_dir = Path(sys.argv[1])
    report = work_dir / "BENCHMARK_REPORT.md"
    if not report.exists():
        print(f"missing report: {report}")
        sys.exit(1)
    print(report.read_text())


if __name__ == "__main__":
    main()
