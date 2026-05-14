"""Back-compat shim entry point.

Provides a callable for users who want to invoke ``midas-pipeline`` with
the FF interface familiar from ``midas-ff-pipeline run …``. Injects
``--scan-mode ff`` before delegating to :func:`midas_pipeline.cli.main`.

This is exposed in ``pyproject.toml`` as a secondary console-script if
the user wants the alias bundled with this package. The original
``midas-ff-pipeline`` console-script (defined in the separate
``midas-ff-pipeline`` package) remains unchanged and continues to work
— per the no-deletion rule we don't touch it in this effort.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence

from .cli import main as _main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run midas-pipeline with ``--scan-mode ff`` injected after the subcommand.

    If the user did not include a subcommand, default to ``run`` (the
    FF orchestrator's only operational subcommand in legacy usage).
    """
    raw: List[str] = list(argv) if argv is not None else sys.argv[1:]
    inj = _inject_ff(raw)
    return _main(inj)


def _inject_ff(argv: List[str]) -> List[str]:
    """Insert ``--scan-mode ff`` after the subcommand if not already present."""
    if not argv:
        # Default to `run --scan-mode ff` if invoked with no args.
        return ["run", "--scan-mode", "ff"]
    # If user already supplied --scan-mode, respect it.
    if "--scan-mode" in argv:
        return argv
    # First non-flag token is the subcommand; insert immediately after it.
    out = list(argv)
    for i, tok in enumerate(out):
        if not tok.startswith("-"):
            out.insert(i + 1, "ff")
            out.insert(i + 1, "--scan-mode")
            return out
    # All flags, no subcommand — argparse will error cleanly downstream.
    return out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
