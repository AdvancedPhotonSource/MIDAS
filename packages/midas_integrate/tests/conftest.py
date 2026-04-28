"""Pytest fixtures shared across midas-integrate tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running tests in-tree.
PKG_DIR = Path(__file__).parent.parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
