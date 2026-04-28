"""I/O sub-package — CIF (and future) format readers/writers."""
from .cif import read_cif, write_cif

__all__ = ["read_cif", "write_cif"]
