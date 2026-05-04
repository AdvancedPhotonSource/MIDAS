"""Parameter dataclass for hex grid generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Union


@dataclass
class HexGridParams:
    """All parameters consumed by ``make_hex_grid``.

    Mirrors the keys read by ``MakeHexGrid.c`` L114-L153.
    """

    grid_size: float = 0.0          # GridSize (um)
    edge_length: float = 0.0        # EdgeLength (um); 0 => defaults to grid_size
    r_sample: float = 0.0           # Rsample (um)
    data_directory: str = "."       # DataDirectory
    output_directory: str = ""      # OutputDirectory; empty => data_directory
    grid_filename: str = "grid.txt" # GridFileName

    def __post_init__(self) -> None:
        if self.edge_length == 0.0:
            self.edge_length = self.grid_size
        if not self.output_directory:
            self.output_directory = self.data_directory

    @classmethod
    def from_paramfile(cls, path: Union[str, Path]) -> "HexGridParams":
        """Parse a MIDAS-format parameter file."""
        keys: list[tuple[str, str, type]] = [
            ("GridSize", "grid_size", float),
            ("EdgeLength", "edge_length", float),
            ("Rsample", "r_sample", float),
            ("DataDirectory", "data_directory", str),
            ("OutputDirectory", "output_directory", str),
            ("GridFileName", "grid_filename", str),
        ]
        kwargs: dict = {}
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, rest = parts
                value_token = rest.split(None, 1)[0]
                for c_key, py_field, cast in keys:
                    if key == c_key:
                        try:
                            kwargs[py_field] = cast(value_token)
                        except ValueError:
                            pass
                        break
        return cls(**kwargs)

    def with_overrides(self, **kwargs) -> "HexGridParams":
        return replace(self, **kwargs)
