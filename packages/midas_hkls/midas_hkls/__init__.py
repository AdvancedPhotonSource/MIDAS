"""midas-hkls — pure-Python crystallography & HKL list generator.

sginfo-equivalent (Ralf W. Grosse-Kunstleve, 1994-96) via Hall-symbol parsing.
Public API:

    from midas_hkls import SpaceGroup, Lattice, generate_hkls, Reflection

    sg = SpaceGroup.from_number(225)        # CeO2 / NaCl / Cu / Au
    lat = Lattice.for_system("cubic", a=5.411)
    refs = generate_hkls(sg, lat, wavelength_A=0.173, two_theta_max_deg=20.0)
"""
from .hkl_gen import Reflection, generate_hkls, reflections_to_dataframe
from .lattice import Lattice
from .space_group import SpaceGroup, list_space_groups
from .symops import SymOp

__version__ = "0.1.0"

__all__ = [
    "Lattice",
    "Reflection",
    "SpaceGroup",
    "SymOp",
    "generate_hkls",
    "list_space_groups",
    "reflections_to_dataframe",
]
