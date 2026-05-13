"""Powder-diffraction stress analyses (sin²ψ method, hkl-strain pooling).

Companion to the FF-HEDM grain-resolved stress workflow already in
midas_stress: where FF-HEDM gives per-grain ε / σ tensors, powder
diffraction averages over the illuminated volume and reports a single
σ tensor (or its in-plane components). The `sin²ψ` ring is the
standard tool for this.

This sub-module is the foundation for the powder-side cross-check on
the same sample geometry — e.g., the same in-situ tensile load tracked
by both FF-HEDM and powder diffraction (Item 48 of the differentiable
integrate scope expansion).
"""
from .sin2psi import (
    extract_d_vs_psi,
    fit_sin2psi,
    Sin2PsiResult,
)

__all__ = ["extract_d_vs_psi", "fit_sin2psi", "Sin2PsiResult"]
