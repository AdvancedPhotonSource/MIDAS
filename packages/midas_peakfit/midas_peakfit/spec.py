"""ParameterSpec — a generic registry of named ``Parameter`` objects.

The unifying abstraction across every differentiable inverse problem in MIDAS
(powder calibration, HEDM grain refinement, joint calibration): every
quantity that *could* be refined is declared as a :class:`Parameter`; the
forward model takes a single packed tensor; refined components carry
autograd, fixed components don't.

``ParameterSpec`` is intentionally minimal — it carries only the parameter
registry + manipulation helpers (``add``, ``freeze``, ``thaw``, ...). Domain-
specific specs (e.g. ``midas_calibrate_v2.parameters.spec.CalibrationSpec``)
inherit and add their own metadata (lattice, ring tables, panel layout, ...).

Pack/unpack (``midas_peakfit.pack``) and the inference backends
(``midas_peakfit.lm_spec``, ``midas_peakfit.laplace``) operate on anything
that exposes a ``.parameters: Dict[str, Parameter]`` attribute, so they work
with both ``ParameterSpec`` and any subclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from midas_peakfit.parameter import Parameter


@dataclass
class ParameterSpec:
    """A registry of named :class:`Parameter` objects.

    Construct directly or subclass for domain-specific extras.
    """

    parameters: Dict[str, Parameter] = field(default_factory=dict)

    # ---------------------------------------------------------- helpers
    def add(self, p: Parameter) -> None:
        if p.name in self.parameters:
            raise ValueError(f"Parameter {p.name!r} already exists in spec")
        self.parameters[p.name] = p

    def get(self, name: str) -> Parameter:
        return self.parameters[name]

    def names(self) -> List[str]:
        return list(self.parameters.keys())

    def refined_names(self) -> List[str]:
        return [n for n, p in self.parameters.items() if p.refined]

    def freeze(self, *names: str) -> None:
        for n in names:
            self.parameters[n].refined = False

    def thaw(self, *names: str) -> None:
        for n in names:
            self.parameters[n].refined = True

    def set_init(self, name: str, value) -> None:
        p = self.parameters[name]
        if isinstance(value, torch.Tensor):
            p.init = value
            p.shape = tuple(value.shape)
        elif isinstance(value, (list, tuple)):
            arr = torch.as_tensor(value, dtype=torch.float64)
            p.init = arr
            p.shape = tuple(arr.shape)
        else:
            p.init = float(value)
            p.shape = ()

    def set_prior(self, name: str, prior) -> None:
        self.parameters[name].prior = prior

    def __contains__(self, name: str) -> bool:
        return name in self.parameters


__all__ = ["ParameterSpec"]
