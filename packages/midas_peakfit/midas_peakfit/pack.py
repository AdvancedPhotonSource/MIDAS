"""Pack/unpack between a ``ParameterSpec`` and a single torch tensor.

Three flavours of packing depending on the inference backend:

- :func:`pack_spec` — pack ALL parameters (refined + fixed) into a flat tensor
  in spec-iteration order, with a parallel index map.  Used by closures that
  need every parameter (the forward model).
- :func:`refined_indices` — index list of refined-only entries within the full
  packed vector; used to mask the gradient.
- :func:`refined_subset` — extract just the refined entries as a flat tensor;
  this is the vector that LM/LBFGS/Adam operates on.

Promoted from ``midas_calibrate_v2.parameters.pack``: the pack/unpack logic
is fully generic — it duck-types on ``spec.parameters: Dict[str, Parameter]``
and works with ``ParameterSpec`` and any subclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from midas_peakfit.spec import ParameterSpec


@dataclass
class PackInfo:
    """Index map produced by :func:`pack_spec`.

    Attributes
    ----------
    names : list[str]
        Parameter names in pack order.
    offsets : list[int]
        Start index of each parameter in the flat vector.
    sizes : list[int]
        Number of scalar entries per parameter.
    refined : list[bool]
        Per-parameter refinement flag.
    n_total : int
    """

    names: List[str]
    offsets: List[int]
    sizes: List[int]
    refined: List[bool]
    n_total: int

    def slice(self, name: str) -> slice:
        i = self.names.index(name)
        return slice(self.offsets[i], self.offsets[i] + self.sizes[i])

    def index_map(self) -> Dict[str, slice]:
        return {n: slice(o, o + s) for n, o, s in zip(self.names, self.offsets, self.sizes)}


def pack_spec(spec: ParameterSpec, dtype=torch.float64, device="cpu",
              ) -> Tuple[torch.Tensor, PackInfo]:
    """Pack every parameter (refined + fixed) into a single flat tensor.

    Returns ``(x, info)`` where ``x[info.slice(name)]`` retrieves the value
    for a given parameter.  The flat tensor is the input to the forward model.
    """
    names: List[str] = []
    offsets: List[int] = []
    sizes: List[int] = []
    refined: List[bool] = []
    pieces: List[torch.Tensor] = []
    cur = 0
    for n, p in spec.parameters.items():
        t = p.init_tensor(dtype=dtype, device=device).reshape(-1)
        names.append(n)
        offsets.append(cur)
        sizes.append(t.numel())
        refined.append(p.refined)
        pieces.append(t)
        cur += t.numel()
    x = torch.cat(pieces) if pieces else torch.zeros(0, dtype=dtype, device=device)
    info = PackInfo(names=names, offsets=offsets, sizes=sizes, refined=refined, n_total=cur)
    return x, info


def unpack_spec(x: torch.Tensor, info: PackInfo, spec: ParameterSpec) -> Dict[str, torch.Tensor]:
    """Slice the flat tensor back into named tensors per parameter.

    The returned dict is what the forward model actually consumes.
    """
    out: Dict[str, torch.Tensor] = {}
    for n, o, s in zip(info.names, info.offsets, info.sizes):
        sub = x[o:o + s]
        # Restore shape from spec
        shape = spec.parameters[n].shape
        if shape == ():
            out[n] = sub.reshape(())
        else:
            out[n] = sub.reshape(shape)
    return out


def refined_indices(info: PackInfo) -> torch.Tensor:
    """Return a long tensor of the indices in the flat vector that are
    refined.  Length = number of refined scalar entries.
    """
    idxs: List[int] = []
    for o, s, r in zip(info.offsets, info.sizes, info.refined):
        if r:
            idxs.extend(range(o, o + s))
    return torch.tensor(idxs, dtype=torch.long)


def refined_bounds(spec: ParameterSpec, info: PackInfo,
                    fallback_span: float = 1.0,
                    dtype=torch.float64, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct (lo, hi) tensors aligned with :func:`refined_subset`.

    For each refined parameter we use its declared ``bounds``; if absent, we
    fabricate symmetric ``±fallback_span`` around the init value.  This is
    purely for inference backends that need a box (LM, sigmoid reparam).
    """
    los: List[float] = []
    his: List[float] = []
    for n, o, s, r in zip(info.names, info.offsets, info.sizes, info.refined):
        if not r:
            continue
        p = spec.parameters[n]
        lo, hi = p.make_logit_bounds(fallback_span=fallback_span)
        for _ in range(s):
            los.append(lo)
            his.append(hi)
    lo_t = torch.tensor(los, dtype=dtype, device=device)
    hi_t = torch.tensor(his, dtype=dtype, device=device)
    return lo_t, hi_t


def refined_subset(x: torch.Tensor, info: PackInfo) -> torch.Tensor:
    """Extract the refined-only entries of the flat vector."""
    idx = refined_indices(info).to(x.device)
    return x.index_select(0, idx)


def write_refined_back(x_full: torch.Tensor, x_refined: torch.Tensor,
                        info: PackInfo) -> torch.Tensor:
    """Place x_refined values back into x_full at refined indices.

    Returns the updated full vector (out-of-place).
    """
    idx = refined_indices(info).to(x_full.device)
    x_new = x_full.clone()
    x_new[idx] = x_refined
    return x_new


# ----------------------------------------------------------- multi-context

@dataclass
class MultiPackInfo:
    """Pack info for a multi-context (multi-image, multi-grain, ...) spec.

    Generic: the per-context block is anything packable (a dict of
    Parameters or a ParameterSpec).  Used by both
    ``midas_calibrate_v2.parameters.spec.MultiImageSpec`` and the
    HEDM-side per-grain layout in ``midas_joint_ff_calibrate``.
    """

    shared_info: PackInfo
    per_context_info: List[PackInfo]
    n_total: int

    # Backwards-compatible alias for the powder-multi-image consumer.
    @property
    def per_image_info(self) -> List[PackInfo]:
        return self.per_context_info


def pack_multi(spec: Any, dtype=torch.float64, device="cpu",
                ) -> Tuple[torch.Tensor, MultiPackInfo]:
    """Pack a multi-context spec into a single flat vector.

    The argument must expose ``shared: Dict[str, Parameter]`` and
    ``per_image: List[Dict[str, Parameter]]`` (the historical
    ``MultiImageSpec`` shape — the name is retained for backwards
    compatibility but the function is duck-typed).

    Layout: ``[shared_block | per-context-1 block | per-context-2 block | ...]``.
    """
    s_spec = ParameterSpec(parameters=spec.shared)
    s_x, s_info = pack_spec(s_spec, dtype=dtype, device=device)

    pieces = [s_x]
    per_infos: List[PackInfo] = []
    base_offset = s_info.n_total
    for ctx_dict in spec.per_image:
        i_spec = ParameterSpec(parameters=ctx_dict)
        i_x, i_info = pack_spec(i_spec, dtype=dtype, device=device)
        i_info_shifted = PackInfo(
            names=list(i_info.names),
            offsets=[o + base_offset for o in i_info.offsets],
            sizes=list(i_info.sizes),
            refined=list(i_info.refined),
            n_total=i_info.n_total,
        )
        per_infos.append(i_info_shifted)
        pieces.append(i_x)
        base_offset += i_info.n_total

    x = torch.cat(pieces)
    return x, MultiPackInfo(shared_info=s_info, per_context_info=per_infos, n_total=base_offset)


__all__ = ["PackInfo", "pack_spec", "unpack_spec",
           "refined_indices", "refined_bounds", "refined_subset",
           "write_refined_back", "MultiPackInfo", "pack_multi"]
