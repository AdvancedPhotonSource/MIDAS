"""Torch backend coverage for the consolidation_pf module.

The consolidation logic itself is single-row CSV ingestion + scalar
quat reduction; no per-voxel batched math benefits from torch. We do
expose one helper (``reduce_quats_to_fz_torch``) that forwards to
``midas_stress.orientation.fundamental_zone`` for callers that already
hold a batched quaternion tensor. These tests cover that helper
against the NumPy path on every supported device and verify autograd
flows through.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_pipeline.stages.consolidation_pf import reduce_quats_to_fz_torch
from midas_stress.orientation import fundamental_zone


def _sample_quats() -> np.ndarray:
    # A handful of unit quaternions, picked so FZ reduction is non-trivial
    # under cubic symmetry (sg=225).
    q = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0],
        [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0],
        [np.cos(np.pi / 6), 0.0, 0.0, np.sin(np.pi / 6)],
    ], dtype=np.float64)
    # Normalize, FZ requires unit quats.
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def test_reduce_quats_to_fz_torch_matches_numpy_cpu():
    q_np = _sample_quats()
    q_t = torch.from_numpy(q_np).to(dtype=torch.float64)
    out_t = reduce_quats_to_fz_torch(q_t, space_group=225)
    # Reference via NumPy backend
    out_np = np.stack([
        fundamental_zone(qr, space_group=225) for qr in q_np
    ])
    np.testing.assert_allclose(out_t.detach().cpu().numpy(), out_np,
                               atol=1e-10, rtol=0)


def test_reduce_quats_to_fz_torch_is_differentiable():
    """Gradient must flow back through the fundamental_zone op."""
    q = torch.tensor(
        [[1.0, 0.05, 0.05, 0.05]], dtype=torch.float64, requires_grad=True,
    )
    q_norm = q / q.norm(dim=-1, keepdim=True)
    out = reduce_quats_to_fz_torch(q_norm, space_group=225)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert torch.isfinite(q.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_reduce_quats_to_fz_torch_runs_on_cuda():
    q = torch.from_numpy(_sample_quats()).to(
        device="cuda", dtype=torch.float64,
    )
    out = reduce_quats_to_fz_torch(q, space_group=225)
    assert out.device.type == "cuda"
    assert out.shape == q.shape


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_reduce_quats_to_fz_torch_runs_on_mps():
    # MPS is fp32 only; the round-trip must still produce unit quats.
    q = torch.from_numpy(_sample_quats()).to(
        device="mps", dtype=torch.float32,
    )
    out = reduce_quats_to_fz_torch(q, space_group=225)
    assert out.device.type == "mps"
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
