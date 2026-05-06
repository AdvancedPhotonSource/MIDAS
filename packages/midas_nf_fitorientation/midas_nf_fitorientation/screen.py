"""Phase 1: hard, non-differentiable orientation screening.

Replicates the C ``CalcFracOverlap`` loop that survives between
``FitOrientationOMP``, ``FitOrientationParameters``, and
``FitOrientationParametersMultiPoint``. For each voxel, evaluate every
candidate orientation in ``OrientMat.bin`` and keep those whose fraction
of detector pixels matched against ``SpotsInfo.bin`` exceeds
``MinFracAccept``.

This version uses the precomputed lab-frame ``(yl, zl, omega_deg)``
spots written by ``midas_nf_preprocess`` (the
:program:`MakeDiffrSpots` replacement) into ``DiffractionSpots.bin``.
For each voxel we project those precomputed spots through each of the
three voxel-triangle vertices via the ``DisplacementSpots`` formula,
optionally apply the NF tilt correction at the primary distance, then
either rasterise the projected triangle on the detector grid (when
``2*gs > px``, matching ``CalcPixels2``) or use the rounded centroid
(``2*gs <= px`` branch, matching the inline path at
SharedFuncsFit.c:595-600). Each rasterised pixel is then looked up at
each detector distance and AND-ed across distances to produce the
``OverlapPixels / TotalPixels`` fraction the C code computes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .io import GridTable, OrientationData
from .obs_volume import ObsVolume
from .params import FitParams
from .reparam import normalize_orient_mat


# ---------------------------------------------------------------------------
#  OrientMat → ZXZ Euler conversion
# ---------------------------------------------------------------------------

def orientmat_to_euler_zxz(matrices: np.ndarray) -> np.ndarray:
    """Convert ``(N, 3, 3)`` proper-rotation matrices to Bunge ZXZ Euler
    triplets (radians).

    Matches ``OrientMat2Euler`` in MIDAS C: when ``|R[2,2]| < 1`` use the
    standard formula, otherwise fall back to the gimbal-locked branch.

    Each input matrix is first passed through
    :func:`midas_nf_fitorientation.reparam.normalize_orient_mat`
    (the ``NormalizeMat`` port — scales by ``det^(-1/3)`` to remove the
    drift accumulated in ``OrientMat.bin``); the C readers do this
    every time they pull a row out of the binary, and skipping it
    leaves ``arccos`` slightly off when the on-disk determinant
    differs from 1.
    """
    R = np.asarray(matrices)
    if R.ndim == 2:
        R = R[None, ...]
    R = normalize_orient_mat(R)
    N = R.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)
    cosPhi = R[:, 2, 2].clip(-1.0, 1.0)
    Phi = np.arccos(cosPhi)
    sinPhi = np.sqrt(1.0 - cosPhi * cosPhi)
    locked = sinPhi < 1e-9
    nl = ~locked
    out[nl, 1] = Phi[nl]
    out[nl, 0] = np.arctan2(R[nl, 0, 2], -R[nl, 1, 2])
    out[nl, 2] = np.arctan2(R[nl, 2, 0],  R[nl, 2, 1])
    out[locked, 1] = Phi[locked]
    out[locked, 0] = np.arctan2(R[locked, 1, 0], R[locked, 0, 0])
    out[locked, 2] = 0.0
    return out


# ---------------------------------------------------------------------------
#  Tilt rotation matrix (RotationTilts in SharedFuncsFit.c)
# ---------------------------------------------------------------------------

def build_rot_tilts(
    tx_deg: float, ty_deg: float, tz_deg: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Construct the 3×3 NF tilt rotation matrix ``Rz(tz) @ Ry(ty) @ Rx(tx)``.

    Direct port of ``RotationTilts`` from
    ``NF_HEDM/src/SharedFuncsFit.c:230-266``.
    """
    d2r = math.pi / 180.0
    tx, ty, tz = tx_deg * d2r, ty_deg * d2r, tz_deg * d2r
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    Rx = torch.tensor(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
        dtype=dtype, device=device,
    )
    Ry = torch.tensor(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]],
        dtype=dtype, device=device,
    )
    Rz = torch.tensor(
        [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
        dtype=dtype, device=device,
    )
    return Rz @ Ry @ Rx


def apply_nf_tilt(
    ydet: torch.Tensor, zdet: torch.Tensor,
    Lsd_val: float | torch.Tensor,
    R: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ray-plane intersection that applies a tilted-detector correction.

    Direct port of the inline block in
    ``NF_HEDM/src/SharedFuncsFit.c:947-958`` (also present as
    ``midas_diffract.HEDMForwardModel._apply_nf_tilt``). Reduces to the
    identity when ``R == I``.

    Both ``ydet`` and ``zdet`` are tensors of arbitrary shape;
    everything broadcasts.
    """
    if not torch.is_tensor(Lsd_val):
        Lsd_val = torch.tensor(float(Lsd_val), dtype=ydet.dtype, device=ydet.device)
    p0x = -Lsd_val * R[0, 0]
    p0y = -Lsd_val * R[1, 0]
    p0z = -Lsd_val * R[2, 0]
    P1x = ydet * R[0, 1] + zdet * R[0, 2]
    P1y = ydet * R[1, 1] + zdet * R[1, 2]
    P1z = ydet * R[2, 1] + zdet * R[2, 2]
    ABCx = P1x - p0x
    ABCy = P1y - p0y
    ABCz = P1z - p0z
    safe = torch.where(
        torch.abs(ABCx) < eps,
        torch.full_like(ABCx, eps),
        ABCx,
    )
    out_y = p0y - ABCy * p0x / safe
    out_z = p0z - ABCz * p0x / safe
    return out_y, out_z


# ---------------------------------------------------------------------------
#  Result containers
# ---------------------------------------------------------------------------

@dataclass
class Winner:
    voxel_idx: int
    orient_idx: int
    frac_overlap: float


@dataclass
class ScreenResult:
    """Per-voxel screening outcome.

    Attributes
    ----------
    winners : list[Winner]
        All ``(voxel, orient, frac)`` tuples whose ``frac`` ≥
        ``MinFracAccept``. Sorted by ``(voxel_idx, orient_idx)`` so it
        matches the C ``screen_cpu.csv`` diagnostic dump column-for-column.
    n_winners_per_voxel : np.ndarray (V,) int64
        Convenience counter for downstream allocation.
    """
    winners: List[Winner]
    n_winners_per_voxel: np.ndarray


# ---------------------------------------------------------------------------
#  Vectorised triangle rasteriser (port of CalcPixels2)
# ---------------------------------------------------------------------------

def rasterize_triangles(
    rel_y: torch.Tensor, rel_z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised rasteriser for ``T`` triangles in integer pixel space.

    Direct port of the half-plane + 0.99-pixel-distance test in
    ``CalcPixels2`` (``NF_HEDM/src/SharedFuncsFit.c:308-370``):

    - vertices are first ``round()``-snapped to integers (line 312-313);
    - a pixel is "inside" if all three signed-area orient2d products
      are non-negative;
    - if it isn't strictly inside, the pixel is still kept when its
      perpendicular squared distance to *any* of the three edges is
      below 0.9801 (= 0.99²).

    Parameters
    ----------
    rel_y, rel_z : Tensor (T, 3) int or float
        The three vertex pixel coordinates per triangle. Both inputs
        live in the same coordinate frame; the rasteriser returns its
        output offsets in that frame too.

    Returns
    -------
    abs_y, abs_z : Tensor (T, K) int64
        Per-triangle pixel coordinates over the bounding-box union.
        The same ``(P, Q)`` grid is used for every triangle so the
        leading-T tensors line up.
    valid : Tensor (T, K) bool
        Per-triangle in-triangle mask. Pixels failing both the
        half-plane test and the 0.99-edge-distance test are False.
    """
    if rel_y.shape != rel_z.shape:
        raise ValueError(
            f"rel_y / rel_z shape mismatch: {rel_y.shape} vs {rel_z.shape}"
        )
    if rel_y.ndim != 2 or rel_y.shape[1] != 3:
        raise ValueError(f"expected (T, 3), got {rel_y.shape}")

    device = rel_y.device
    rel_y = rel_y.round().long()
    rel_z = rel_z.round().long()

    min_y = rel_y.min(dim=1).values         # (T,)
    max_y = rel_y.max(dim=1).values
    min_z = rel_z.min(dim=1).values
    max_z = rel_z.max(dim=1).values

    P = int((max_y - min_y).max().item()) + 1
    Q = int((max_z - min_z).max().item()) + 1

    if P <= 0 or Q <= 0:
        # Degenerate (single-point triangles) — fall through with one
        # pixel per triangle.
        P = max(P, 1)
        Q = max(Q, 1)

    # Build the (T, P, Q) absolute pixel grid, anchored at each
    # triangle's bbox min corner.
    ys_grid = torch.arange(P, device=device, dtype=torch.long).reshape(1, P, 1)
    zs_grid = torch.arange(Q, device=device, dtype=torch.long).reshape(1, 1, Q)
    abs_y = min_y.reshape(-1, 1, 1) + ys_grid          # (T, P, Q)
    abs_z = min_z.reshape(-1, 1, 1) + zs_grid          # (T, P, Q)

    # Vertex tensors (T, 2) — broadcast against (T, P, Q) via reshape.
    v0y = rel_y[:, 0:1].unsqueeze(2)
    v0z = rel_z[:, 0:1].unsqueeze(2)
    v1y = rel_y[:, 1:2].unsqueeze(2)
    v1z = rel_z[:, 1:2].unsqueeze(2)
    v2y = rel_y[:, 2:3].unsqueeze(2)
    v2z = rel_z[:, 2:3].unsqueeze(2)

    # orient2d(a, b, p) = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
    # using (y, z) as (x, y) per the C convention in CalcPixels2.
    def orient2d(ay, az, by, bz, py, pz):
        return (by - ay) * (pz - az) - (bz - az) * (py - ay)

    w0 = orient2d(v1y, v1z, v2y, v2z, abs_y, abs_z)
    w1 = orient2d(v2y, v2z, v0y, v0z, abs_y, abs_z)
    w2 = orient2d(v0y, v0z, v1y, v1z, abs_y, abs_z)
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

    # 0.99-pixel edge tolerance: dist² < 0.9801.
    # distSq2d(a, b, c) = orient2d² / |ab|²  (perpendicular distance²
    # from c to line ab); see SharedFuncsFit.c:302-306.
    def dist_sq(ay, az, by, bz, w):
        # w is the precomputed orient2d(a, b, .). |ab|² = (a.y-b.y)²+(b.x-a.x)²
        # in the C (y, z) convention.
        denom = (az - bz) ** 2 + (by - ay) ** 2
        denom_f = denom.to(torch.float64).clamp(min=1.0)  # avoid div-by-0
        return (w.to(torch.float64) ** 2) / denom_f

    near_edge = (
        (dist_sq(v1y, v1z, v2y, v2z, w0) < 0.9801)
        | (dist_sq(v2y, v2z, v0y, v0z, w1) < 0.9801)
        | (dist_sq(v0y, v0z, v1y, v1z, w2) < 0.9801)
    )
    # Clip each triangle's mask to its own bbox: the C rasteriser
    # iterates only over a triangle's own [min_y..max_y] × [min_z..max_z],
    # so an edge-tolerance pixel belonging to a *different* triangle in
    # the batch must not bleed in here. Without this, batches of
    # triangles with different bbox sizes produce false-positive pixels
    # near the union-bbox boundary.
    own_bbox = (
        (abs_y >= min_y.reshape(-1, 1, 1))
        & (abs_y <= max_y.reshape(-1, 1, 1))
        & (abs_z >= min_z.reshape(-1, 1, 1))
        & (abs_z <= max_z.reshape(-1, 1, 1))
    )
    valid = (inside | near_edge) & own_bbox             # (T, P, Q)

    K = P * Q
    return (
        abs_y.expand(rel_y.shape[0], P, Q).reshape(-1, K),
        abs_z.expand(rel_y.shape[0], P, Q).reshape(-1, K),
        valid.reshape(-1, K),
    )


# ---------------------------------------------------------------------------
#  Screen kernel (DiffractionSpots.bin path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def screen(
    grid: GridTable,
    orientations: OrientationData,
    obs: ObsVolume,
    p: FitParams,
    *,
    voxel_indices: Optional[np.ndarray] = None,
    progress: Optional[callable] = None,
    dtype: torch.dtype = torch.float64,
) -> ScreenResult:
    """Screen every (voxel, orientation) pair against the obs bitmap.

    Uses the precomputed ``DiffractionSpots.bin`` rows
    ``(yl, zl, omega_deg)`` produced by :program:`midas-nf-preprocess`
    (the :program:`MakeDiffrSpots` replacement). Per voxel we project
    those rows through the voxel position via the ``DisplacementSpots``
    formula, scale to each detector distance, apply the NF tilt
    correction if any tilts are non-zero, look up the ``ObsVolume``
    bitmap, AND-product across distances, and aggregate hits per
    orientation via ``scatter_add``.

    Parameters
    ----------
    grid : GridTable
        Voxel positions (centroids) from ``grid.txt``.
    orientations : OrientationData
        Bundle of ``OrientMat.bin`` + ``Key.bin`` + ``DiffractionSpots.bin``.
    obs : ObsVolume
        Decoded ``SpotsInfo.bin`` bitmap, on ``device``.
    p : FitParams
        Reads ``min_frac_accept``, per-distance ``Lsd``/``ybc``/``zbc``,
        ``tx/ty/tz``, ``px``, ``omega_start``, ``omega_step``,
        ``n_pixels_y``/``n_pixels_z``, and ``n_frames_per_distance``.
    voxel_indices : np.ndarray, optional
        Subset of voxel indices to screen (used by the block-decomposition
        in ``FitOrientationOMP``). ``None`` ⇒ all voxels.
    progress : callable, optional
        ``progress(done, total)`` callback after each voxel.
    dtype : torch.dtype
        Working precision for the projection arithmetic. Default
        ``float64`` matches the C path; ``float32`` is fine for
        screening on GPU.

    Returns
    -------
    :class:`ScreenResult`
    """
    device = obs.device

    if voxel_indices is None:
        voxel_indices = np.arange(grid.n_voxels)

    # ---- ship orientation data to device (one shot) ----
    n_orient = orientations.n_orientations
    n_spots_t = torch.from_numpy(
        orientations.n_spots.astype(np.int64)
    ).to(device)
    starts_t = torch.from_numpy(
        orientations.starts.astype(np.int64)
    ).to(device)
    spots_t = torch.from_numpy(
        np.asarray(orientations.spots, dtype=np.float64)
    ).to(device=device, dtype=dtype)
    yl_0 = spots_t[:, 0]
    zl_0 = spots_t[:, 1]
    omega_deg = spots_t[:, 2]
    omega_rad = omega_deg * (math.pi / 180.0)
    cos_w = torch.cos(omega_rad)
    sin_w = torch.sin(omega_rad)
    T = spots_t.shape[0]

    # ---- per-spot orientation index for scatter_add ----
    # spot_to_orient[t] = orientation owning spot t. Build by repeating
    # arange(N) by n_spots[i], using torch.repeat_interleave.
    spot_to_orient = torch.repeat_interleave(
        torch.arange(n_orient, device=device, dtype=torch.int64),
        n_spots_t,
    )
    if spot_to_orient.numel() != T:
        raise ValueError(
            f"Inconsistent: spot_to_orient has {spot_to_orient.numel()} entries "
            f"but DiffractionSpots.bin has {T} rows; check Key.bin counts."
        )

    # ---- per-distance geometry tensors ----
    Lsd = torch.tensor(p.Lsd, device=device, dtype=dtype)        # (D,)
    ybc = torch.tensor(p.ybc, device=device, dtype=dtype)        # (D,)
    zbc = torch.tensor(p.zbc, device=device, dtype=dtype)        # (D,)
    Lsd_0 = Lsd[0]
    px = float(p.px)
    n_y = int(p.n_pixels_y)
    n_z = int(p.n_pixels_z)
    D = int(p.n_distances)
    n_frames = int(p.n_frames_per_distance)

    has_tilts = bool(p.tx != 0 or p.ty != 0 or p.tz != 0)
    R_tilt = build_rot_tilts(p.tx, p.ty, p.tz, device, dtype) if has_tilts else None

    # ---- per-spot frame index (orientation-independent of voxel) ----
    frame_idx = (
        (omega_deg - p.omega_start) / p.omega_step
    ).long()
    frame_in_range = (frame_idx >= 0) & (frame_idx < n_frames)
    frame_clamped = frame_idx.clamp(0, n_frames - 1)

    # ---- vectorised across voxels ----
    # Replaces the per-voxel Python loop. All V voxels go through the
    # projection / rasterisation / multi-distance lookup / scatter-add
    # together, leaving a single (V, n_orient) frac tensor at the end.
    # On H100 with V≈3.5k, T≈1k this drops the screen step from ~10s to
    # well under a second versus the old per-voxel loop.
    winners: List[Winner] = []
    counts = np.zeros(grid.n_voxels, dtype=np.int64)

    voxel_indices_arr = np.asarray(voxel_indices, dtype=np.int64)
    V = int(voxel_indices_arr.shape[0])
    if V == 0:
        return ScreenResult(winners=winners, n_winners_per_voxel=counts)

    # Per-voxel super-pixel decision.  All voxels in the batch must
    # pick the same path (the centroid vs. rasterise branch is decided
    # by the gs/px ratio).  In every NF dataset we've seen gs is
    # uniform across voxels, so this just collapses to one bool.  If
    # we ever encounter a mixed grid we fall back to the legacy
    # per-voxel path below.
    gs_arr = grid.gs[voxel_indices_arr]
    super_pixel_mask = (2.0 * gs_arr) > px
    all_super = bool(super_pixel_mask.all())
    none_super = bool((~super_pixel_mask).all())

    if not (all_super or none_super):
        # Mixed gs across voxels — fall back to the per-voxel loop.
        return _screen_per_voxel(
            grid=grid, voxel_indices=voxel_indices_arr,
            cos_w=cos_w, sin_w=sin_w, yl_0=yl_0, zl_0=zl_0,
            Lsd=Lsd, Lsd_0=Lsd_0, ybc=ybc, zbc=zbc,
            R_tilt=R_tilt, has_tilts=has_tilts, px=px,
            n_y=n_y, n_z=n_z, D=D, T=T,
            frame_clamped=frame_clamped, frame_in_range=frame_in_range,
            spot_to_orient=spot_to_orient, n_orient=n_orient,
            obs=obs, p=p, device=device, dtype=dtype,
            counts=counts, winners=winners,
        )

    # ---- one-shot HtoD of voxel triangle vertices ----
    # ``triangle_vertices`` returns a (3, 2) array; stack to (V, 3, 2).
    verts_np = np.empty((V, 3, 2), dtype=np.float64)
    for k, vi in enumerate(voxel_indices_arr):
        verts_np[k] = grid.triangle_vertices(int(vi))
    XG_all = torch.from_numpy(verts_np[:, :, 0].copy()).to(
        device=device, dtype=dtype,
    )                                                         # (V, 3)
    YG_all = torch.from_numpy(verts_np[:, :, 1].copy()).to(
        device=device, dtype=dtype,
    )

    # ---- (V, T, 3) projection ----
    # DisplacementSpots (SharedFuncsFit.c:269-292), broadcast over the
    # voxel axis.  cos/sin are spot-only (1, T, 1); XG/YG are voxel-only
    # (V, 1, 3); the tensor that comes out is (V, T, 3).
    cos_w_ = cos_w.reshape(1, T, 1)
    sin_w_ = sin_w.reshape(1, T, 1)
    XG_ = XG_all.reshape(V, 1, 3)
    YG_ = YG_all.reshape(V, 1, 3)
    xa = XG_ * cos_w_ - YG_ * sin_w_                          # (V, T, 3)
    ya = XG_ * sin_w_ + YG_ * cos_w_
    t_0 = 1.0 - xa / Lsd_0
    yl_0_ = yl_0.reshape(1, T, 1)
    zl_0_ = zl_0.reshape(1, T, 1)
    dy_v = ya + yl_0_ * t_0
    dz_v = zl_0_ * t_0
    if has_tilts:
        dy_v, dz_v = apply_nf_tilt(dy_v, dz_v, Lsd_0, R_tilt)
    v_y_px = dy_v / px + ybc[0]                               # (V, T, 3)
    v_z_px = dz_v / px + zbc[0]
    del xa, ya, t_0, dy_v, dz_v

    # ---- spot centres at primary distance (voxel-independent) ----
    cy_lab = yl_0
    cz_lab = zl_0
    if has_tilts:
        cy_lab, cz_lab = apply_nf_tilt(cy_lab, cz_lab, Lsd_0, R_tilt)
    c_y_px = cy_lab / px + ybc[0]                             # (T,)
    c_z_px = cz_lab / px + zbc[0]

    # Vertex-on-detector + frame-in-range gate. (V, T)
    vert_in_bounds = (
        (v_y_px >= 0) & (v_y_px < n_y)
        & (v_z_px >= 0) & (v_z_px < n_z)
    ).all(dim=2) & frame_in_range.unsqueeze(0)

    rel_y = v_y_px - c_y_px.reshape(1, T, 1)                  # (V, T, 3)
    rel_z = v_z_px - c_z_px.reshape(1, T, 1)
    del v_y_px, v_z_px

    if all_super:
        # CalcPixels2 path: rasterise V*T triangles in one shot.
        offsets_y_flat, offsets_z_flat, valid_flat = rasterize_triangles(
            rel_y.reshape(V * T, 3), rel_z.reshape(V * T, 3),
        )
        K = offsets_y_flat.shape[1]
        offsets_y = offsets_y_flat.reshape(V, T, K)           # (V, T, K)
        offsets_z = offsets_z_flat.reshape(V, T, K)
        valid_mask = valid_flat.reshape(V, T, K) & vert_in_bounds.unsqueeze(-1)
    else:
        # Single rounded centroid (SharedFuncsFit.c:595-600).
        cent_y = ((rel_y[..., 0] + rel_y[..., 1] + rel_y[..., 2]) / 3.0
                  ).round().long()                            # (V, T)
        cent_z = ((rel_z[..., 0] + rel_z[..., 1] + rel_z[..., 2]) / 3.0
                  ).round().long()
        offsets_y = cent_y.unsqueeze(-1)                      # (V, T, 1)
        offsets_z = cent_z.unsqueeze(-1)
        valid_mask = vert_in_bounds.unsqueeze(-1)
        K = 1
    del rel_y, rel_z

    # ---- per-distance, per-pixel multi-distance lookup ----
    hits_per_pixel = torch.ones((V, T, K), device=device, dtype=dtype)
    bounds_per_pixel = valid_mask.clone()
    for d in range(D):
        scale = (Lsd[d] / Lsd_0)
        cy_d = (c_y_px - ybc[0]) * scale + ybc[d]             # (T,)
        cz_d = (c_z_px - zbc[0]) * scale + zbc[d]
        abs_y = cy_d.floor().long().reshape(1, T, 1) + offsets_y   # (V, T, K)
        abs_z = cz_d.floor().long().reshape(1, T, 1) + offsets_z

        in_bd = (
            (abs_y >= 0) & (abs_y < n_y)
            & (abs_z >= 0) & (abs_z < n_z)
        )
        bounds_per_pixel = bounds_per_pixel & in_bd

        ay = abs_y.clamp(0, n_y - 1)
        az = abs_z.clamp(0, n_z - 1)
        f_idx = frame_clamped.reshape(1, T, 1).expand_as(ay)
        d_idx = torch.full_like(f_idx, d)
        hit = obs.lookup(d_idx, f_idx, ay, az)                # uint8 0/1
        hits_per_pixel = hits_per_pixel * hit.to(dtype)

    hits_per_pixel = hits_per_pixel * bounds_per_pixel.to(dtype)
    total_per_spot = bounds_per_pixel.to(dtype).sum(dim=2)    # (V, T)
    hits_per_spot = hits_per_pixel.sum(dim=2)
    del hits_per_pixel, bounds_per_pixel

    # ---- scatter spot-level totals into (V, n_orient) ----
    hits_per_orient = torch.zeros((V, n_orient), device=device, dtype=dtype)
    total_per_orient = torch.zeros((V, n_orient), device=device, dtype=dtype)
    spot_to_orient_b = spot_to_orient.unsqueeze(0).expand(V, T)
    hits_per_orient.scatter_add_(1, spot_to_orient_b, hits_per_spot)
    total_per_orient.scatter_add_(1, spot_to_orient_b, total_per_spot)
    frac = hits_per_orient / total_per_orient.clamp(min=1.0)

    # ---- collect winners across all (voxel, orient) pairs ----
    keep = frac >= p.min_frac_accept
    keep_pairs = torch.nonzero(keep, as_tuple=False)          # (W, 2)
    if keep_pairs.numel() > 0:
        keep_frac_t = frac[keep_pairs[:, 0], keep_pairs[:, 1]]
        keep_v_local = keep_pairs[:, 0].cpu().numpy()
        keep_o = keep_pairs[:, 1].cpu().numpy().astype(np.int64)
        keep_v = voxel_indices_arr[keep_v_local]
        keep_f = keep_frac_t.cpu().numpy()
        # Sort by (voxel_idx, orient_idx) so winners list ordering
        # matches the legacy per-voxel iteration (and the C
        # screen_cpu.csv diagnostic) byte-for-byte.
        order = np.lexsort((keep_o, keep_v))
        for i in order:
            v_int = int(keep_v[i])
            winners.append(Winner(v_int, int(keep_o[i]), float(keep_f[i])))
            counts[v_int] += 1

    if progress is not None:
        progress(V, V)

    return ScreenResult(winners=winners, n_winners_per_voxel=counts)


# ---------------------------------------------------------------------------
#  Per-voxel fallback (mixed-gs grids)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _screen_per_voxel(
    *,
    grid: GridTable,
    voxel_indices: np.ndarray,
    cos_w: torch.Tensor, sin_w: torch.Tensor,
    yl_0: torch.Tensor, zl_0: torch.Tensor,
    Lsd: torch.Tensor, Lsd_0: torch.Tensor,
    ybc: torch.Tensor, zbc: torch.Tensor,
    R_tilt: Optional[torch.Tensor], has_tilts: bool, px: float,
    n_y: int, n_z: int, D: int, T: int,
    frame_clamped: torch.Tensor, frame_in_range: torch.Tensor,
    spot_to_orient: torch.Tensor, n_orient: int,
    obs: ObsVolume, p: FitParams,
    device: torch.device, dtype: torch.dtype,
    counts: np.ndarray, winners: List[Winner],
) -> ScreenResult:
    """Legacy per-voxel screen kernel; used only when ``gs`` varies
    across the batch.  See the docstring of :func:`screen` for the
    physics — this is a literal copy of the pre-vectorised loop body."""
    for vi_local, vi in enumerate(voxel_indices):
        verts = grid.triangle_vertices(int(vi))
        XG = torch.tensor(verts[:, 0], device=device, dtype=dtype)
        YG = torch.tensor(verts[:, 1], device=device, dtype=dtype)

        cos_w_ = cos_w.unsqueeze(1)
        sin_w_ = sin_w.unsqueeze(1)
        XG_ = XG.unsqueeze(0)
        YG_ = YG.unsqueeze(0)
        xa = XG_ * cos_w_ - YG_ * sin_w_
        ya = XG_ * sin_w_ + YG_ * cos_w_
        t_0 = 1.0 - xa / Lsd_0
        dy_v = ya + yl_0.unsqueeze(1) * t_0
        dz_v = zl_0.unsqueeze(1) * t_0
        if has_tilts:
            dy_v, dz_v = apply_nf_tilt(dy_v, dz_v, Lsd_0, R_tilt)
        v_y_px = dy_v / px + ybc[0]
        v_z_px = dz_v / px + zbc[0]

        cy_lab = yl_0
        cz_lab = zl_0
        if has_tilts:
            cy_lab, cz_lab = apply_nf_tilt(cy_lab, cz_lab, Lsd_0, R_tilt)
        c_y_px = cy_lab / px + ybc[0]
        c_z_px = cz_lab / px + zbc[0]

        vert_in_bounds = (
            (v_y_px >= 0) & (v_y_px < n_y)
            & (v_z_px >= 0) & (v_z_px < n_z)
        ).all(dim=1) & frame_in_range

        rel_y = v_y_px - c_y_px.unsqueeze(1)
        rel_z = v_z_px - c_z_px.unsqueeze(1)

        gs_um = float(grid.gs[vi])
        super_pixel = (2.0 * gs_um) > px

        if super_pixel:
            offsets_y, offsets_z, valid_mask = rasterize_triangles(rel_y, rel_z)
            valid_mask = valid_mask & vert_in_bounds.unsqueeze(1)
        else:
            cent_y = ((rel_y[:, 0] + rel_y[:, 1] + rel_y[:, 2]) / 3.0).round().long()
            cent_z = ((rel_z[:, 0] + rel_z[:, 1] + rel_z[:, 2]) / 3.0).round().long()
            offsets_y = cent_y.unsqueeze(1)
            offsets_z = cent_z.unsqueeze(1)
            valid_mask = vert_in_bounds.unsqueeze(1)

        K = offsets_y.shape[1]
        hits_per_pixel = torch.ones((T, K), device=device, dtype=dtype)
        bounds_per_pixel = valid_mask.clone()
        for d in range(D):
            scale = (Lsd[d] / Lsd_0)
            cy_d = (c_y_px - ybc[0]) * scale + ybc[d]
            cz_d = (c_z_px - zbc[0]) * scale + zbc[d]
            abs_y = cy_d.floor().long().unsqueeze(1) + offsets_y
            abs_z = cz_d.floor().long().unsqueeze(1) + offsets_z
            in_bd = (
                (abs_y >= 0) & (abs_y < n_y)
                & (abs_z >= 0) & (abs_z < n_z)
            )
            bounds_per_pixel = bounds_per_pixel & in_bd
            ay = abs_y.clamp(0, n_y - 1)
            az = abs_z.clamp(0, n_z - 1)
            f_idx = frame_clamped.unsqueeze(1).expand_as(ay)
            d_idx = torch.full_like(f_idx, d)
            hit = obs.lookup(d_idx, f_idx, ay, az)
            hits_per_pixel = hits_per_pixel * hit.to(dtype)

        hits_per_pixel = hits_per_pixel * bounds_per_pixel.to(dtype)
        total_per_spot = bounds_per_pixel.to(dtype).sum(dim=1)
        hits_per_spot = hits_per_pixel.sum(dim=1)

        hits_per_orient = torch.zeros(n_orient, device=device, dtype=dtype)
        total_per_orient = torch.zeros(n_orient, device=device, dtype=dtype)
        hits_per_orient.scatter_add_(0, spot_to_orient, hits_per_spot)
        total_per_orient.scatter_add_(0, spot_to_orient, total_per_spot)
        frac = hits_per_orient / total_per_orient.clamp(min=1.0)

        keep = frac >= p.min_frac_accept
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
        if keep_idx.numel() > 0:
            keep_frac = frac[keep_idx].cpu().numpy()
            keep_orient = keep_idx.cpu().numpy().astype(np.int64)
            for f, oi in zip(keep_frac, keep_orient):
                winners.append(Winner(int(vi), int(oi), float(f)))
                counts[vi] += 1

    return ScreenResult(winners=winners, n_winners_per_voxel=counts)


# ---------------------------------------------------------------------------
#  Diagnostic CSV (matches C screen_cpu.csv)
# ---------------------------------------------------------------------------

def write_screen_csv(result: ScreenResult, path: str) -> None:
    """Dump winners sorted ``(voxel_idx, orient_idx)`` for diff vs. C.

    Output format matches the C ``screen_cpu.csv``::

        voxelIdx,orientIdx,fracOverlap
        0,12,0.842311
        ...
    """
    sorted_winners = sorted(
        result.winners,
        key=lambda w: (w.voxel_idx, w.orient_idx),
    )
    with open(path, "w") as f:
        f.write("voxelIdx,orientIdx,fracOverlap\n")
        for w in sorted_winners:
            f.write(f"{w.voxel_idx},{w.orient_idx},{w.frac_overlap:.6f}\n")
