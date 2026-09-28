"""GPU   PyTorch Levenberg-Marquardt  。

non-English text removed K non-English text removed，non-English text removed XYZ。
non-English text removed；non-English text removed autograd non-English text removed，
non-English text removed XYZ CD-L1 non-English text removed LM non-English text removed、non-English text removed。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TorchLMDiagnostics:
    iterations: int
    converged_fraction: float
    accepted_fraction: float
    mean_squared_range_residual: float
    max_squared_range_residual: float


def _as_batched_inputs(
    distances: torch.Tensor,
    anchors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Normalize ``[N,K]``/``[K,3]`` and batched inputs to one representation."""
    squeeze_batch = distances.ndim == 2
    if distances.ndim not in (2, 3):
        raise ValueError(
            "distances must have shape [N,K] or [B,N,K], got {}".format(
                tuple(distances.shape)
            )
        )
    if anchors.ndim not in (2, 3):
        raise ValueError(
            "anchors must have shape [K,3] or [B,K,3], got {}".format(
                tuple(anchors.shape)
            )
        )

    if distances.ndim == 2:
        distances = distances.unsqueeze(0)
    if anchors.ndim == 2:
        anchors = anchors.unsqueeze(0)

    if anchors.size(0) == 1 and distances.size(0) > 1:
        anchors = anchors.expand(distances.size(0), -1, -1)
    if distances.size(0) != anchors.size(0):
        raise ValueError(
            "batch mismatch between distances {} and anchors {}".format(
                tuple(distances.shape), tuple(anchors.shape)
            )
        )
    if distances.size(-1) != anchors.size(-2):
        raise ValueError(
            "anchor count mismatch between distances {} and anchors {}".format(
                tuple(distances.shape), tuple(anchors.shape)
            )
        )
    if anchors.size(-1) != 3:
        raise ValueError("anchors must contain XYZ coordinates in the last dimension")
    return distances, anchors, squeeze_batch


def _residual_and_jacobian(
    points: torch.Tensor,
    distances: torch.Tensor,
    anchors: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """  XYZ   Jacobian。

    non-English text removed n non-English text removed k non-English text removed：
        r_nk = ||x_n-a_k||_2 - d_nk
        J_nk = (x_n-a_k) / ||x_n-a_k||_2
    """
    difference = points.unsqueeze(-2) - anchors.unsqueeze(1)
    geometric_distance = torch.linalg.vector_norm(difference, dim=-1)
    safe_distance = geometric_distance.clamp_min(eps)
    residual = geometric_distance - distances
    jacobian = difference / safe_distance.unsqueeze(-1)
    return residual, jacobian


def find_points_from_distance_torch(
    distances: torch.Tensor,
    anchors: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    max_iterations: int = 50,
    initial_damping: float = 1e-3,
    damping_up: float = 10.0,
    damping_down: float = 0.3,
    min_damping: float = 1e-12,
    max_damping: float = 1e12,
    xtol: float = 1e-8,
    ftol: float = 1e-10,
    gtol: float = 1e-8,
    eps: float = 1e-12,
    solve_dtype: torch.dtype = torch.float64,
    target_converged_fraction: float = 1.0,
    initial_points: Optional[torch.Tensor] = None,
    return_diagnostics: bool = False,
):
    """  LM   XYZ。

    non-English text removed：

    ``sum_k w_k (||x-a_k||_2 - d_k)^2``。non-English text removed ``weights`` non-English text removed，
    non-English text removed8non-English text removed。

    non-English text removed/non-English text removed。non-English text removed16,384non-English text removed
    non-English text removed GPU non-English text removed，non-English text removed Python non-English text removed。

    Args:
        distances: Tensor shaped ``[N,K]`` or ``[B,N,K]``.
        anchors: Tensor shaped ``[K,3]`` or ``[B,K,3]``.
        weights: Optional non-negative tensor with the same shape as
            ``distances``.  These are fixed measurement precision weights;
            they remain differentiable in the truncated training solve.
        solve_dtype: ``torch.float64`` is the parity default because SciPy LM
            solves in double precision.
        target_converged_fraction: Stop the batch once this fraction of points
            has converged.  Full evaluation can use 0.999 so a few pathological
            points do not hold all 65,536 point solves at the iteration limit.
        initial_points: Optional zero-replacing initialization with shape
            ``[N,3]`` or ``[B,N,3]``.  The baseline uses the origin.
        return_diagnostics: Return ``(points, diagnostics)`` when true.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if initial_damping <= 0:
        raise ValueError("initial_damping must be positive")
    if not 0.0 < target_converged_fraction <= 1.0:
        raise ValueError("target_converged_fraction must be in (0, 1]")

    if solve_dtype == torch.float32:
        xtol = max(xtol, 1e-6)
        ftol = max(ftol, 1e-8)
        gtol = max(gtol, 1e-6)
        eps = max(eps, 1e-8)

    distances, anchors, squeeze_batch = _as_batched_inputs(distances, anchors)
    if weights is not None:
        if weights.ndim == 2:
            weights = weights.unsqueeze(0)
        if tuple(weights.shape) != tuple(distances.shape):
            raise ValueError(
                "weights must have shape {}, got {}".format(
                    tuple(distances.shape), tuple(weights.shape)
                )
            )
        if not torch.isfinite(weights).all():
            raise ValueError("weights must be finite")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative")

    output_dtype = distances.dtype
    device = distances.device
    distances = distances.to(device=device, dtype=solve_dtype)
    anchors = anchors.to(device=device, dtype=solve_dtype)
    if weights is None:
        weights = torch.ones_like(distances)
    else:
        weights = weights.to(device=device, dtype=solve_dtype)

    batch_size, point_count, _ = distances.shape
    if initial_points is None:
        points = torch.zeros(
            batch_size, point_count, 3, device=device, dtype=solve_dtype
        )
    else:
        if initial_points.ndim == 2:
            initial_points = initial_points.unsqueeze(0)
        if tuple(initial_points.shape) != (batch_size, point_count, 3):
            raise ValueError(
                "initial_points must have shape {}, got {}".format(
                    (batch_size, point_count, 3), tuple(initial_points.shape)
                )
            )
        points = initial_points.to(device=device, dtype=solve_dtype).clone()

    damping = torch.full(
        (batch_size, point_count),
        float(initial_damping),
        device=device,
        dtype=solve_dtype,
    )
    active = torch.ones(
        batch_size, point_count, device=device, dtype=torch.bool
    )
    ever_accepted = torch.zeros_like(active)
    identity = torch.eye(3, device=device, dtype=solve_dtype).view(1, 1, 3, 3)

    completed_iterations = 0
    for iteration in range(max_iterations):
        completed_iterations = iteration + 1
        residual, jacobian = _residual_and_jacobian(
            points, distances, anchors, eps
        )
        cost = 0.5 * (weights * residual.square()).sum(dim=-1)

        hessian = torch.einsum(
            "bnk,bnki,bnkj->bnij", weights, jacobian, jacobian
        )
        gradient = torch.einsum(
            "bnk,bnki,bnk->bni", weights, jacobian, residual
        )
        hessian_diagonal = torch.diagonal(hessian, dim1=-2, dim2=-1).clamp_min(eps)
        system = (
            hessian
            + damping.unsqueeze(-1).unsqueeze(-1)
            * torch.diag_embed(hessian_diagonal)
            + eps * identity
        )

        step = torch.linalg.solve(system, -gradient.unsqueeze(-1)).squeeze(-1)
        finite_step = torch.isfinite(step).all(dim=-1)
        candidate = points + step
        candidate_residual, _ = _residual_and_jacobian(
            candidate, distances, anchors, eps
        )
        candidate_cost = 0.5 * (
            weights * candidate_residual.square()
        ).sum(dim=-1)

        accepted = active & finite_step & torch.isfinite(candidate_cost) & (candidate_cost < cost)
        ever_accepted |= accepted
        points = torch.where(accepted.unsqueeze(-1), candidate, points)
        damping = torch.where(
            accepted,
            (damping * damping_down).clamp_min(min_damping),
            (damping * damping_up).clamp_max(max_damping),
        )

        step_norm = torch.linalg.vector_norm(step, dim=-1)
        point_norm = torch.linalg.vector_norm(points, dim=-1)
        cost_change = (cost - candidate_cost).abs()
        gradient_norm = gradient.abs().amax(dim=-1)
        converged = active & (
            (gradient_norm <= gtol)
            | (
                accepted
                & (
                    (step_norm <= xtol * (1.0 + point_norm))
                    | (cost_change <= ftol * (1.0 + cost))
                )
            )
        )
        active &= ~converged
        converged_fraction = (~active).float().mean()
        if float(converged_fraction.item()) >= target_converged_fraction:
            break

    final_residual, _ = _residual_and_jacobian(points, distances, anchors, eps)
    squared_residual = final_residual.square()
    diagnostics = TorchLMDiagnostics(
        iterations=completed_iterations,
        converged_fraction=float((~active).double().mean().item()),
        accepted_fraction=float(ever_accepted.double().mean().item()),
        mean_squared_range_residual=float(squared_residual.mean().item()),
        max_squared_range_residual=float(squared_residual.max().item()),
    )

    points = points.to(dtype=output_dtype)
    if squeeze_batch:
        points = points.squeeze(0)
    if return_diagnostics:
        return points, diagnostics
    return points
