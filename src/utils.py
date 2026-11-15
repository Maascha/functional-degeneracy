"""Hessian-vector products and the Lanczos eigensolver."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from train import task_mse


def make_hvp_operator(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return v -> H v for the Hessian of the MSE task loss at the current parameters."""
    params = [p for p in model.parameters() if p.requires_grad]

    def hvp(v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 1:
            raise ValueError("HVP input must be a 1D vector")

        pieces: list[torch.Tensor] = []
        idx = 0
        for p in params:
            size = p.numel()
            pieces.append(v[idx: idx + size].view_as(p))
            idx += size

        if idx != v.numel():
            raise ValueError("HVP input length does not match parameter dimension")

        loss = task_mse(model, x, y)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        gv = torch.zeros((), device=v.device, dtype=v.dtype)
        for g, vp in zip(grads, pieces):
            gv = gv + torch.sum(g * vp)

        hv_parts = torch.autograd.grad(gv, params, create_graph=False, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hv_parts]).detach()

    return hvp


def build_dense_hessian_from_hvp(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dense symmetric matrix from dim matrix-vector products H e_j."""
    h_cols: list[torch.Tensor] = []
    eye = torch.eye(dim, device=device, dtype=dtype)
    for j in range(dim):
        h_cols.append(hvp(eye[:, j]))

    h = torch.stack(h_cols, dim=1)
    # Symmetrize to reduce tiny autograd numerical asymmetries.
    h = 0.5 * (h + h.T)
    return h.detach()


def lanczos_top_eigenpairs(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    top_m: int,
    num_iter: int | None = None,
    tol: float = 1e-10,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leading eigenpairs of a symmetric operator by Lanczos with full re-orthogonalization.

    The start vector is drawn from the global torch RNG. Returns at most top_m
    eigenvalues (descending) and unit eigenvectors as columns.
    """
    if top_m < 1:
        raise ValueError("top_m must be >= 1")

    if device is None:
        device = torch.device("cpu")

    if num_iter is None:
        num_iter = min(dim, max(4 * top_m, 30))
    num_iter = min(num_iter, dim)

    q = torch.randn(dim, device=device, dtype=dtype)
    q = q / (q.norm() + 1e-12)

    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), device=device, dtype=dtype)

    q_basis: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    # Breakdown test relative to the operator norm, estimated on the fly from the
    # Lanczos coefficients, so that the number of returned eigenpairs does not
    # depend on the output scale.
    scale_est = 0.0

    for it in range(num_iter):
        z = matvec(q)
        if it > 0:
            z = z - beta_prev * q_prev

        alpha = torch.dot(q, z)
        z = z - alpha * q

        for qi in q_basis:
            z = z - torch.dot(qi, z) * qi

        beta = torch.norm(z)

        q_basis.append(q)
        alphas.append(alpha)

        scale_est = max(scale_est, abs(float(alpha.item())), float(beta.item()))
        if float(beta.item()) < tol * max(scale_est, 1e-300):
            break

        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    m_eff = len(alphas)
    q_mat = torch.stack(q_basis, dim=1)

    t_mat = torch.zeros((m_eff, m_eff), device=device, dtype=dtype)
    for i in range(m_eff):
        t_mat[i, i] = alphas[i]
        if i < m_eff - 1:
            t_mat[i, i + 1] = betas[i]
            t_mat[i + 1, i] = betas[i]

    evals, evecs_t = torch.linalg.eigh(t_mat)
    order = torch.argsort(evals, descending=True)
    order = order[: min(top_m, m_eff)]

    evals_top = evals[order]
    evecs_top = q_mat @ evecs_t[:, order]
    evecs_top = evecs_top / (torch.norm(evecs_top, dim=0, keepdim=True) + 1e-12)

    return evals_top.detach(), evecs_top.detach()
