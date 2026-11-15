"""Behavioral Hessian eigenpairs and projection curves.

The behavioral Hessian at theta_star on a probe set P is H_beh = |P|^-1 J_P^T J_P.
Its leading eigenpairs are computed through the dual operator |P|^-1 J_P J_P^T in
probe-output space (dimension D = |P| * m), applied matrix-free with one
vector-Jacobian and one Jacobian-vector product, and diagonalized by symmetric
Lanczos with full orthogonalization. Parameter-space eigenvectors are recovered
lazily as v_j = J_P^T u_j / ||J_P^T u_j|| and re-orthonormalized by modified
Gram-Schmidt before they are accumulated into the projection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from utils import lanczos_top_eigenpairs


def _format_metric_value(value: float, fmt: str | None) -> str:
    if not fmt:
        return f"{value:.6e}"
    return format(value, fmt)


def _resolve_projection_target(
    theta_0: torch.Tensor,
    theta_star: torch.Tensor,
    projection_vector_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    """Vector to project and the offset it is added back to.

    'displacement': project theta_star - theta_0 and add theta_0 back (theta_ref = theta_0).
    'absolute':     project theta_star itself (theta_ref = 0).
    """
    if projection_vector_mode == "displacement":
        projection_target = theta_star - theta_0
        theta_offset = theta_0
    elif projection_vector_mode == "absolute":
        projection_target = theta_star
        theta_offset = None
    else:
        raise ValueError(
            f"Unsupported projection_vector_mode '{projection_vector_mode}'. Use one of: displacement, absolute"
        )
    projection_target_norm_sq = torch.sum(projection_target**2).item()
    return projection_target, theta_offset, projection_target_norm_sq


def make_beh_hvp_operator(model: nn.Module, x_probe: torch.Tensor, logits_star: torch.Tensor):
    """Hessian-vector product of the behavioral loss

        L_beh(theta) = (1 / (2|P|)) * sum_{x in P} || f_theta(x) - f_theta_star(x) ||^2,

    used to check the recovered eigenpairs in parameter space.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    def hvp(v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 1:
            raise ValueError("HVP input must be a 1D vector")

        pieces: list[torch.Tensor] = []
        idx = 0
        for p in params:
            size = p.numel()
            pieces.append(v[idx : idx + size].view_as(p))
            idx += size

        if idx != v.numel():
            raise ValueError("HVP input length does not match parameter dimension")

        logits = model(x_probe)
        residuals = logits - logits_star
        loss = torch.sum(residuals**2) / (2.0 * x_probe.shape[0])

        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        gv = torch.zeros((), device=v.device, dtype=v.dtype)
        for g, vp in zip(grads, pieces):
            gv = gv + torch.sum(g * vp)

        hv_parts = torch.autograd.grad(gv, params, create_graph=False, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hv_parts]).detach()

    return hvp


def compute_behavioral_eigenspace_dual(
    model: nn.Module,
    x_probe: torch.Tensor,
    top_m: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_path: Path | None = None,
    min_positive_eval: float = 1e-8,
    min_relative_eval: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Leading eigenpairs of the dual behavioral operator |P|^-1 J_P J_P^T.

    top_m <= 0 requests the full dual spectrum. Lanczos runs for
    min(D, max(4 * top_m, 60)) steps. Eigenpairs are retained if
    lambda_j > min_relative_eval * lambda_1; min_positive_eval is used instead
    only if lambda_1 itself is numerically zero. With cache_path, the
    eigenpairs are read from / written to that file (see
    compute_top_eigenpairs_with_auto_cache).

    Returns (dual eigenvalues, dual eigenvectors as columns, recovery state for
    recover_behavioral_param_eigenvector_lazy).
    """
    with torch.no_grad():
        logits_ref = model(x_probe)
    batch_size = x_probe.shape[0]
    output_dim = logits_ref.shape[1]
    dual_dim = batch_size * output_dim

    if top_m <= 0:
        top_m_dual = dual_dim
    else:
        top_m_dual = min(top_m, dual_dim)

    num_iter_dual = min(dual_dim, max(4 * top_m_dual, 60))
    print(
        f"  Running matrix-free dual Lanczos in probe-output space "
        f"(dual_dim={dual_dim}, top_m={top_m_dual}, ~{num_iter_dual} iterations)..."
    )

    params_dict = {
        name: p.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    buffers_dict = {
        name: b.detach().clone().to(device=device, dtype=dtype)
        for name, b in model.named_buffers()
    }

    def logits_fn(pdict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.func.functional_call(model, (pdict, buffers_dict), (x_probe,))

    def dual_matvec(u_flat: torch.Tensor) -> torch.Tensor:
        if u_flat.ndim != 1 or u_flat.numel() != dual_dim:
            raise ValueError(f"Dual matvec expects shape ({dual_dim},), got {tuple(u_flat.shape)}")
        u = u_flat.view(batch_size, output_dim)

        _, vjp_fn = torch.func.vjp(logits_fn, params_dict)
        jt_u = vjp_fn(u)[0]
        _, jjtu = torch.func.jvp(logits_fn, (params_dict,), (jt_u,))

        return (jjtu.reshape(-1) / float(batch_size)).detach()

    if cache_path is None:
        evals_dual, evecs_dual = lanczos_top_eigenpairs(
            matvec=dual_matvec,
            dim=dual_dim,
            top_m=top_m_dual,
            num_iter=num_iter_dual,
            device=device,
            dtype=dtype,
        )
    else:
        evals_dual, evecs_dual = compute_top_eigenpairs_with_auto_cache(
            matvec=dual_matvec,
            dim=dual_dim,
            target_m=top_m_dual,
            cache_path=cache_path,
            device=device,
            dtype=dtype,
            operator_name="behavioral_dual_top",
            num_iter_hint=num_iter_dual,
        )

    # Scale-invariant cutoff: an absolute threshold would depend on the output
    # scale, which varies across widths. The two floors must not be combined
    # with max(), which would make the cutoff absolute again for small lambda_1.
    lam1 = float(evals_dual.max().item()) if evals_dual.numel() > 0 else 0.0
    if float(min_relative_eval) > 0.0 and lam1 > float(min_positive_eval):
        threshold = float(min_relative_eval) * lam1
        binding = "relative"
    else:
        threshold = float(min_positive_eval)
        binding = "absolute"
    keep_mask = evals_dual > threshold
    kept = int(torch.count_nonzero(keep_mask).item())
    if kept < evals_dual.numel():
        print(
            f"  Filtering behavioral dual eigenpairs by eval>{threshold:.3e} "
            f"({binding} cutoff; lambda_1={lam1:.3e}): kept={kept}, dropped={int(evals_dual.numel() - kept)}"
        )
    if kept == 0:
        # Keep at least the top mode so that downstream projections are not empty.
        keep_mask = torch.zeros_like(evals_dual, dtype=torch.bool)
        keep_mask[0] = True
    evals_dual = evals_dual[keep_mask]
    evecs_dual = evecs_dual[:, keep_mask]

    param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    recovery_state: dict[str, object] = {
        "params_dict": params_dict,
        "buffers_dict": buffers_dict,
        "logits_fn": logits_fn,
        "batch_size": int(batch_size),
        "output_dim": int(output_dim),
        "param_names": param_names,
    }

    print(f"  Behavioral dual eigenspectrum computed: {len(evals_dual)} eigenpairs")

    return evals_dual.to(device=device, dtype=dtype), evecs_dual.to(device=device, dtype=dtype), recovery_state


def recover_behavioral_param_eigenvector_lazy(
    dual_evec_flat: torch.Tensor,
    dual_eval: torch.Tensor,
    recovery_state: dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Recover one parameter-space eigenvector v = J^T u / ||J^T u||."""
    batch_size = int(recovery_state["batch_size"])
    output_dim = int(recovery_state["output_dim"])
    param_names = recovery_state["param_names"]
    params_dict = recovery_state["params_dict"]
    logits_fn = recovery_state["logits_fn"]

    u = dual_evec_flat.to(device=device, dtype=dtype).view(batch_size, output_dim)
    lam = dual_eval.to(device=device, dtype=dtype)

    _, vjp_fn = torch.func.vjp(logits_fn, params_dict)
    jt_u = vjp_fn(u)[0]
    v_flat = torch.cat([jt_u[name].reshape(-1) for name in param_names])

    eps = torch.tensor(1e-12, device=device, dtype=dtype)
    scale = torch.sqrt(torch.clamp(lam * float(batch_size), min=eps))
    v_flat = v_flat / scale
    v_flat = v_flat / (v_flat.norm() + 1e-12)
    return v_flat.detach()


def _mgs_orthonormalize_against(
    v: torch.Tensor,
    basis: list[torch.Tensor],
    passes: int = 2,
) -> tuple[torch.Tensor, float]:
    """Modified Gram-Schmidt (two passes) of a unit vector v against an orthonormal basis.

    Returns (v_orth, retention) with retention = ||v_after|| / ||v_before||. A
    small retention means v adds no new direction to the span.
    """
    v_work = v
    for _ in range(max(1, passes)):
        for b in basis:
            v_work = v_work - torch.dot(b, v_work) * b
    norm_after = float(v_work.norm().item())
    norm_before = float(v.norm().item())
    retention = norm_after / norm_before if norm_before > 0 else 0.0
    if norm_after > 0:
        v_work = v_work / norm_after
    return v_work.detach(), retention


def project_and_evaluate_behavioral_eigenspace_lazy(
    evals_dual: torch.Tensor,
    evecs_dual: torch.Tensor,
    recovery_state: dict[str, object],
    theta_0: torch.Tensor,
    theta_star: torch.Tensor,
    k_values: list[int],
    model: nn.Module,
    layout: dict,
    device: torch.device,
    dtype: torch.dtype,
    set_params_from_flat_fn: Callable[[nn.Module, torch.Tensor, dict], None],
    metric_fns: dict[str, Callable[[nn.Module], float]],
    metric_formats: dict[str, str] | None = None,
    short_name: str = "behavioral",
    projection_vector_mode: str = "displacement",
    dependence_tol: float = 1e-3,
    max_basis_mb: int = 1024,
    diagnostics_out: dict[str, object] | None = None,
) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """Projection curves: evaluate theta_k = theta_ref + Pi_k (theta_star - theta_ref) for k = 1..m.

    Eigenvectors are recovered one at a time and re-orthonormalized by MGS
    against the retained basis, so the running sum is the orthogonal projection
    Pi_k onto the span of the top-k eigenvectors. The curve stops at the first
    vector whose MGS retention falls below dependence_tol. The retained basis
    is capped at max_basis_mb megabytes; beyond that cap, vectors are
    accumulated without re-orthonormalization and a warning is printed.

    Returns ({metric name: {k: value}}, {k: ||Pi_k target||^2 / ||target||^2}).
    k_values only controls which k are printed.
    """
    projection_target, theta_offset, projection_target_norm_sq = _resolve_projection_target(
        theta_0=theta_0,
        theta_star=theta_star,
        projection_vector_mode=projection_vector_mode,
    )
    delta_running = torch.zeros_like(projection_target)
    k_set = set(k_values)

    metric_values_by_k: dict[str, dict[int, float]] = {name: {} for name in metric_fns}
    delta_energy_frac_by_k: dict[int, float] = {}

    m = len(evals_dual)
    print(
        f"[5/5] Projecting onto top-k {short_name.upper()} eigenspace "
        f"with projection_vector_mode={projection_vector_mode} "
        f"(evaluating k=1..{m}, printing at k={k_values})..."
    )

    t_proj_start = time.perf_counter()

    param_dim = int(projection_target.numel())
    bytes_per_vec = max(1, param_dim * projection_target.element_size())
    max_basis_vectors = max(1, (max(1, int(max_basis_mb)) * 1024 * 1024) // bytes_per_vec)

    basis: list[torch.Tensor] = []
    min_retention = 1.0
    exhausted_at_k: int | None = None
    budget_exceeded_at_k: int | None = None
    k_evaluated = 0

    print(
        f"  reorthogonalization=MGS (2 passes), dependence_tol={dependence_tol:.1e}, "
        f"basis budget={max_basis_vectors} vectors ({max_basis_mb}MB at p={param_dim})"
    )

    for j in range(m):
        v_j = recover_behavioral_param_eigenvector_lazy(
            dual_evec_flat=evecs_dual[:, j],
            dual_eval=evals_dual[j],
            recovery_state=recovery_state,
            device=device,
            dtype=dtype,
        )
        k = j + 1

        if len(basis) >= max_basis_vectors:
            if budget_exceeded_at_k is None:
                budget_exceeded_at_k = k
                print(
                    f"  WARNING: basis memory budget reached at k={k}; "
                    f"continuing WITHOUT reorthogonalization beyond this point"
                )
        else:
            v_j, retention = _mgs_orthonormalize_against(v_j, basis)
            min_retention = min(min_retention, retention)
            if retention < dependence_tol:
                exhausted_at_k = k
                print(
                    f"  Recovered subspace numerically exhausted at k={k} "
                    f"(retention={retention:.2e} < {dependence_tol:.1e}); stopping projection."
                )
                del v_j
                break
            basis.append(v_j)

        coeff_j = torch.dot(v_j, projection_target)
        delta_running = delta_running + coeff_j * v_j

        delta_norm_sq_k = torch.sum(delta_running**2).item()
        delta_energy_frac = (
            delta_norm_sq_k / projection_target_norm_sq if projection_target_norm_sq > 0 else 0.0
        )
        delta_energy_frac_by_k[k] = delta_energy_frac

        theta_k = delta_running if theta_offset is None else theta_offset + delta_running
        set_params_from_flat_fn(model, theta_k, layout)
        model.eval()

        for metric_name, metric_fn in metric_fns.items():
            metric_values_by_k[metric_name][k] = float(metric_fn(model))

        k_evaluated = k

        if k in k_set:
            metric_line_parts: list[str] = []
            for metric_name in metric_fns:
                fmt = metric_formats.get(metric_name) if metric_formats is not None else None
                metric_line_parts.append(
                    f"{metric_name}={_format_metric_value(metric_values_by_k[metric_name][k], fmt)}"
                )
            print(f"  k={k}: DeltaEnergy={delta_energy_frac:.6f}, {', '.join(metric_line_parts)}")

        del v_j

    t_proj_sec = time.perf_counter() - t_proj_start
    print(
        f"  lazy behavioral projection done in {t_proj_sec:.1f}s "
        f"(evaluated k=1..{k_evaluated} of {m} available)"
    )
    print(f"  min MGS retention over retained basis: {min_retention:.3e}\n")

    if diagnostics_out is not None:
        diagnostics_out.update(
            {
                "reorthogonalize": "mgs",
                "dependence_tol": float(dependence_tol),
                "min_mgs_retention": float(min_retention),
                "numerically_exhausted_at_k": exhausted_at_k,
                "basis_budget_exceeded_at_k": budget_exceeded_at_k,
                "k_evaluated_max": int(k_evaluated),
                "eigenpairs_available": int(m),
                "param_dim": param_dim,
            }
        )

    return metric_values_by_k, delta_energy_frac_by_k


def build_k_grid(m: int) -> list[int]:
    """Powers of two up to m, plus m itself."""
    if m < 1:
        return []

    k_values: list[int] = []
    k = 1
    while k <= m:
        k_values.append(k)
        k *= 2

    if k_values[-1] != m:
        k_values.append(m)

    return k_values


def _project_orthogonal(v: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    if basis is None or basis.numel() == 0:
        return v
    return v - basis @ (basis.T @ v)


def _orthonormalize_columns(
    vecs: torch.Tensor,
    basis: torch.Tensor | None = None,
    tol: float = 1e-10,
) -> torch.Tensor:
    if vecs.ndim != 2:
        raise ValueError(f"Expected 2D tensor for vecs, got shape={tuple(vecs.shape)}")

    cols: list[torch.Tensor] = []
    for j in range(vecs.shape[1]):
        v = vecs[:, j]
        if basis is not None and basis.numel() > 0:
            v = _project_orthogonal(v, basis)
        for q in cols:
            v = v - torch.dot(q, v) * q
        nrm = torch.norm(v)
        if float(nrm.item()) > tol:
            cols.append(v / (nrm + 1e-12))

    if not cols:
        return torch.zeros((vecs.shape[0], 0), device=vecs.device, dtype=vecs.dtype)
    return torch.stack(cols, dim=1)


def _load_eigenpair_cache(
    cache_path: Path,
    expected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not cache_path.exists():
        return None

    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"  Warning: failed to load cache at {cache_path} ({exc}); ignoring cache")
        return None

    evals = payload.get("evals") if isinstance(payload, dict) else None
    evecs = payload.get("evecs") if isinstance(payload, dict) else None
    if not isinstance(evals, torch.Tensor) or not isinstance(evecs, torch.Tensor) or evals.ndim != 1 or evecs.ndim != 2:
        print(f"  Warning: malformed cache at {cache_path}; ignoring cache")
        return None
    if evecs.shape[0] != expected_dim:
        print(
            f"  Warning: cache dim mismatch at {cache_path} "
            f"(cache_dim={evecs.shape[0]}, expected_dim={expected_dim}); ignoring cache"
        )
        return None

    m = min(evals.numel(), evecs.shape[1])
    if m < 1:
        return None

    return evals[:m].contiguous(), evecs[:, :m].contiguous()


def _save_eigenpair_cache(
    cache_path: Path,
    evals: torch.Tensor,
    evecs: torch.Tensor,
    operator_name: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "operator": operator_name,
        "dim": int(evecs.shape[0]),
        "num_eigenpairs": int(evals.numel()),
        "evals": evals.detach().cpu().contiguous(),
        "evecs": evecs.detach().cpu().contiguous(),
    }
    torch.save(payload, cache_path)


def compute_top_eigenpairs_with_auto_cache(
    matvec,
    dim: int,
    target_m: int,
    cache_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    operator_name: str,
    num_iter_hint: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top eigenpairs of matvec, reusing and extending a cache file.

    If the cache holds at least target_m pairs, they are returned directly.
    Without a cache, a fresh Lanczos run computes them. If the cache holds
    fewer pairs (e.g. an earlier run stopped at a Lanczos breakdown), the cached
    vectors are re-orthonormalized and the missing pairs are computed by
    Lanczos on the operator deflated against them. The result is written back
    to the cache, so later runs on the same configuration reuse it.
    """
    if target_m < 1:
        raise ValueError("target_m must be >= 1")

    target_m = min(target_m, dim)
    cached = _load_eigenpair_cache(cache_path, expected_dim=dim)

    evals_cached: torch.Tensor | None = None
    evecs_cached: torch.Tensor | None = None
    cached_m = 0
    if cached is not None:
        evals_cached = cached[0].to(device=device, dtype=dtype)
        evecs_cached = cached[1].to(device=device, dtype=dtype)
        cached_m = int(evals_cached.numel())
        print(f"  Found cached {operator_name} eigenspace with m={cached_m} at {cache_path}")
        if cached_m >= target_m:
            print(f"  Reusing cached {operator_name} eigenpairs (target m={target_m}, no new Lanczos needed)")
            return evals_cached[:target_m], evecs_cached[:, :target_m]

    need_m = target_m - cached_m

    if cached_m == 0:
        print(f"  No reusable cache for {operator_name}; running fresh Lanczos for m={target_m}")
        num_iter = num_iter_hint if num_iter_hint is not None else min(dim, max(4 * target_m, 60))
        evals_all, evecs_all = lanczos_top_eigenpairs(
            matvec=matvec,
            dim=dim,
            top_m=target_m,
            num_iter=num_iter,
            device=device,
            dtype=dtype,
        )
    else:
        print(f"  Extending {operator_name} eigenspace from m={cached_m} to m={target_m} (need +{need_m})")

        q_cached = _orthonormalize_columns(evecs_cached)

        def matvec_deflated(v: torch.Tensor) -> torch.Tensor:
            v_orth = _project_orthogonal(v, q_cached)
            av_orth = _project_orthogonal(matvec(v_orth), q_cached)
            return av_orth

        num_iter_extra = min(dim, max(4 * need_m, 60))
        _, evecs_extra = lanczos_top_eigenpairs(
            matvec=matvec_deflated,
            dim=dim,
            top_m=need_m,
            num_iter=num_iter_extra,
            device=device,
            dtype=dtype,
        )
        evecs_extra = _orthonormalize_columns(evecs_extra, basis=q_cached)

        if evecs_extra.shape[1] < need_m:
            print(
                f"  Warning: requested {need_m} extra vectors for {operator_name}, "
                f"recovered {evecs_extra.shape[1]} after orthogonalization"
            )

        if evecs_extra.shape[1] == 0:
            evals_all = evals_cached
            evecs_all = q_cached
        else:
            evals_extra = torch.stack([torch.dot(vj, matvec(vj)) for vj in evecs_extra.T])
            evals_all = torch.cat([evals_cached, evals_extra], dim=0)
            evecs_all = torch.cat([q_cached, evecs_extra], dim=1)

        order = torch.argsort(evals_all, descending=True)
        evals_all = evals_all[order]
        evecs_all = evecs_all[:, order]

        if evals_all.numel() > target_m:
            evals_all = evals_all[:target_m]
            evecs_all = evecs_all[:, :target_m]

    _save_eigenpair_cache(
        cache_path=cache_path,
        evals=evals_all,
        evecs=evecs_all,
        operator_name=operator_name,
    )
    print(f"  Updated {operator_name} cache at {cache_path} (m={evals_all.numel()})")

    return evals_all, evecs_all


def validate_behavioral_eigenspace_lazy(
    evals_dual: torch.Tensor,
    evecs_dual: torch.Tensor,
    recovery_state: dict[str, object],
    hvp_operator,
    device: torch.device,
    dtype: torch.dtype,
    max_pairs: int = 5,
) -> dict[str, object]:
    """Check the leading recovered eigenpairs in parameter space.

    Reports ||H v - lambda v|| / ||v|| for the first max_pairs pairs and the
    largest deviation of their Gram matrix from the identity.
    """
    m = int(evals_dual.numel())
    n_check = min(max_pairs, m)

    residuals: list[dict[str, float | int]] = []
    checked_vectors: list[torch.Tensor] = []

    for i in range(n_check):
        v = recover_behavioral_param_eigenvector_lazy(
            dual_evec_flat=evecs_dual[:, i],
            dual_eval=evals_dual[i],
            recovery_state=recovery_state,
            device=device,
            dtype=dtype,
        )
        hv = hvp_operator(v)
        lam = evals_dual[i]
        resid = torch.norm(hv - lam * v) / (torch.norm(v) + 1e-12)
        residuals.append(
            {
                "pair_index": int(i + 1),
                "eigenvalue": float(lam.item()),
                "relative_residual": float(resid.item()),
            }
        )
        checked_vectors.append(v.detach().cpu())

    if checked_vectors:
        v_mat = torch.stack(checked_vectors, dim=1)
        gram = v_mat.T @ v_mat
        eye = torch.eye(n_check, dtype=gram.dtype)
        max_abs_ortho_dev = float(torch.max(torch.abs(gram - eye)).item())
    else:
        max_abs_ortho_dev = 0.0

    return {
        "num_eigenpairs": int(m),
        "num_residuals_checked": int(n_check),
        "residuals": residuals,
        "max_abs_orthogonality_deviation": float(max_abs_ortho_dev),
        "orthogonality_scope": f"first_{n_check}_recovered_pairs",
    }
