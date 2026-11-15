"""Behavioral recovery rank during training.

Takes a configuration directory written by run_toy_teacher_student.py, whose
checkpoint holds parameter snapshots theta_t taken during training. For every
snapshot, the behavioral eigenpairs at theta_t are computed on the stored probe
set, theta_t is projected onto the top-k eigendirections for every k, and the
recovery ranks are read off the projection curves as in the final-model
analysis (running best over k):

    tau rank: first k with metric(k) <= tau
    q rank:   first k recovering a fraction q of the gap L_ref -> L_star,
              with L_ref the metric at theta_0 and L_star the metric at theta_t.

The behavioral target at each snapshot is the snapshot's own output on the
probe set, so the rank measures how many eigendirections reconstruct the model
at step t.

Output: <run_dir>/temporal/recovery_ranks_long.csv  one row per (step, threshold, metric)
        <run_dir>/temporal/results.json             eigenvalues and projection curves per step
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from curvature_projection import (
    build_k_grid,
    compute_behavioral_eigenspace_dual,
    project_and_evaluate_behavioral_eigenspace_lazy,
)
from run_toy_teacher_student import StudentSingleHidden
from train import get_param_layout, set_params_from_flat


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Vector projected onto the eigendirections: 'absolute' (theta_ref = 0) or 'displacement' (theta_t - theta_0). L_ref is the metric at
# theta_0 in both cases.
PROJECTION_VECTOR_MODE = "absolute"

# Absolute MSE thresholds tau and relative recovery thresholds q.
TAU_THRESHOLDS = [0.01, 0.05, 0.075]
Q_THRESHOLDS = [0.8, 0.9, 0.95]

# Lanczos eigenpairs per snapshot; -1 uses min(p, |P| * output_dim), the rank cap
# of the behavioral operator.
TOP_M = -1

DEVICE = "cpu"
OUT_SUBDIR = "temporal"


# ---------------------------------------------------------------------------
# Recovery ranks
# ---------------------------------------------------------------------------

def _running_best(curve: dict[int, float]) -> tuple[list[int], dict[int, float]]:
    """Running minimum of a metric-vs-k curve (the raw curves are not monotone in k)."""
    sorted_ks = sorted(curve)
    best: dict[int, float] = {}
    running = float("inf")
    for k in sorted_ks:
        running = min(running, curve[k])
        best[k] = running
    return sorted_ks, best


def _rank_at_or_below(
    sorted_ks: list[int], best: dict[int, float], level: float
) -> tuple[int | None, bool]:
    """First k whose running-best value is <= level (absolute threshold tau)."""
    for k in sorted_ks:
        if best[k] <= level:
            return k, True
    return None, False


def _rank_recovered_fraction(
    sorted_ks: list[int], best: dict[int, float], l_ref: float, l_star: float, q: float
) -> tuple[int | None, bool]:
    """First k with (L_ref - L_k) / (L_ref - L_star) >= q.

    A non-positive denominator (the snapshot is no better than theta_0) leaves
    the fraction undefined, and the rank is reported as not reached.
    """
    denom = l_ref - l_star
    if not (denom > 0.0):
        return None, False
    for k in sorted_ks:
        if (l_ref - best[k]) / denom >= q:
            return k, True
    return None, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral recovery rank at the training snapshots of a teacher-student run.",
    )
    parser.add_argument("--run-dir", required=True,
                        help="Configuration directory containing model/checkpoint.pt.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / "model" / "checkpoint.pt"
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(DEVICE)
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    snapshots: list[dict] = ckpt.get("snapshots", [])
    if not snapshots:
        sys.exit("No snapshots found in checkpoint; train with --num-snapshots > 0.")
    student_arch: dict = ckpt["student_arch"]

    theta_0 = ckpt["theta_0"].to(device)
    dtype = theta_0.dtype
    x_probe = ckpt["x_probe"].to(device=device, dtype=dtype)
    x_train = ckpt["x_train"].to(device=device, dtype=dtype)
    y_train = ckpt["y_train"].to(device=device, dtype=dtype)
    y_train_teacher = ckpt["y_train_teacher"].to(device=device, dtype=dtype)
    x_test = ckpt["x_test"].to(device=device, dtype=dtype)
    y_test = ckpt["y_test"].to(device=device, dtype=dtype)

    model = StudentSingleHidden(
        student_arch["input_dim"], student_arch["output_dim"], student_arch["hidden_dim"],
        student_arch["student_activation"], bool(student_arch["student_bias"]),
    ).to(device=device, dtype=dtype)
    layout = get_param_layout(model)

    dual_dim = x_probe.shape[0] * student_arch["output_dim"]
    top_m = min(int(theta_0.numel()), dual_dim) if TOP_M <= 0 else min(TOP_M, dual_dim)
    q_thresholds = sorted(Q_THRESHOLDS)
    tau_thresholds = sorted(TAU_THRESHOLDS)
    # The Lanczos start vector of snapshot t is seeded with seed + t, so a
    # snapshot's eigensolve does not depend on the snapshots before it.
    seed_base = int(ckpt.get("seed", 0))

    print(f"  Snapshots: {len(snapshots)}, probe: {tuple(x_probe.shape)}, top_m: {top_m}")
    print(f"  q={q_thresholds}, tau={tau_thresholds}, projection_vector_mode={PROJECTION_VECTOR_MODE}")
    print(f"  Output: {out_dir}\n")

    long_rows: list[dict] = []
    all_results: list[dict] = []

    for i, snap in enumerate(snapshots):
        step: int = snap["step"]
        theta_t: torch.Tensor = snap["theta"].to(device=device, dtype=dtype)

        t_start = time.perf_counter()
        set_params_from_flat(model, theta_t, layout)
        model.eval()
        with torch.no_grad():
            behavioral_target = model(x_probe).detach()

        torch.manual_seed(seed_base + step)
        evals, evecs, recovery_state = compute_behavioral_eigenspace_dual(
            model=model,
            x_probe=x_probe,
            top_m=top_m,
            device=device,
            dtype=theta_t.dtype,
            cache_path=None,
        )

        def _beh_mse(m: nn.Module, _t: torch.Tensor = behavioral_target) -> float:
            with torch.no_grad():
                return float(F.mse_loss(m(x_probe), _t).item())

        def _train_noisy_mse(m: nn.Module) -> float:
            with torch.no_grad():
                return float(F.mse_loss(m(x_train), y_train).item())

        def _train_teacher_mse(m: nn.Module) -> float:
            with torch.no_grad():
                return float(F.mse_loss(m(x_train), y_train_teacher).item())

        def _test_mse(m: nn.Module) -> float:
            with torch.no_grad():
                return float(F.mse_loss(m(x_test), y_test).item())

        metric_fns = {
            "beh_mse": _beh_mse,
            "train_noisy_mse": _train_noisy_mse,
            "train_teacher_mse": _train_teacher_mse,
            "test_mse": _test_mse,
        }

        # L_star at theta_t and L_ref at theta_0 for every metric.
        loss_star = {name: fn(model) for name, fn in metric_fns.items()}
        set_params_from_flat(model, theta_0, layout)
        loss_ref = {name: fn(model) for name, fn in metric_fns.items()}
        set_params_from_flat(model, theta_t, layout)

        metrics_by_k, _ = project_and_evaluate_behavioral_eigenspace_lazy(
            evals_dual=evals,
            evecs_dual=evecs,
            recovery_state=recovery_state,
            theta_0=theta_0,
            theta_star=theta_t,
            k_values=build_k_grid(len(evals)),
            model=model,
            layout=layout,
            device=device,
            dtype=theta_t.dtype,
            set_params_from_flat_fn=set_params_from_flat,
            metric_fns=metric_fns,
            projection_vector_mode=PROJECTION_VECTOR_MODE,
        )
        set_params_from_flat(model, theta_t, layout)

        losses = {key: snap[key] for key in ("loss_train_noisy", "loss_train_teacher", "loss_test")}
        for metric_name, curve in metrics_by_k.items():
            m_sorted_ks, m_best = _running_best(curve)
            l_ref, l_star = loss_ref[metric_name], loss_star[metric_name]
            ranks = [("tau", tau, *_rank_at_or_below(m_sorted_ks, m_best, tau)) for tau in tau_thresholds]
            ranks += [("q", q, *_rank_recovered_fraction(m_sorted_ks, m_best, l_ref, l_star, q))
                      for q in q_thresholds]
            for kind, threshold, rank, reached in ranks:
                long_rows.append({
                    "step": step,
                    "kind": kind,
                    "threshold": threshold,
                    "metric": metric_name,
                    "rank": rank if rank is not None else "",
                    "reached": reached,
                    "n_evals": len(evals),
                    "k_max": m_sorted_ks[-1] if m_sorted_ks else 0,
                    "loss_ref": l_ref,
                    "loss_star": l_star,
                    **losses,
                })

        all_results.append({
            "step": step,
            "evals": evals.tolist(),
            "beh_mse_0": loss_ref["beh_mse"],
            "losses": losses,
            **{f"{name}_by_k": {str(k): v for k, v in curve.items()} for name, curve in metrics_by_k.items()},
        })

        print(
            f"  [{i+1}/{len(snapshots)}] step={step:5d}  evals={len(evals)}  "
            f"beh_mse_0={loss_ref['beh_mse']:.4e}  ({time.perf_counter() - t_start:.1f}s)"
        )

    json_path = out_dir / "results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "projection_vector_mode": PROJECTION_VECTOR_MODE,
                "lanczos_seed_base": seed_base,
                "top_m": top_m,
                "q_thresholds": q_thresholds,
                "tau_thresholds": tau_thresholds,
                "num_snapshots": len(snapshots),
                "steps_data": all_results,
            },
            f,
            indent=2,
        )
    print(f"Wrote full results to {json_path}")

    # Written last and atomically: scripts/run_teacher_student.sh treats this file as
    # the marker of a finished run.
    long_csv_path = out_dir / "recovery_ranks_long.csv"
    long_fieldnames = [
        "step", "kind", "threshold", "metric", "rank", "reached",
        "n_evals", "k_max", "loss_ref", "loss_star",
        "loss_train_noisy", "loss_train_teacher", "loss_test",
    ]
    long_csv_tmp = long_csv_path.with_suffix(".csv.partial")
    with long_csv_tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)
    long_csv_tmp.replace(long_csv_path)
    print(f"Wrote {len(long_rows)} rows to {long_csv_path}")


if __name__ == "__main__":
    main()
