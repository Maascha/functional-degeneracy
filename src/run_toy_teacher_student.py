"""Teacher-student regression with behavioral projection curves.

A frozen two-layer ReLU teacher d_in -> 10 -> d_out (with biases) generates
regression targets for Gaussian inputs. A single-hidden-layer ReLU student
d_in -> d -> d_out is trained with full-batch Adam on the MSE. At the trained
parameters theta_star the leading eigenpairs of the behavioral Hessian are
computed on a probe set (matrix-free dual Lanczos, see curvature_projection.py), theta_star is
projected onto the top-k eigendirections for every k, and the train, test and
behavioral MSE of each projected model are recorded.

Output layout under --out-dir:
    model/checkpoint.pt      trained student, theta_0, data, training snapshots
    model/eigen_cache/       behavioral eigenpairs per probe set (reused by later runs)
    <run-subdir-name>/       summary.json, experiment_outputs.pt,
                             eigenvalues_behavioral.txt, quick-look plots

Terminology:
    teacher outputs          noiseless teacher predictions (test targets)
    behavioral target        student outputs at theta_star on the probe set
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

# Required for deterministic CuBLAS kernels on CUDA when deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from curvature_projection import (
    build_k_grid,
    compute_behavioral_eigenspace_dual,
    make_beh_hvp_operator,
    project_and_evaluate_behavioral_eigenspace_lazy,
    validate_behavioral_eigenspace_lazy,
)
from plots import save_eigenvalues_plot, save_topk_metrics, save_training_curve
from train import flatten_params, get_param_layout, set_params_from_flat


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

class TwoLayerTeacher:
    """Frozen two-layer ReLU teacher input -> hidden -> output with zero biases."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        noise_std: float = 0.0,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Weights are drawn from the global RNG, so the teacher is fixed by set_seed.
        self.W1 = (torch.randn(input_dim, hidden_dim, device=device, dtype=dtype) / np.sqrt(max(1, input_dim))).detach()
        self.W2 = (torch.randn(hidden_dim, output_dim, device=device, dtype=dtype) / np.sqrt(max(1, hidden_dim))).detach()
        self.b1 = torch.zeros(hidden_dim, device=device, dtype=dtype)
        self.b2 = torch.zeros(output_dim, device=device, dtype=dtype)
        self.activation = "relu"
        self.noise_std = float(noise_std)
        self._input_dim = int(input_dim)
        self._hidden_dim = int(hidden_dim)
        self._output_dim = int(output_dim)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Noiseless teacher outputs."""
        return F.relu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    @torch.no_grad()
    def generate_labels(self, x: torch.Tensor) -> torch.Tensor:
        """Training labels: teacher outputs plus Gaussian noise of std noise_std."""
        y = self.__call__(x)
        if self.noise_std > 0.0:
            y = y + torch.randn_like(y) * self.noise_std
        return y

    def info(self) -> dict:
        return {
            "type": "two_layer",
            "input_dim": self._input_dim,
            "hidden_dim": self._hidden_dim,
            "output_dim": self._output_dim,
            "activation": self.activation,
            "bias": True,
            "noise_std": self.noise_std,
        }


def make_teacher(
    teacher_type: str = "two_layer",
    input_dim: int = 10,
    output_dim: int = 10,
    teacher_hidden_dim: int = 10,
    teacher_activation: str = "relu",
    teacher_bias: bool = True,
    noise_std: float = 0.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> TwoLayerTeacher:
    """Build the teacher. Only the two-layer ReLU teacher with biases exists.

    teacher_type, teacher_activation and teacher_bias are validated; they are
    accepted so that summary.json["teacher_info"] can be passed back in unchanged.
    """
    if teacher_type != "two_layer" or teacher_activation != "relu" or not teacher_bias:
        raise ValueError(
            "only the two-layer ReLU teacher with biases is supported, got "
            f"type={teacher_type!r}, activation={teacher_activation!r}, bias={teacher_bias!r}"
        )
    return TwoLayerTeacher(
        input_dim=input_dim,
        hidden_dim=teacher_hidden_dim,
        output_dim=output_dim,
        noise_std=noise_std,
        device=device,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------

class StudentSingleHidden(nn.Module):
    """Single-hidden-layer ReLU student input -> hidden -> output."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, activation: str = "relu", bias: bool = True) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(f"only the ReLU student is supported, got activation={activation!r}")
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def task_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(model(x), y)


def generate_synthetic_data(
    teacher: TwoLayerTeacher,
    train_size: int,
    test_size: int,
    probe_size: int,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gaussian inputs x ~ N(0, I) and teacher targets.

    Returns (x_train, y_train, x_test, y_test, x_probe, y_probe): noisy training
    labels, noiseless test targets, and an independent probe set with noiseless
    targets (--probe-source train/test use the train or test inputs instead).
    Each split has its own generator; the offsets are spaced so that the
    streams of different seeds never overlap. Label noise is drawn from the
    global RNG.
    """
    stream_stride = 1_000_003
    g_train = torch.Generator(device=device).manual_seed(seed * stream_stride + 0)
    x_train = torch.randn(train_size, input_dim, generator=g_train, dtype=dtype, device=device)
    y_train = teacher.generate_labels(x_train)

    g_test = torch.Generator(device=device).manual_seed(seed * stream_stride + 1)
    x_test = torch.randn(test_size, input_dim, generator=g_test, dtype=dtype, device=device)
    y_test = teacher(x_test)

    g_probe = torch.Generator(device=device).manual_seed(seed * stream_stride + 2)
    x_probe = torch.randn(probe_size, input_dim, generator=g_probe, dtype=dtype, device=device)
    y_probe = teacher(x_probe)

    return x_train, y_train, x_test, y_test, x_probe, y_probe


def make_regression_metric_fns(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    x_probe: torch.Tensor,
    behavioral_target_probe: torch.Tensor,
) -> tuple[dict[str, Callable[[nn.Module], float]], dict[str, str]]:
    """Metrics evaluated at every projected model.

    train_loss    train MSE against the (noisy) training labels
    test_metric   test MSE against the noiseless teacher outputs
    behavior_mse  MSE to the outputs of theta_star on the probe set
    """

    def train_loss_metric(model: nn.Module) -> float:
        with torch.no_grad():
            return float(F.mse_loss(model(x_train), y_train).item())

    def test_mse_metric(model: nn.Module) -> float:
        with torch.no_grad():
            return float(F.mse_loss(model(x_test), y_test).item())

    def behavior_mse_metric(model: nn.Module) -> float:
        with torch.no_grad():
            return float(torch.mean((model(x_probe) - behavioral_target_probe) ** 2).item())

    metric_fns: dict[str, Callable[[nn.Module], float]] = {
        "train_loss": train_loss_metric,
        "test_metric": test_mse_metric,
        "behavior_mse": behavior_mse_metric,
    }
    metric_formats = {name: ".6e" for name in metric_fns}
    return metric_fns, metric_formats


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    out_dir: str,
    run_subdir_name: str,
    seed: int = 0,
    input_dim: int = 10,
    output_dim: int = 10,
    teacher_hidden_dim: int = 10,
    hidden_dim: int = 10,
    noise_std: float = 0.0,
    train_size: int = 2048,
    test_size: int = 512,
    probe_size: int = 512,
    probe_source: str = "train",
    steps: int = 4000,
    lr: float = 1e-3,
    top_m: int = -1,
    projection_vector_mode: str = "absolute",
    num_snapshots: int = 40,
    dependence_tol: float = 1e-3,
) -> None:
    # Runs are CPU, float64, with MGS re-orthonormalization.
    device_t = torch.device("cpu")
    dtype = torch.float64
    precision = "float64"
    reorthogonalize = "mgs"
    set_seed(seed)

    if projection_vector_mode not in {"displacement", "absolute"}:
        raise ValueError(f"projection_vector_mode must be displacement or absolute, got {projection_vector_mode!r}")
    if probe_source not in {"synthetic", "train", "test"}:
        raise ValueError(f"probe_source must be synthetic, train or test, got {probe_source!r}")
    if Path(run_subdir_name).is_absolute():
        raise ValueError("--run-subdir-name must be a relative path")

    base_path = Path(out_dir)
    model_dir = base_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = model_dir / "checkpoint.pt"
    eig_cache_dir = model_dir / "eigen_cache"
    eig_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_path / run_subdir_name
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Teacher and data
    # ------------------------------------------------------------------
    print("[0/5] Constructing teacher and generating synthetic data...")
    teacher = make_teacher(
        input_dim=input_dim,
        output_dim=output_dim,
        teacher_hidden_dim=teacher_hidden_dim,
        noise_std=noise_std,
        device=device_t,
        dtype=dtype,
    )
    print(f"  Teacher info: {teacher.info()}")

    x_train, y_train, x_test, y_test, x_probe, y_probe = generate_synthetic_data(
        teacher=teacher,
        train_size=train_size,
        test_size=test_size,
        probe_size=probe_size,
        input_dim=input_dim,
        device=device_t,
        dtype=dtype,
        seed=seed,
    )
    if probe_source == "train":
        x_probe = x_train
    elif probe_source == "test":
        x_probe = x_test

    # Noiseless train targets, for the training snapshots.
    with torch.no_grad():
        y_train_teacher = teacher(x_train)

    print(
        f"  x_train={tuple(x_train.shape)}, y_train={tuple(y_train.shape)}, "
        f"x_test={tuple(x_test.shape)}, x_probe={tuple(x_probe.shape)}\n"
    )

    # ------------------------------------------------------------------
    # Student
    # ------------------------------------------------------------------
    model = StudentSingleHidden(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        activation="relu",
        bias=True,
    ).to(device=device_t, dtype=dtype)
    print(f"  Student: {input_dim} -> {hidden_dim} -> {output_dim} (ReLU, bias), "
          f"parameter_count={sum(p.numel() for p in model.parameters())}\n")

    layout = get_param_layout(model)
    theta0 = flatten_params(model).clone()

    # ------------------------------------------------------------------
    # Training (or reuse of an existing checkpoint)
    # ------------------------------------------------------------------
    print("[1/5] Training student on teacher labels (MSE)...")

    expected_config_validation = {
        "teacher_hidden_dim": teacher_hidden_dim,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": hidden_dim,
        "noise_std": noise_std,
        "train_size": train_size,
        "test_size": test_size,
        "steps": steps,
        "lr": lr,
        "seed": seed,
    }

    if model_checkpoint_path.exists():
        print(f"  Loading checkpoint from {model_checkpoint_path}...")
        checkpoint = torch.load(model_checkpoint_path, map_location=device_t)
        cfg = checkpoint.get("config_validation")
        if not isinstance(cfg, dict):
            raise ValueError("Checkpoint has no config_validation entry; remove it and retrain.")
        mismatches = [
            f"{key}: checkpoint={cfg.get(key, '<missing>')!r}, current={value!r}"
            for key, value in expected_config_validation.items()
            if cfg.get(key, object()) != value
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint config mismatch. Cannot reuse checkpoint.\n  - "
                + "\n  - ".join(mismatches)
                + "\nRemove the checkpoint or use a different --out-dir."
            )
        model.load_state_dict({
            k: (v.to(dtype=dtype) if torch.is_floating_point(v) else v)
            for k, v in checkpoint["model_state_dict"].items()
        })
        model.to(device=device_t, dtype=dtype)
        theta_star = checkpoint["theta_star"].to(device=device_t, dtype=dtype)
        losses_train = checkpoint["losses_train"]
        losses_test = checkpoint["losses_test"]
        print(f"  Checkpoint loaded. Training time was {checkpoint['train_time_sec']:.1f}s\n")
    else:
        print("  No checkpoint found, training from scratch...")
        t_train_start = time.perf_counter()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        losses_train: list[float] = []
        losses_test: list[float] = []

        # Parameter snapshots at num_snapshots equally spaced steps in [0, steps],
        # used by run_temporal_recovery.py.
        snapshots: list[dict] = []
        if num_snapshots > 0:
            snapshot_steps = set(np.linspace(0, steps, max(num_snapshots, 2), dtype=int).tolist()) | {0, steps}
        else:
            snapshot_steps = set()

        def _snapshot(step_count: int, loss_test: float) -> dict:
            return {
                "step": step_count,
                "theta": flatten_params(model).clone().cpu(),
                "loss_train_noisy": float(F.mse_loss(model(x_train), y_train).item()),
                "loss_train_teacher": float(F.mse_loss(model(x_train), y_train_teacher).item()),
                "loss_test": loss_test,
            }

        if num_snapshots > 0:
            with torch.no_grad():
                snapshots.append(_snapshot(0, float(F.mse_loss(model(x_test), y_test).item())))

        for step in range(steps):
            opt.zero_grad()
            loss = task_mse(model, x_train, y_train)
            loss.backward()
            opt.step()
            losses_train.append(float(loss.item()))

            with torch.no_grad():
                loss_test = float(F.mse_loss(model(x_test), y_test).item())
                losses_test.append(loss_test)
                if (step + 1) in snapshot_steps:
                    snapshots.append(_snapshot(step + 1, loss_test))

            if (step + 1) % max(1, steps // 10) == 0:
                print(f"  step {step + 1}/{steps}, train_mse={losses_train[-1]:.6e}, test_mse={loss_test:.6e}")

        t_train_sec = time.perf_counter() - t_train_start
        print(f"  training done in {t_train_sec:.1f}s\n")

        theta_star = flatten_params(model).clone()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "theta_star": theta_star,
            "theta_0": theta0,
            "losses_train": losses_train,
            "losses_test": losses_test,
            "train_time_sec": t_train_sec,
            "seed": seed,
            "teacher_info": teacher.info(),
            "config_validation": expected_config_validation,
            "snapshots": snapshots,
            "x_probe": x_probe.cpu(),
            "x_train": x_train.cpu(),
            "y_train": y_train.cpu(),
            "y_train_teacher": y_train_teacher.cpu(),
            "x_test": x_test.cpu(),
            "y_test": y_test.cpu(),
            "student_arch": {
                "student_type": "single_hidden",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": hidden_dim,
                "student_activation": "relu",
                "student_bias": True,
            },
        }
        torch.save(checkpoint, model_checkpoint_path)
        print(f"  Checkpoint saved to {model_checkpoint_path}\n")

    # ------------------------------------------------------------------
    # Reference metrics at theta_star and theta_0
    # ------------------------------------------------------------------
    print("[2/5] Evaluating theta_star on train/test sets...")
    model.eval()
    train_mse_star = float(F.mse_loss(model(x_train), y_train).item())
    test_mse_star = float(F.mse_loss(model(x_test), y_test).item())
    print(f"  train_mse={train_mse_star:.6e}, test_mse={test_mse_star:.6e}\n")

    training_curve_path = model_dir / "training_loss_curve.png"
    if losses_train and not training_curve_path.exists():
        save_training_curve(losses_train, training_curve_path, ylabel="Task MSE (log scale)",
                            title="Student Training MSE Curve")

    with torch.no_grad():
        behavioral_target_probe = model(x_probe).detach().clone()
        behavior_mse_star = float(torch.mean((model(x_probe) - behavioral_target_probe) ** 2).item())

    # Losses at the two reference points: theta_0 (*_mse_theta0) and theta = 0
    # (*_mse_zero). ts_rank_data
    # normalises the relative rank by the loss at the projection's reference point
    # (M(0) for absolute mode, M(theta_0) for displacement mode); with
    # --relative-ref theta0 by M(theta_0) in both modes.
    with torch.no_grad():
        set_params_from_flat(model, theta0, layout)
        train_mse_theta0 = float(F.mse_loss(model(x_train), y_train).item())
        test_mse_theta0 = float(F.mse_loss(model(x_test), y_test).item())
        behavior_mse_theta0 = float(F.mse_loss(model(x_probe), behavioral_target_probe).item())
        set_params_from_flat(model, torch.zeros_like(theta_star), layout)
        train_mse_zero = float(F.mse_loss(model(x_train), y_train).item())
        test_mse_zero = float(F.mse_loss(model(x_test), y_test).item())
        set_params_from_flat(model, theta_star, layout)
    print(
        f"  theta0 ref: train_mse_theta0={train_mse_theta0:.6e}, "
        f"test_mse_theta0={test_mse_theta0:.6e}, behavior_mse_theta0={behavior_mse_theta0:.6e}\n"
    )

    # ------------------------------------------------------------------
    # Behavioral eigenpairs
    # ------------------------------------------------------------------
    dim = theta_star.numel()
    print(f"[3/5] Computing behavioral eigenpairs (param_dim={dim}, top_m={top_m}, "
          f"projection_vector_mode={projection_vector_mode})...\n")

    # The behavioral operator has rank at most min(|P| * output_dim, p).
    dual_dim = int(x_probe.shape[0] * output_dim)
    behavioral_rank_cap_effective = min(dual_dim, dim)
    top_m_requested_effective_behavioral = (
        behavioral_rank_cap_effective if top_m <= 0 else min(top_m, behavioral_rank_cap_effective)
    )
    evals_beh, evecs_beh_dual, beh_recovery_state = compute_behavioral_eigenspace_dual(
        model=model,
        x_probe=x_probe,
        top_m=top_m_requested_effective_behavioral,
        device=device_t,
        dtype=theta_star.dtype,
        cache_path=eig_cache_dir / f"behavioral_dual_top_{probe_source}_probe{int(x_probe.shape[0])}.pt",
    )
    top_m_used_behavioral = len(evals_beh)
    print(f"  Behavioral dual eigensolve done (max eigenvalue: {evals_beh[0].item():.6e}), "
          f"got {top_m_used_behavioral} dual eigenpairs\n")

    hvp_beh = make_beh_hvp_operator(model, x_probe, behavioral_target_probe)
    behavioral_validation = validate_behavioral_eigenspace_lazy(
        evals_dual=evals_beh,
        evecs_dual=evecs_beh_dual,
        recovery_state=beh_recovery_state,
        hvp_operator=hvp_beh,
        device=device_t,
        dtype=theta_star.dtype,
        max_pairs=5,
    )
    print("  Behavioral eigenspace validation (parameter space):")
    for rec in behavioral_validation["residuals"]:
        print(f"    pair {int(rec['pair_index'])}: lambda={rec['eigenvalue']:.6e}, "
              f"rel_residual={rec['relative_residual']:.3e}")
    print(f"    max |V^T V - I| over checked pairs: {behavioral_validation['max_abs_orthogonality_deviation']:.3e}\n")

    # ------------------------------------------------------------------
    # Projection curves
    # ------------------------------------------------------------------
    k_values_beh = build_k_grid(top_m_used_behavioral)
    metric_fns, metric_formats = make_regression_metric_fns(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_probe=x_probe,
        behavioral_target_probe=behavioral_target_probe,
    )
    projection_diagnostics: dict[str, object] = {}
    metrics_by_k, _ = project_and_evaluate_behavioral_eigenspace_lazy(
        evals_beh,
        evecs_beh_dual,
        beh_recovery_state,
        theta0,
        theta_star,
        k_values_beh,
        model,
        layout,
        device_t,
        theta_star.dtype,
        set_params_from_flat_fn=set_params_from_flat,
        metric_fns=metric_fns,
        metric_formats=metric_formats,
        short_name="behavioral",
        projection_vector_mode=projection_vector_mode,
        dependence_tol=dependence_tol,
        diagnostics_out=projection_diagnostics,
    )
    set_params_from_flat(model, theta_star, layout)

    train_mse_by_k = metrics_by_k["train_loss"]
    test_mse_by_k = metrics_by_k["test_metric"]
    beh_mse_by_k = metrics_by_k["behavior_mse"]
    all_k = sorted(train_mse_by_k)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    print("[4/5] Saving plots and experiment outputs...")

    evals_np = evals_beh.detach().cpu().numpy()
    with (out_path / "eigenvalues_behavioral.txt").open("w", encoding="utf-8") as f:
        f.write("# idx,eigenvalue\n")
        for idx, val in enumerate(evals_np, start=1):
            f.write(f"{idx},{val:.16e}\n")

    req = top_m_requested_effective_behavioral
    k_xlabel = f"k (requested up to {req}, available up to {top_m_used_behavioral})"
    save_eigenvalues_plot(
        evals_np, out_path / "eigenvalues_behavioral.png",
        title=f"Top Hessian Eigenvalues (Behavioral, requested={req}, used={top_m_used_behavioral})",
    )
    for curve, name, label in [(train_mse_by_k, "train_mse", "Train MSE"),
                               (test_mse_by_k, "test_mse", "Test MSE"),
                               (beh_mse_by_k, "behavior_mse", "Behavior MSE")]:
        save_topk_metrics(
            k_values=all_k,
            values=[curve[k] for k in all_k],
            ylabel=f"{label}(k)",
            title=f"{label} vs Top-k Projection (Behavioral)",
            out_path=out_path / f"{name}_vs_k_behavioral.png",
            xlabel=k_xlabel,
        )

    reference_metrics = {
        "train_mse_star": train_mse_star,
        "test_mse_star": test_mse_star,
        "behavior_mse_star": behavior_mse_star,
        "train_mse_theta0": train_mse_theta0,
        "test_mse_theta0": test_mse_theta0,
        "behavior_mse_theta0": behavior_mse_theta0,
        "train_mse_zero": train_mse_zero,
        "test_mse_zero": test_mse_zero,
    }
    k_last = int(all_k[-1])
    behavioral_final_metrics = {
        "train_mse_true_model": float(train_mse_star),
        "train_mse_behavioral_last": float(train_mse_by_k[k_last]),
        "train_mse_gap_last_minus_true": float(train_mse_by_k[k_last]) - float(train_mse_star),
        "test_mse_true_model": float(test_mse_star),
        "test_mse_behavioral_last": float(test_mse_by_k[k_last]),
        "test_mse_gap_last_minus_true": float(test_mse_by_k[k_last]) - float(test_mse_star),
        "behavior_mse_true_model": float(behavior_mse_star),
        "behavior_mse_behavioral_last": float(beh_mse_by_k[k_last]),
        "behavior_mse_gap_last_minus_true": float(beh_mse_by_k[k_last]) - float(behavior_mse_star),
    }

    experiment_outputs: dict[str, object] = {
        "seed": int(seed),
        "teacher_info": teacher.info(),
        "student_config": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "activation": "ReLU",
            "architecture": "Linear-Activation-Linear",
            "param_dim": int(dim),
        },
        "theta_0": theta0.cpu(),
        "theta_star": theta_star.cpu(),
        "train_loss_curve": torch.tensor(losses_train),
        "test_loss_curve": torch.tensor(losses_test),
        **reference_metrics,
        "projection_vector_mode": projection_vector_mode,
        "top_eigenvalues_behavioral": evals_beh.cpu(),
        "dual_dim_behavioral": dual_dim,
        "behavioral_rank_cap_effective": int(behavioral_rank_cap_effective),
        "top_m_used_behavioral": int(top_m_used_behavioral),
        "behavioral_eigenspace_validation": behavioral_validation,
        "k_values_behavioral": torch.tensor(all_k, dtype=torch.int64),
        "k_values": torch.tensor(all_k, dtype=torch.int64),
        "train_mse_by_k_behavioral": torch.tensor([train_mse_by_k[k] for k in all_k]),
        "test_mse_by_k_behavioral": torch.tensor([test_mse_by_k[k] for k in all_k]),
        "beh_mse_by_k_behavioral": torch.tensor([beh_mse_by_k[k] for k in all_k]),
        "behavioral_last_k": k_last,
        "behavioral_final_metrics": behavioral_final_metrics,
    }
    torch.save(experiment_outputs, out_path / "experiment_outputs.pt")

    summary: dict[str, object] = {
        "seed": int(seed),
        "curvature": "behavioral",
        "projection_mode": "forward",
        "projection_vector_mode": projection_vector_mode,
        "probe_source": probe_source,
        "precision": precision,
        "reorthogonalize": reorthogonalize,
        "dependence_tol": float(dependence_tol),
        "teacher_info": teacher.info(),
        "student_type": "StudentSingleHidden",
        "student_architecture": "Linear-Activation-Linear",
        "student_param_dim": int(dim),
        "student_hidden_dim": int(hidden_dim),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "probe_size": int(x_probe.shape[0]),
        "steps": int(steps),
        "lr": float(lr),
        "top_m_requested": int(top_m),
        **{k: float(v) for k, v in reference_metrics.items()},
        "top_m_used_behavioral": int(top_m_used_behavioral),
        "top_m_requested_effective_behavioral": int(top_m_requested_effective_behavioral),
        "behavioral_top_m_truncated": bool(top_m_used_behavioral < top_m_requested_effective_behavioral),
        "dual_dim_behavioral": dual_dim,
        "behavioral_rank_cap_effective": int(behavioral_rank_cap_effective),
        "behavioral_eigenspace_validation": behavioral_validation,
        "behavioral_last_k": k_last,
        "behavioral_final_metrics": behavioral_final_metrics,
        "run_subdir_name": run_subdir_name,
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"  Outputs saved to {out_path}\n")
    print("[5/5] Done.")



# ---------------------------------------------------------------------------
# Rebuilding a finished run
# ---------------------------------------------------------------------------

def rebuild_run_data(summary: dict, device: torch.device = torch.device("cpu")):
    """Teacher and data of a finished run, rebuilt from its summary.json.

    The teacher is drawn from the global RNG, so set_seed is called first with the
    run's seed, as in main(). Returns (teacher, (x_train, y_train, x_test, y_test,
    x_probe, y_probe), dtype); the training labels include the noise draw.
    """
    ti = summary["teacher_info"]
    dtype = torch.float64 if summary.get("precision", "float32") == "float64" else torch.float32
    seed = int(summary["seed"])
    set_seed(seed)
    teacher = make_teacher(
        teacher_type=ti["type"], input_dim=int(ti["input_dim"]), output_dim=int(ti["output_dim"]),
        teacher_hidden_dim=int(ti.get("hidden_dim", 0)), teacher_activation=ti.get("activation", "relu"),
        teacher_bias=bool(ti.get("bias", True)), noise_std=float(ti.get("noise_std", 0.0)),
        device=device, dtype=dtype)
    data = generate_synthetic_data(
        teacher=teacher, train_size=int(summary["train_size"]), test_size=int(summary["test_size"]),
        probe_size=int(summary.get("probe_size", 512)), input_dim=int(ti["input_dim"]),
        device=device, dtype=dtype, seed=seed)
    return teacher, data, dtype


def student_for_run(summary: dict, dtype: torch.dtype, device: torch.device = torch.device("cpu")):
    """An (untrained) student with the architecture of a finished run."""
    ti = summary["teacher_info"]
    return StudentSingleHidden(
        input_dim=int(ti["input_dim"]), output_dim=int(ti["output_dim"]),
        hidden_dim=int(summary["student_hidden_dim"]),
        activation=summary.get("student_activation", ti.get("activation", "relu")), bias=True,
    ).to(device=device, dtype=dtype)


def recompute_mse_at_zero(summary: dict, payload: dict) -> dict[str, float]:
    """Train and test MSE at theta = 0 for a run whose summary has no *_mse_zero.

    As a check, the stored M(theta*) and M(theta_0) must be reproduced to 1e-12.
    """
    _, (x_tr, y_tr, x_te, y_te, _, _), dtype = rebuild_run_data(summary)
    model = student_for_run(summary, dtype)
    model.eval()
    layout = get_param_layout(model)

    def mse_at(theta: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            set_params_from_flat(model, theta.to(dtype), layout)
            return (float(F.mse_loss(model(x_tr), y_tr).item()),
                    float(F.mse_loss(model(x_te), y_te).item()))

    theta_star = payload["theta_star"].to(dtype)
    for name, theta in (("star", theta_star), ("theta0", payload["theta_0"])):
        for split, val in zip(("train", "test"), mse_at(theta)):
            ref = float(summary[f"{split}_mse_{name}"])
            if abs(val - ref) > 1e-12 * max(abs(val), abs(ref), 1e-300):
                raise RuntimeError(f"cannot reproduce stored {split}_mse_{name} ({val!r} vs {ref!r}); "
                                   "teacher/data do not match the run")
    tr0, te0 = mse_at(torch.zeros_like(theta_star))
    return {"train_mse_zero": tr0, "test_mse_zero": te0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teacher-student regression: train a student and compute behavioral projection curves.",
        epilog=(
            "Example (main setting, student width 5):\n"
            "  python src/run_toy_teacher_student.py --seed 0 --hidden-dim 5 \\\n"
            "      --probe-source train --projection-vector-mode absolute \\\n"
            "      --out-dir results/ts_in10_out10/in10_th10_out10_sd5_bias_seed0 \\\n"
            "      --run-subdir-name runs/topm_-1_probesrc_train_pvec_absolute\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Fixes the teacher, the train/test inputs and the student initialization.")
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--output-dim", type=int, default=10)
    parser.add_argument("--teacher-hidden-dim", type=int, default=10, help="Teacher width.")
    parser.add_argument("--hidden-dim", type=int, required=True, help="Student width d.")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Std of the Gaussian label noise on the training targets.")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--probe-size", type=int, default=512,
                        help="Size of the independent probe set (only used with --probe-source synthetic).")
    parser.add_argument("--probe-source", choices=["train", "test", "synthetic"], default="train",
                        help="Probe set P for the behavioral Hessian.")
    parser.add_argument("--steps", type=int, default=4000, help="Full-batch Adam steps.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-m", type=int, default=-1,
                        help="Number of behavioral eigenpairs; -1 uses the rank cap min(|P| * output_dim, p).")
    parser.add_argument("--projection-vector-mode", choices=["absolute", "displacement"], default="absolute",
                        help="absolute projects theta_star (theta_ref = 0); "
                             "displacement projects theta_star - theta_0 (theta_ref = theta_0).")
    parser.add_argument("--num-snapshots", type=int, default=40,
                        help="Equally spaced parameter snapshots saved during training (for run_temporal_recovery.py).")
    parser.add_argument("--dependence-tol", type=float, default=1e-3,
                        help="Stop the projection when a re-orthonormalized eigenvector retains less than this norm.")
    parser.add_argument("--out-dir", type=str, required=True, help="Configuration directory (holds model/).")
    parser.add_argument("--run-subdir-name", type=str, required=True,
                        help="Relative subdirectory under --out-dir for this run's outputs.")
    args = parser.parse_args()

    run_experiment(
        out_dir=args.out_dir,
        run_subdir_name=args.run_subdir_name,
        seed=args.seed,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        teacher_hidden_dim=args.teacher_hidden_dim,
        hidden_dim=args.hidden_dim,
        noise_std=args.noise_std,
        train_size=args.train_size,
        test_size=args.test_size,
        probe_size=args.probe_size,
        probe_source=args.probe_source,
        steps=args.steps,
        lr=args.lr,
        top_m=args.top_m,
        projection_vector_mode=args.projection_vector_mode,
        num_snapshots=args.num_snapshots,
        dependence_tol=args.dependence_tol,
    )
