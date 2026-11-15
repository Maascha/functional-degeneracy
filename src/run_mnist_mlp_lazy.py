"""MNIST MLP with behavioral projection curves.

Trains a two-hidden-layer ReLU MLP 784 -> h1 -> h2 -> 10 on MNIST with
mini-batch Adam and cross-entropy, computes the leading behavioral eigenpairs
on a probe set of training images (matrix-free dual Lanczos, see curvature_projection.py), and
evaluates the models obtained by projecting theta_star - theta_0 onto the
top-k eigendirections (theta_ref = theta_0) for every k. Parameter-space
eigenvectors are recovered one at a time to bound GPU memory.

Output layout under --out-dir:
    width_<h1>_<h2>_bs_<b>/model/          checkpoint.pt, eigen_cache/, training curve
    width_<h1>_<h2>_bs_<b>_topm_<M>_curv_behavioral_pmode_both_probesrc_<src>/
                                           experiment_outputs.pt, summary.json, quick-look plots
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


class SmallMLP(nn.Module):
    def __init__(self, hidden_dims: tuple[int, int] = (128, 64)) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(784, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def task_cross_entropy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(model(x), y)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = model(x).argmax(dim=1)
    return float((pred == y).float().mean().item())


def select_subset_indices(total_size: int, subset_size: int, seed: int) -> torch.Tensor:
    """First subset_size entries of a random permutation drawn from its own generator."""
    if subset_size > total_size:
        raise ValueError(f"subset_size={subset_size} cannot exceed dataset size={total_size}")
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(total_size, generator=g)[:subset_size]


def parse_hidden_dims(hidden_dims: str) -> tuple[int, int]:
    parts = [p.strip() for p in hidden_dims.split(",")]
    if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"hidden_dims must be two comma-separated positive integers, got '{hidden_dims}'")
    return (int(parts[0]), int(parts[1]))


def load_mnist_subsets(
    data_root: Path,
    train_subset_size: int,
    test_subset_size: int,
    probe_size: int,
    probe_source: str,
    seed: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    """Load MNIST, flatten, and normalize with the training-set mean and std.

    Subset sizes <= 0 use the full split. The probe set is a random subset of
    probe_size images (all if <= 0) of the train or test split. Returns
    (x_train, y_train, x_test, y_test, x_probe, y_probe, normalization stats).
    """
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    if train_subset_size <= 0:
        train_subset_size = len(train_set)
    if test_subset_size <= 0:
        test_subset_size = len(test_set)

    train_idx = select_subset_indices(len(train_set), train_subset_size, seed)
    test_idx = select_subset_indices(len(test_set), test_subset_size, seed + 1)

    train_imgs = train_set.data[train_idx].float() / 255.0
    train_labels = train_set.targets[train_idx].long()

    test_imgs = test_set.data[test_idx].float() / 255.0
    test_labels = test_set.targets[test_idx].long()

    x_train = train_imgs.view(train_subset_size, -1)
    x_test = test_imgs.view(test_subset_size, -1)

    mean = x_train.mean()
    std = x_train.std().clamp_min(1e-6)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    if probe_source == "test":
        probe_base_x = x_test
        probe_base_y = test_labels
    elif probe_source == "train":
        probe_base_x = x_train
        probe_base_y = train_labels
    else:
        raise ValueError(f"Unsupported probe_source '{probe_source}'. Use one of: test, train")

    probe_base_size = probe_base_x.shape[0]
    if probe_size <= 0:
        probe_idx = torch.arange(probe_base_size)
    else:
        probe_idx = select_subset_indices(probe_base_size, probe_size, seed + 2)

    x_probe = probe_base_x[probe_idx]
    y_probe = probe_base_y[probe_idx]

    return (
        x_train.to(device),
        train_labels.to(device),
        x_test.to(device),
        test_labels.to(device),
        x_probe.to(device),
        y_probe.to(device),
        {"input_mean": float(mean.item()), "input_std": float(std.item())},
    )


def make_projection_metric_fns(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    x_probe: torch.Tensor,
    logits_star_probe: torch.Tensor,
) -> tuple[dict[str, Callable[[nn.Module], float]], dict[str, str]]:
    """Metrics evaluated at every projected model: train CE, train and test
    accuracy, and the MSE to the logits of theta_star on the probe set."""

    def train_loss_metric(model: nn.Module) -> float:
        with torch.no_grad():
            return float(task_cross_entropy(model, x_train, y_train).item())

    def train_metric(model: nn.Module) -> float:
        return accuracy(model, x_train, y_train)

    def test_metric(model: nn.Module) -> float:
        return accuracy(model, x_test, y_test)

    def behavior_mse_metric(model: nn.Module) -> float:
        with torch.no_grad():
            return float(torch.mean((model(x_probe) - logits_star_probe) ** 2).item())

    metric_fns = {
        "train_loss": train_loss_metric,
        "train_metric": train_metric,
        "test_metric": test_metric,
        "behavior_mse": behavior_mse_metric,
    }
    metric_formats = {
        "train_loss": ".6f",
        "train_metric": ".4f",
        "test_metric": ".4f",
        "behavior_mse": ".6e",
    }
    return metric_fns, metric_formats


def run_experiment(
    out_dir: str,
    hidden_dims: tuple[int, int],
    seed: int = 0,
    probe_size: int = 10000,
    probe_source: str = "train",
    steps: int = 2000,
    lr: float = 1e-3,
    train_batch_size: int = 64,
    top_m: int = 12000,
    device: str = "cuda",
    data_root: str = "data",
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    device_t = torch.device(device)
    set_seed(seed)
    if train_batch_size < 1:
        raise ValueError(f"train_batch_size must be positive, got {train_batch_size}")
    # Full training and test sets (recorded as -1 in the checkpoint).
    train_subset_size = -1
    test_subset_size = -1

    base_path = Path(out_dir) / f"width_{hidden_dims[0]}_{hidden_dims[1]}_bs_{train_batch_size}"
    model_dir = base_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = model_dir / "checkpoint.pt"
    eig_cache_dir = model_dir / "eigen_cache"
    eig_cache_dir.mkdir(parents=True, exist_ok=True)

    # This directory name is the layout read by compare_widths.py and
    # prune_vs_recovery_rank_mnist.py.
    out_path = Path(f"{base_path}_topm_{top_m}_curv_behavioral_pmode_both_probesrc_{probe_source}")
    out_path.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test, x_probe, y_probe, norm_stats = load_mnist_subsets(
        data_root=Path(data_root),
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
        probe_size=probe_size,
        probe_source=probe_source,
        seed=seed,
        device=device_t,
    )
    effective_train_batch_size = min(train_batch_size, x_train.shape[0])

    model = SmallMLP(hidden_dims=hidden_dims).to(device_t)
    layout = get_param_layout(model)
    theta0 = flatten_params(model).clone()

    print("[1/5] Training model...")
    if model_checkpoint_path.exists():
        print(f"  Loading checkpoint from {model_checkpoint_path}...")
        checkpoint = torch.load(model_checkpoint_path, map_location=device_t)
        if checkpoint.get("model_config", {}).get("hidden_dims") != list(hidden_dims):
            raise ValueError(f"Checkpoint at {model_checkpoint_path} has a different architecture.")
        model.load_state_dict(checkpoint["model_state_dict"])
        theta_star = checkpoint["theta_star"]
        losses_train = checkpoint["losses_train"]
        losses_test = checkpoint["losses_test"]
        accs_train = checkpoint["accs_train"]
        accs_test = checkpoint["accs_test"]
        print(f"  Checkpoint loaded. Training time: {checkpoint['train_time_sec']:.1f}s\n")
    else:
        print("  No checkpoint found, training from scratch...")
        t_train_start = time.perf_counter()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        losses_train: list[float] = []
        losses_test: list[float] = []
        accs_train: list[float] = []
        accs_test: list[float] = []

        for step in range(steps):
            opt.zero_grad()
            if effective_train_batch_size < x_train.shape[0]:
                # Each step's mini-batch has its own seeded generator.
                batch_idx = select_subset_indices(
                    x_train.shape[0],
                    effective_train_batch_size,
                    seed + 100000 + step,
                ).to(device_t)
                x_step = x_train[batch_idx]
                y_step = y_train[batch_idx]
            else:
                x_step = x_train
                y_step = y_train

            loss = task_cross_entropy(model, x_step, y_step)
            loss.backward()
            opt.step()
            losses_train.append(float(loss.item()))

            with torch.no_grad():
                model.eval()
                loss_test = float(task_cross_entropy(model, x_test, y_test).item())
                acc_train = accuracy(model, x_train, y_train)
                acc_test = accuracy(model, x_test, y_test)
                losses_test.append(loss_test)
                accs_train.append(acc_train)
                accs_test.append(acc_test)
                model.train()

            if (step + 1) % max(1, steps // 5) == 0:
                print(f"  step {step + 1}/{steps}, train_loss={losses_train[-1]:.6f}, test_loss={loss_test:.6f}, "
                      f"train_acc={acc_train:.4f}, test_acc={acc_test:.4f}")

        t_train_sec = time.perf_counter() - t_train_start
        print(f"  training done in {t_train_sec:.1f}s\n")

        theta_star = flatten_params(model).clone()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "theta_star": theta_star,
            "theta_0": theta0,
            "losses_train": losses_train,
            "losses_test": losses_test,
            "accs_train": accs_train,
            "accs_test": accs_test,
            "train_time_sec": t_train_sec,
            "seed": seed,
            "train_subset_size": train_subset_size,
            "test_subset_size": test_subset_size,
            "steps": steps,
            "lr": lr,
            "train_batch_size_requested": train_batch_size,
            "train_batch_size_effective": int(effective_train_batch_size),
            "model_config": {
                "input_dim": 784,
                "hidden_dims": list(hidden_dims),
                "activation": "ReLU",
                "output_dim": 10,
            },
        }
        torch.save(checkpoint, model_checkpoint_path)
        print(f"  Checkpoint saved to {model_checkpoint_path}\n")
        save_training_curve(losses_train, model_dir / "training_loss_curve.png",
                            ylabel="Task CE (log scale)", title="Training Cross-Entropy Curve")

    print("[2/5] Evaluating theta_star on train/test sets...")
    model.eval()
    train_ce_star = float(task_cross_entropy(model, x_train, y_train).item())
    train_acc_star = accuracy(model, x_train, y_train)
    test_acc_star = accuracy(model, x_test, y_test)
    # Test accuracy at the reference point theta_0, for the relative recovery rank.
    set_params_from_flat(model, theta0, layout)
    test_acc_theta0 = accuracy(model, x_test, y_test)
    set_params_from_flat(model, theta_star, layout)
    print(f"  train CE={train_ce_star:.6f}, acc={train_acc_star:.4f}; test acc={test_acc_star:.4f} "
          f"(theta_0: {test_acc_theta0:.4f})\n")

    # ------------------------------------------------------------------
    # Behavioral eigenpairs
    # ------------------------------------------------------------------
    print(f"[3/5] Computing behavioral eigenpairs (param_dim={theta_star.numel()}, top_m={top_m})...\n")
    with torch.no_grad():
        logits_star_probe = model(x_probe).detach().clone()

    evals_beh, evecs_beh_dual, beh_recovery_state = compute_behavioral_eigenspace_dual(
        model=model,
        x_probe=x_probe,
        top_m=top_m,
        device=device_t,
        dtype=theta_star.dtype,
        cache_path=eig_cache_dir / f"behavioral_dual_top_{probe_source}_n{x_probe.shape[0]}.pt",
    )
    print(f"  Behavioral dual eigensolve done (max eigenvalue: {evals_beh[0].item():.6e}), "
          f"got {len(evals_beh)} dual eigenpairs\n")

    hvp_beh = make_beh_hvp_operator(model, x_probe, logits_star_probe)
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
    # Projection curves (theta_ref = theta_0)
    # ------------------------------------------------------------------
    metric_fns, metric_formats = make_projection_metric_fns(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_probe=x_probe,
        logits_star_probe=logits_star_probe,
    )
    metrics_by_k, _ = project_and_evaluate_behavioral_eigenspace_lazy(
        evals_beh,
        evecs_beh_dual,
        beh_recovery_state,
        theta0,
        theta_star,
        build_k_grid(len(evals_beh)),
        model,
        layout,
        device_t,
        theta_star.dtype,
        set_params_from_flat_fn=set_params_from_flat,
        metric_fns=metric_fns,
        metric_formats=metric_formats,
        short_name="behavioral",
        projection_vector_mode="displacement",
    )
    set_params_from_flat(model, theta_star, layout)

    curves = {
        "task_ce": metrics_by_k["train_loss"],
        "train_accuracy": metrics_by_k["train_metric"],
        "test_accuracy": metrics_by_k["test_metric"],
        "behavior_mse": metrics_by_k["behavior_mse"],
    }
    all_k_values = sorted(curves["task_ce"])

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    save_eigenvalues_plot(evals_beh.detach().cpu().numpy(), out_path / "top_hessian_eigenvalues_behavioral.png")
    for name, label in [("task_ce", "Task CE"), ("train_accuracy", "Train Accuracy"),
                        ("test_accuracy", "Test Accuracy"), ("behavior_mse", "Behavior MSE")]:
        save_topk_metrics(
            k_values=all_k_values,
            values=[curves[name][k] for k in all_k_values],
            ylabel=f"{label}(k)",
            title=f"{label} vs Top-k Projection (Behavioral)",
            out_path=out_path / f"{name}_vs_k_behavioral.png",
        )

    experiment_outputs = {
        "seed": int(seed),
        "subset_sizes": {
            "train_subset_size": int(train_subset_size),
            "test_subset_size": int(test_subset_size),
            "probe_subset_size": int(x_probe.shape[0]),
        },
        "model_config": {
            "input_dim": 784,
            "hidden_dims": list(hidden_dims),
            "activation": "ReLU",
            "output_dim": 10,
        },
        "train_batch_size_requested": int(train_batch_size),
        "train_batch_size_effective": int(effective_train_batch_size),
        "projection_vector_mode": "displacement",
        "normalization": norm_stats,
        "theta_0": theta0.cpu(),
        "theta_star": theta_star.cpu(),
        "train_loss_curve": torch.tensor(losses_train),
        "train_acc_curve": torch.tensor(accs_train),
        "test_loss_curve": torch.tensor(losses_test),
        "test_acc_curve": torch.tensor(accs_test),
        "train_ce_star": train_ce_star,
        "train_acc_star": train_acc_star,
        "test_acc_star": test_acc_star,
        "test_acc_theta0": test_acc_theta0,
        "probe_labels": y_probe.cpu(),
        "top_eigenvalues_behavioral": evals_beh.cpu(),
        "behavioral_eigenspace_validation": behavioral_validation,
        "k_values_behavioral": torch.tensor(all_k_values, dtype=torch.int64),
    }
    for name, curve in curves.items():
        experiment_outputs[f"{name}_by_k_behavioral"] = torch.tensor([curve[k] for k in all_k_values])
    torch.save(experiment_outputs, out_path / "experiment_outputs.pt")

    summary: dict[str, object] = {
        "seed": int(seed),
        "curvature": "behavioral",
        "projection_vector_mode": "displacement",
        "probe_source": probe_source,
        "train_subset_size": int(train_subset_size),
        "test_subset_size": int(test_subset_size),
        "probe_size_used": int(x_probe.shape[0]),
        "train_batch_size_requested": int(train_batch_size),
        "train_batch_size_effective": int(effective_train_batch_size),
        "width": int(hidden_dims[0]),
        "steps": int(steps),
        "lr": float(lr),
        "top_m_requested": int(top_m),
        "train_ce_star": float(train_ce_star),
        "train_acc_star": float(train_acc_star),
        "test_acc_star": float(test_acc_star),
        "test_acc_theta0": float(test_acc_theta0),
        "top_m_used_behavioral": int(len(evals_beh)),
        "behavioral_eigenspace_validation": behavioral_validation,
    }
    with (out_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[5/5] Done. model: {model_dir}, outputs: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST MLP: train and compute behavioral projection curves (theta_ref = theta_0).",
        epilog="Example:\n  python src/run_mnist_mlp_lazy.py --hidden-dims 32,32 --out-dir results/mnist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=str, required=True, help="Hidden widths as 'H1,H2', e.g. '32,32'.")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000, help="Adam steps.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--probe-source", choices=["train", "test"], default="train",
                        help="Split from which the probe set is drawn.")
    parser.add_argument("--probe-size", type=int, default=10000, help="Probe images; -1 uses the whole split.")
    parser.add_argument("--top-m", type=int, default=12000,
                        help="Number of behavioral eigenpairs; -1 uses the full dual spectrum (|P| * 10).")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--data-root", type=str, default="data", help="MNIST location (downloaded if missing).")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    run_experiment(
        out_dir=args.out_dir,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        seed=args.seed,
        probe_size=args.probe_size,
        probe_source=args.probe_source,
        steps=args.steps,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        top_m=args.top_m,
        device=args.device,
        data_root=args.data_root,
    )
