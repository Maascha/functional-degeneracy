"""Quick-look plots written next to the run outputs (quick diagnostics only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_training_curve(
    losses: list[float],
    out_path: Path,
    ylabel: str = "Task MSE (log scale)",
    title: str = "Training Loss Curve",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(losses)), losses, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_eigenvalues_plot(
    eigenvalues: np.ndarray,
    out_path: Path,
    log_y: bool = False,
    title: str = "Top Hessian Eigenvalues",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    x_idx = np.arange(1, len(eigenvalues) + 1)
    y_vals = eigenvalues

    # Log scale requires strictly positive y values.
    if log_y:
        positive_mask = y_vals > 0
        x_idx = x_idx[positive_mask]
        y_vals = y_vals[positive_mask]

    ax.plot(x_idx, y_vals, linewidth=1)
    if log_y and len(y_vals) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("Index (sorted descending)")
    ax.set_ylabel("Hessian eigenvalue" + (" (log scale)" if log_y else ""))
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_topk_metrics(
    k_values: list[int],
    values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
    xlabel: str = "k (top Hessian directions kept)",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, values, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
