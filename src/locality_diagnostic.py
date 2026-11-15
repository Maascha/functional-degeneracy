"""Quadratic approximation of the behavioral loss along the projection path.

Along the projection path theta_k = sum_{j<=k} <v_j, theta*> v_j (reference
point 0, eigenvalue order, as for r_tau) this compares

    true       L_beh(theta_k) = 1/(2|P|) sum_{x in P} ||f_{theta_k}(x) - f_{theta*}(x)||^2
    quadratic  Q_k            = 1/2 sum_{j>k} lambda_j <v_j, theta*>^2

Q_k is the quadratic (second-order) approximation evaluated on the path: H_beh = J_P^T J_P / |P| is the exact
Hessian of L_beh at theta*. Dense eigendecomposition (p <= 430), with the same
eigenvalue cutoff as the rank computation (lambda_j > 1e-6 lambda_1) and the training
inputs as probe set. Plots both curves for a few student widths and one seed,
marking k = r_tau.

    python src/locality_diagnostic.py \\
        --checkpoint-root results/ts_in10_out10 \\
        --out-dir figures/ts/in10_out10/locality
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import prune_vs_recovery_rank as P
from train import flatten_params, get_param_layout, set_params_from_flat
from ts_plot_utils import PAPER_FIGSIZE, save_figure  # also sets pdf.fonttype 42

MIN_RELATIVE_EVAL = 1e-6


def path_curves(model, x_pr, x_te, y_te):
    """Return (true L_beh, quadratic Q, test MSE) for k = 0..K."""
    layout = get_param_layout(model)
    theta = flatten_params(model).clone()
    with torch.no_grad():
        f_star = model(x_pr).clone()
    H = P.behavioral_hessian_dense(model, x_pr)
    evals, evecs = torch.linalg.eigh(H)
    evals, evecs = evals.flip(0), evecs.flip(1)
    keep = evals > max(MIN_RELATIVE_EVAL * float(evals[0]), 1e-8)
    evals, evecs = evals[keep], evecs[:, keep]
    c = evecs.T @ theta                       # theta_ref = 0
    cost = evals * c ** 2
    total = float(cost.sum())
    acc = torch.zeros_like(theta)
    true, quad, test = [], [], []
    removed = 0.0
    with torch.no_grad():
        set_params_from_flat(model, acc, layout)   # k = 0
        true.append(float(((model(x_pr) - f_star) ** 2).sum() / (2 * x_pr.shape[0])))
        quad.append(0.5 * total)
        test.append(P.test_mse(model, x_te, y_te))
        for j in range(len(evals)):
            acc = acc + c[j] * evecs[:, j]
            removed += float(cost[j])
            set_params_from_flat(model, acc, layout)
            true.append(float(((model(x_pr) - f_star) ** 2).sum() / (2 * x_pr.shape[0])))
            quad.append(0.5 * max(total - removed, 0.0))
            test.append(P.test_mse(model, x_te, y_te))
    set_params_from_flat(model, theta, layout)
    return np.array(true), np.array(quad), np.array(test)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--teacher-width", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.075)
    ap.add_argument("--example-widths", type=int, nargs="+", default=[4, 7, 10, 14, 20])
    ap.add_argument("--example-seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = {}
    for cfg in sorted(args.checkpoint_root.glob("in*_th*_out*_sd*_seed*")):
        m = P.CFG_RE.search(cfg.name)
        if (m and int(m.group(2)) == args.teacher_width
                and int(m.group(6)) == args.example_seed):
            cfgs[int(m.group(4))] = cfg

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    cols = plt.cm.viridis(np.linspace(0.0, 0.85, len(args.example_widths)))
    for d, col in zip(args.example_widths, cols):
        run = P.load_run(cfgs[d], probe_source="train", pvec="absolute") if d in cfgs else None
        if run is None:
            continue
        L, Q, T = path_curves(run["model"], run["probe"], run["x_te"], run["y_te"])
        K = len(L) - 1
        r_tau = next((k for k in range(1, K + 1) if T[k] <= args.tau), None)
        print(f"  d={d:2d} seed={args.example_seed}  K={K}  r_tau={r_tau}", flush=True)
        k = np.arange(len(L))
        ax.plot(k, L, color=col, lw=1.4, label=f"$d={d}$")
        ax.plot(k, Q, color=col, lw=1.2, ls="--")
        if r_tau is not None:
            ax.plot(r_tau, L[r_tau], marker="o", ms=4, color=col, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e5)   # tails past r_tau fall to ~1e-10; headroom for the legend
    ax.set_yticks([1e-4, 1e-2, 1e0, 1e2])   # no tick labels behind the legend
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"Leading eigendirections $k$")
    ax.set_ylabel(r"$L_{\mathrm{beh}}(\hat\theta_k)$")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color="0.3", lw=1.4), Line2D([0], [0], color="0.3", lw=1.2, ls="--"),
          Line2D([0], [0], color="0.3", ls="none", marker="o", ms=4)]
    l += ["true", "quadratic", r"$k=r_\tau$"]
    ax.legend(h, l, loc="upper right", ncol=2, frameon=False, handlelength=1.0,
              handletextpad=0.3, labelspacing=0.15, columnspacing=0.6, borderpad=0.2,
              borderaxespad=0.3)
    for f in save_figure(fig, args.out_dir / "locality_example_curves"):
        print("wrote", f)


if __name__ == "__main__":
    main()
