"""Recovery rank during training.

Reads <root>/<cfg>/temporal/recovery_ranks_long.csv, written by
run_temporal_recovery.py, and plots r_tau (tau = 0.075, test MSE):

  (a) rank_vs_width_by_step: r_tau vs width at five checkpoints, mean over the
      seeds that reach tau, error bar = std (ddof=1); hollow marker when fewer
      than all seeds reach tau, no point when none does;
  (b) rank_vs_step: r_tau vs training step for a few widths (band = std);
  (c) test_mse_vs_step: mean test MSE vs training step for the same widths.

    python src/plot_ts_temporal.py --root-dir results/ts_in10_out10 \
        --out-dir figures/ts/in10_out10/temporal
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ts_plot_utils import save_figure  # also sets fonttype 42 and font sizes

CFG_RE = re.compile(r"in(\d+)_th(\d+)_out(\d+)_sd(\d+)_(?:bias|nobias)_seed(\d+)$")
ERR_ALPHA, ERR_CAP = 0.25, 2.0
WIDTH_TICKS = [6, 10, 14, 18]
TAU = 0.075
METRIC = "test_mse"
STEPS = [512, 1025, 2051, 2974, 4000]      # checkpoints shown in (a)
LINE_WIDTHS = [6, 10, 12, 16, 20]          # student widths shown in (b), (c)
MIN_WIDTH = 3
FIGSIZE = (3.9, 2.9)


def load(root: Path, kind: str, thr: float, metric: str, min_width: int):
    """rank[(d, step)] -> {seed: rank or nan}, loss[(d, step)] -> {seed: test mse}."""
    rank, loss, d_star = defaultdict(dict), defaultdict(dict), None
    for cfg in sorted(root.iterdir()):
        m = CFG_RE.match(cfg.name)
        f = cfg / "temporal" / "recovery_ranks_long.csv"
        if not m or not f.exists():
            continue
        d, seed, d_star = int(m.group(4)), int(m.group(5)), int(m.group(2))
        if d < min_width:
            continue
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if r["kind"] != kind or r["metric"] != metric or abs(float(r["threshold"]) - thr) > 1e-12:
                    continue
                step = int(r["step"])
                rank[(d, step)][seed] = float(r["rank"]) if r["reached"] == "True" else np.nan
                loss[(d, step)][seed] = float(r["loss_test"])
    return rank, loss, d_star


def stats(vals: dict) -> dict:
    a = np.asarray(list(vals.values()), float)
    ok = a[np.isfinite(a)]
    n = ok.size
    return {"n": n, "n_total": a.size,
            "mean": ok.mean() if n else np.nan,
            "median": float(np.median(ok)) if n else np.nan,
            "std": ok.std(ddof=1) if n > 1 else (0.0 if n == 1 else np.nan)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rank, loss, d_star = load(args.root_dir, "tau", TAU, METRIC, MIN_WIDTH)
    if not rank:
        raise SystemExit(f"no temporal data under {args.root_dir}")
    widths = sorted({d for d, _ in rank})
    steps_all = sorted({s for _, s in rank})
    missing = [s for s in STEPS if s not in steps_all]
    if missing:
        raise SystemExit(f"steps {missing} not in the data; available: {steps_all}")
    S = {k: stats(v) for k, v in rank.items()}
    L = {k: float(np.mean(list(v.values()))) for k, v in loss.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok = f"tau{TAU:g}".replace(".", "p")
    ylab = rf"$r_\tau$, $\tau={TAU:g}$"

    # ---- (a) rank vs width at checkpoints -----------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    cols = plt.cm.viridis(np.linspace(0.0, 0.9, len(STEPS)))
    handles = []
    for step, c in zip(STEPS, cols):
        x = np.array(widths, float)
        st = [S[(d, step)] for d in widths]
        y = np.array([s["mean"] for s in st])
        e = np.array([s["std"] for s in st])
        full = np.array([s["n"] == s["n_total"] for s in st])
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=c, lw=1.2, zorder=1)
        ax.errorbar(x[ok], y[ok], yerr=np.nan_to_num(e[ok]), fmt="none", ecolor=c,
                    elinewidth=1.0, capsize=ERR_CAP, alpha=ERR_ALPHA, zorder=0)
        for msk, face in ((ok & full, c), (ok & ~full, "white")):
            if msk.any():
                ax.plot(x[msk], y[msk], ls="none", marker="o", ms=4.0, color=c, mfc=face,
                        mew=1.0, zorder=2)
        handles.append((Line2D([0], [0], color=c, lw=1.2, marker="o", ms=4.0), f"$t={step}$"))
    ax.axvline(d_star, ls=":", lw=1.5, color="black")
    handles.append((Line2D([0], [0], color="black", lw=1.5, ls=":"), f"teacher $d^*={d_star}$"))
    ax.set_xticks(WIDTH_TICKS)
    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel(ylab)
    # Room below the curves for the legend, so it never sits on the data.
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend([h for h, _ in handles], [l for _, l in handles], loc="lower right", ncol=2,
              columnspacing=0.8, frameon=False, handlelength=1.4, handletextpad=0.4, labelspacing=0.15,
              borderpad=0.2, borderaxespad=0.3)
    saved = save_figure(fig, args.out_dir / f"rank_vs_width_by_step_{tok}_test")

    # Colour by width on the same scale as the eigenvalue spectrum plot.
    norm = plt.Normalize(min(widths), max(widths))

    # ---- (b) rank over training -----------------------------------------------
    # No legend here; the widths are named in the legend of (c).
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for d in LINE_WIDTHS:
        c = plt.cm.viridis(norm(d))
        st = [S[(d, s)] for s in steps_all]
        x = np.array(steps_all, float)
        y = np.array([s["mean"] for s in st])
        e = np.nan_to_num(np.array([s["std"] for s in st]))
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=c, lw=1.2)
        ax.fill_between(x[ok], (y - e)[ok], (y + e)[ok], color=c, alpha=0.15, lw=0)
    ax.set_ylim(60, 200)
    ax.set_xlabel("Training step")
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_ylabel(ylab)
    saved += save_figure(fig, args.out_dir / f"rank_vs_step_{tok}_test")

    # ---- (c) test MSE over training --------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for d in LINE_WIDTHS:
        ax.plot(steps_all, [L[(d, s)] for s in steps_all], color=plt.cm.viridis(norm(d)), lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(ax.get_ylim()[0], 10.0)   # one decade of headroom for the legend
    ax.set_xlabel("Training step")
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_ylabel("test MSE")
    ax.legend([Line2D([0], [0], color=plt.cm.viridis(norm(d)), lw=2.0) for d in LINE_WIDTHS],
              [f"$d={d}$" for d in LINE_WIDTHS], loc="upper right", ncol=3,
              frameon=False, handlelength=1.2, handletextpad=0.4, labelspacing=0.1,
              columnspacing=0.8, borderpad=0.2, borderaxespad=0.2)
    saved += save_figure(fig, args.out_dir / f"test_mse_vs_step_{tok}_test")

    # ---- CSV -----------------------------------------------------------------
    csv_path = args.out_dir / f"temporal_{tok}_test.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["width", "step", "rank_mean", "rank_median", "rank_std", "n_reached",
                    "n_seeds", "test_mse_mean"])
        for d in widths:
            for s in steps_all:
                st = S[(d, s)]
                w.writerow([d, s, st["mean"], st["median"], st["std"], st["n"], st["n_total"],
                            L[(d, s)]])
    for f in saved + [str(csv_path)]:
        print("wrote", f)


if __name__ == "__main__":
    main()
