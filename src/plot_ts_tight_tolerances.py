"""Recovery rank at tighter tolerances.

Same data as the rank-vs-width plot (final checkpoint, test MSE, probe = train,
theta_ref = 0) at tighter thresholds:

  rank vs width for tau in {0.01, 0.025, 0.05, 0.1} and q in {0.8, 0.9, 0.95, 0.98},
  plus the mean number of retained eigenpairs (lambda_j >= 1e-6 lambda_1). Mean
  over the seeds that reach the tolerance, error bar = std (ddof=1); hollow
  marker when fewer than all seeds reach it.

  A second panel (<name>_seeds_reaching) shows how many seeds reach each tolerance.
  --relative-ref sets the reference loss of the relative ranks (see ts_rank_data);
  the default theta_ref is M(0) here.

    python src/plot_ts_tight_tolerances.py --root-dir results/ts_in10_out10 \
        --out-dir figures/ts/in10_out10/tight_tolerances
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import ts_rank_data as _trd
from ts_rank_data import STABLE_EIG_COL, maybe_float, maybe_int, sanitize_float_token
from ts_plot_utils import PAPER_FIGSIZE, filter_variant, save_figure

TAUS = [0.01, 0.025, 0.05, 0.1]
QS = [0.8, 0.9, 0.95, 0.98]
MIN_WIDTH = 3
SPLIT = "test"
LEGEND_W = 1.2          # legend column, same panel geometry as the rank-vs-width plot
TICKS = [6, 10, 14, 18]
ERR_ALPHA, ERR_CAP = 0.25, 2.0


def shades(cmap_name: str, n: int) -> list:
    """n shades, darkest first."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(v) for v in np.linspace(0.95, 0.45, n)]


def series(records, key, widths, n_total):
    out = []
    for d in widths:
        vals = [maybe_float(r.get(key)) for r in records if maybe_int(r.get("d")) == d]
        vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        n = int(vals.size)
        out.append({
            "d": d, "n": n, "n_total": n_total[d],
            "mean": float(vals.mean()) if n else np.nan,
            "std": float(vals.std(ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan),
        })
    return out


def draw_series(ax, s, color, ls, mk):
    x = np.array([p["d"] for p in s], float)
    y = np.array([p["mean"] for p in s], float)
    e = np.array([p["std"] for p in s], float)
    full = np.array([p["n"] == p["n_total"] for p in s])
    ok = np.isfinite(y)
    ax.plot(x[ok], y[ok], color=color, ls=ls, lw=1.2, zorder=1)
    if np.any(ok & (e > 0)):
        ax.errorbar(x[ok], y[ok], yerr=np.nan_to_num(e[ok]), fmt="none", ecolor=color,
                    elinewidth=1.0, capsize=ERR_CAP, alpha=ERR_ALPHA, zorder=0)
    for m, face in ((ok & full, color), (ok & ~full, "white")):
        if np.any(m):
            ax.plot(x[m], y[m], ls="none", marker=mk, ms=4.0, color=color,
                    mfc=face, mew=1.0, zorder=2)


def finish_x(ax, d_star):
    ax.axvline(d_star, ls=":", lw=1.5, color="black")
    ax.set_xticks(TICKS)
    ax.set_xlabel(r"Hidden width $d$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", default="tight_tolerances",
                    help="Output file stem, e.g. tight_tolerances_noise.")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="Fix the rank panel's y-range, e.g. to match another setup.")
    ap.add_argument("--no-seeds-panel", action="store_true",
                    help="Skip the seeds-reaching panel.")
    _trd.add_relative_ref_arg(ap)
    args = ap.parse_args()

    records = _trd.gather_records(
        root_dir=args.root_dir, test_thresholds=TAUS, relative_q=QS, max_k=None,
        exclude_widths=set(range(MIN_WIDTH)), relative_ref=args.relative_ref)
    records = filter_variant(records, probe_source="train",
                             pvec_mode="absolute")
    if not records:
        raise SystemExit(f"no records under {args.root_dir}")
    d_star = int(maybe_int(records[0].get("teacher_hidden_dim")))
    widths = sorted({maybe_int(r["d"]) for r in records if maybe_int(r.get("d")) >= MIN_WIDTH})
    n_total = defaultdict(int)
    for r in records:
        n_total[maybe_int(r["d"])] += 1

    taus = sorted(TAUS)                # tightest first
    qs = sorted(QS, reverse=True)      # tightest first
    specs = ([("tau", t, f"k_{SPLIT}_tau_{sanitize_float_token(t)}", c, "-", "o")
              for t, c in zip(taus, shades("Blues", len(taus)))]
             + [("q", q, f"r_{SPLIT}_q_{sanitize_float_token(q)}", c, "--", "s")
                for q, c in zip(qs, shades("Reds", len(qs)))])
    data = {(k, v): series(records, key, widths, n_total) for k, v, key, *_ in specs}
    eig = series(records, STABLE_EIG_COL, widths, n_total)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- (a) rank vs width -------------------------------------------------
    fig, ax = plt.subplots(figsize=(PAPER_FIGSIZE[0] + LEGEND_W, PAPER_FIGSIZE[1]))
    fig.subplots_adjust(right=PAPER_FIGSIZE[0] / (PAPER_FIGSIZE[0] + LEGEND_W))
    ex = np.array([p["d"] for p in eig], float)
    ey = np.array([p["mean"] for p in eig], float)
    ax.plot(ex, ey, color="0.55", lw=1.4, zorder=0.5)
    for k, v, _, c, ls, mk in specs:
        draw_series(ax, data[(k, v)], c, ls, mk)
    finish_x(ax, d_star)
    ax.set_ylabel(r"$\hat r_q$ or $r_\tau$")
    if args.ylim:
        ax.set_ylim(*args.ylim)
    print(f"rank panel ylim: {ax.get_ylim()}")

    empty = Line2D([], [], ls="none")
    ent = [(empty, r"$\mathbf{Absolute}$")]
    ent += [(Line2D([0], [0], color=c, lw=2.0, ls=ls, marker=mk, ms=5.0), rf"$\tau={v:g}$")
            for k, v, _, c, ls, mk in specs if k == "tau"]
    ent += [(empty, r"$\mathbf{Relative}$")]
    ent += [(Line2D([0], [0], color=c, lw=2.0, ls=ls, marker=mk, ms=5.0), rf"$q={v:g}$")
            for k, v, _, c, ls, mk in specs if k == "q"]
    ent += [(Line2D([0], [0], color="0.55", lw=1.4), "retained\neigendirections"),
            (Line2D([0], [0], color="black", lw=1.5, ls=":"), f"teacher\n$d^*={d_star}$")]
    ax.legend([h for h, _ in ent], [l for _, l in ent], loc="upper left",
              bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.15, borderpad=0.2)
    saved = save_figure(fig, args.out_dir / f"{args.name}_rank_vs_width")

    # ---- (b) seeds reaching ------------------------------------------------
    if not args.no_seeds_panel:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        for k, v, _, c, ls, mk in specs:
            s = data[(k, v)]
            ax.plot([p["d"] for p in s], [p["n"] for p in s], color=c, ls=ls, lw=1.2,
                    marker=mk, ms=4.0)
        finish_x(ax, d_star)
        nmax = max(n_total.values())
        ax.set_yticks(range(nmax + 1))
        ax.set_ylim(-0.3, nmax + 0.3)
        ax.set_ylabel("seeds reaching tolerance")
        saved += save_figure(fig, args.out_dir / f"{args.name}_seeds_reaching")

    # ---- CSV ---------------------------------------------------------------
    csv_path = args.out_dir / f"{args.name}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["width", "kind", "threshold", "mean_rank", "std_rank", "n_reached",
                    "n_seeds", "retained_eigenpairs_mean", "relative_ref"])
        for k, v, *_ in specs:
            for p, e in zip(data[(k, v)], eig):
                w.writerow([p["d"], k, v, p["mean"], p["std"], p["n"], p["n_total"], e["mean"],
                            args.relative_ref])
    for f in saved + [str(csv_path)]:
        print("wrote", f)


if __name__ == "__main__":
    main()
