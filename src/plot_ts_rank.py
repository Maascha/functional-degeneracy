"""Behavioral recovery rank against width with a piecewise regression broken at d*.

Writes, for one (probe set, reference point) variant of a teacher-student sweep:
  - paper_ts_rank_piecewise_fit_with_mean_std_<split>_thresholds_...: mean rank over
    seeds per width for every tau and q (error bars: std), and one two-segment
    linear regression over the means of all thresholds, broken at the teacher
    width d*;
  - paper_ts_<split>_mse_heatmap_...: the mean projection loss over (k, d) with
    the same regression overlaid;
  - piecewise_fit.json: the regression coefficients (read by prune_vs_recovery_rank.py).

Relative ranks use --relative-ref (default theta_ref; see
ts_rank_data). With theta_ref a hollow marker means that only some seeds reach
the tolerance at that width.

    python src/plot_ts_rank.py --root-dir results/ts_in10_out10 --probe-source train \
        --pvec-mode absolute --split test --out-dir figures/ts/in10_out10/rank
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ts_rank_data import maybe_int
from ts_plot_utils import (
    TICK_FS,
    LEGEND_W,
    PAPER_FIGSIZE,
    add_common_args,
    anchored_ticks,
    draw_rank_curves,
    finalize_paths,
    load_threshold_payloads,
    piecewise_line,
    pooled_points,
    save_figure,
    side_legend,
    teacher_d_star,
    threshold_colors,
    threshold_legend_entries,
    threshold_tag,
)


def _mean_curve_matrix(records: list[dict[str, Any]], split: str) -> tuple[list[int], np.ndarray]:
    """Mean over seeds of M(theta_hat_k) on ``split``: rows k = 1, 2, ..., one column per width."""
    by_d: dict[int, list[np.ndarray]] = defaultdict(list)
    d_all: set[int] = set()
    for r in records:
        d = maybe_int(r.get("d"))
        if d is None:
            continue
        d_all.add(int(d))
        arr = (r.get("_curves") or {}).get(split)
        if isinstance(arr, np.ndarray) and len(arr):
            by_d[int(d)].append(np.asarray(arr, dtype=float))
    if not by_d:
        return [], np.zeros((0, 0), dtype=float)

    d_sorted = sorted(d_all)
    max_len = max(len(c) for curves in by_d.values() for c in curves)
    mat = np.full((max_len, len(d_sorted)), np.nan, dtype=float)
    for j, d in enumerate(d_sorted):
        if not by_d.get(d):
            continue
        stacked = np.full((len(by_d[d]), max_len), np.nan, dtype=float)
        for i, c in enumerate(by_d[d]):
            stacked[i, : len(c)] = c
        mat[:, j] = np.nanmean(stacked, axis=0)
    return d_sorted, mat


def plot_flipped_heatmap_with_piecewise_fit(
    records_variant: list[dict[str, Any]],
    d_union: list[int],
    piecewise: dict[str, Any],
    d_star: int,
    out_dir: Path,
    split: str,
    tag: str,
    vmax_cap: float = 0.3,
) -> list[str]:
    """Loss heatmap with x = k and y = d, and the shared piecewise regression of the rank.

    The colour scale is logarithmic from the 5th percentile of the values to
    ``vmax_cap``; values above it share the top colour.
    """
    d_sorted, mat = _mean_curve_matrix(records_variant, split)
    if not d_sorted:
        return []

    masked = np.ma.masked_where(~np.isfinite(mat) | (mat <= 0), mat)
    mat_flip = masked.T
    finite_by_k = np.any(np.isfinite(mat_flip.filled(np.nan)), axis=0)
    if not np.any(finite_by_k):
        return []
    mat_flip = mat_flip[:, : int(np.where(finite_by_k)[0][-1]) + 1]

    positive = mat_flip.compressed()
    positive = positive[positive > 0]
    vmin = float(np.percentile(positive, 5))
    vmax = max(float(vmax_cap), vmin * 1.01)

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    y_pos = np.arange(len(d_sorted), dtype=float)
    im = ax.imshow(
        mat_flip,
        origin="lower",
        aspect="auto",
        extent=[1.0, float(mat_flip.shape[1]), -0.5, len(d_sorted) - 0.5],
        interpolation="nearest",
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    # Label placed low on the bar and over the tick column to save width.
    cbar.set_label(f"{split} MSE", fontsize=TICK_FS, rotation=90, y=0.28, labelpad=-24)

    # Regression of the rank (x) against width (y), with widths mapped to row indices.
    x0, yhat0, _, _ = piecewise_line(piecewise, float(min(d_union)), float(max(d_union)), float(d_star))
    rows0 = np.interp(x0, d_sorted, np.arange(len(d_sorted)))
    left = np.isfinite(yhat0) & (x0 <= float(d_star))
    right = np.isfinite(yhat0) & (x0 >= float(d_star))
    ax.plot(yhat0[left], rows0[left], linestyle="-", linewidth=2.4, color="black", zorder=3)
    ax.plot(yhat0[right], rows0[right], linestyle="-", linewidth=2.4, color="black", zorder=3)

    if d_star in d_sorted:
        ax.axhline(float(d_sorted.index(d_star)), linestyle=":", linewidth=1.5, color="black")

    tick_idx = list(range(0, len(d_sorted), 2))
    if len(d_sorted) > 1 and (len(d_sorted) - 1) not in tick_idx:
        tick_idx.append(len(d_sorted) - 1)
    ax.set_yticks([y_pos[i] for i in tick_idx])
    ax.set_yticklabels([str(d_sorted[i]) for i in tick_idx])
    ax.set_xlabel(r"Leading eigendirections $k$")
    ax.set_ylabel(r"Hidden width $d$")
    return save_figure(fig, out_dir / f"paper_ts_{split}_mse_heatmap_explode_allruns_{tag}")


FIT_MIN_POINTS = 2      # fewer points on a side: that side is a constant at the median


def _fit_line(xs: np.ndarray, ys: np.ndarray, min_points: int) -> tuple[float, float]:
    finite = np.isfinite(xs) & np.isfinite(ys)
    x, y = xs[finite], ys[finite]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size < max(1, int(min_points)) or np.allclose(x, x[0]):
        return 0.0, float(np.nanmedian(y))
    m, b = np.polyfit(x, y, deg=1)
    return float(m), float(b)


def piecewise_fit(d_vals: np.ndarray, rank_vals: np.ndarray, d_star: int, min_points: int) -> dict[str, float]:
    """Independent least-squares lines for d <= d* and d >= d* (points at d* enter both)."""
    ok = np.isfinite(d_vals) & np.isfinite(rank_vals)
    left = ok & (d_vals <= float(d_star))
    right = ok & (d_vals >= float(d_star))
    m_left, b_left = _fit_line(d_vals[left], rank_vals[left], min_points)
    m_right, b_right = _fit_line(d_vals[right], rank_vals[right], min_points)
    return {"m_left": m_left, "b_left": b_left, "m_right": m_right, "b_right": b_right}


def _json_scalar(v: Any) -> float | None:
    """Float for JSON; NaN/inf become null."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _plot_rank(payloads: list[dict[str, Any]], fit: dict[str, float], d_star: int, out_dir, split: str,
               hollow_partial: bool) -> list[str]:
    tau_vals, q_vals, tau_colors, q_colors = threshold_colors(payloads)

    # Canvas wider than the other panels by the legend column, so the axes
    # have the same size as theirs.
    fig, ax = plt.subplots(figsize=(PAPER_FIGSIZE[0] + LEGEND_W, PAPER_FIGSIZE[1]))
    fig.subplots_adjust(right=PAPER_FIGSIZE[0] / (PAPER_FIGSIZE[0] + LEGEND_W))

    draw_rank_curves(ax, payloads, tau_colors, q_colors, hollow_partial)

    d_union = sorted({int(d) for p in payloads for d in p["d_vals"]})
    x0, yhat0, left0, right0 = piecewise_line(fit, float(min(d_union)), float(max(d_union)), float(d_star))
    ax.plot(x0[left0], yhat0[left0], linestyle="-", linewidth=2.4, color="black", zorder=3)
    ax.plot(x0[right0], yhat0[right0], linestyle="-", linewidth=2.4, color="black", zorder=3)

    ax.axvline(float(d_star), linestyle=":", linewidth=1.5, color="black")
    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel(r"$\hat r_q$ or $r_\tau$")
    anchored_ticks(ax, d_star)

    entries = threshold_legend_entries(tau_vals, q_vals, tau_colors, q_colors) + [
        (Line2D([0], [0], color="black", linewidth=2.4, linestyle="-"), "regression on\n$\\hat r_q$/$r_\\tau$"),
        (Line2D([0], [0], color="black", linewidth=1.5, linestyle=":"), f"teacher\n$d^*={d_star}$"),
    ]
    side_legend(ax, entries)

    base = out_dir / f"paper_ts_rank_piecewise_fit_with_mean_std_{split}_thresholds_{threshold_tag(payloads)}_vs_width"
    return save_figure(fig, base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()

    records_variant, payloads, split = load_threshold_payloads(args)
    d_star = teacher_d_star(records_variant)
    if d_star is None:
        raise SystemExit("Unable to infer the teacher width d* from the records.")

    x_all, y_all = pooled_points(payloads)
    fit = piecewise_fit(x_all, y_all, d_star, FIT_MIN_POINTS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.out_dir / "piecewise_fit.json"
    with fit_path.open("w") as fh:
        json.dump({
            "d_star": float(d_star),
            "split": str(split),
            "relative_ref": str(args.relative_ref),   # reference loss of the relative ranks
            "taus": sorted({p["value"] for p in payloads if p["kind"] == "tau"}),
            "qs": sorted({p["value"] for p in payloads if p["kind"] == "q"}),
            **{k: _json_scalar(fit[k]) for k in ("m_left", "b_left", "m_right", "b_right")},
        }, fh, indent=1)

    paths = _plot_rank(payloads, fit, d_star, args.out_dir, split,
                       hollow_partial=(args.relative_ref == "theta_ref"))
    paths += plot_flipped_heatmap_with_piecewise_fit(
        records_variant=records_variant,
        d_union=sorted({int(d) for p in payloads for d in p["d_vals"]}),
        piecewise=fit,
        d_star=d_star,
        out_dir=args.out_dir,
        split=split,
        tag=threshold_tag(payloads),
    )
    paths = finalize_paths(paths, args)

    print("Rank keys used:")
    for p in payloads:
        print(f"  {p['kind']}={p['value']:g}: {p['rank_key']}")
    print(f"d* used: {d_star}")
    print(f"  left:  r = {fit['m_left']:.6g} * d + {fit['b_left']:.6g}")
    print(f"  right: r = {fit['m_right']:.6g} * d + {fit['b_right']:.6g}")
    print(f"wrote {fit_path}")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
