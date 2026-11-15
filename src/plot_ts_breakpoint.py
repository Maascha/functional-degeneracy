"""Recovery rank against width with an estimated breakpoint.

Same data as the fixed-breakpoint rank-vs-width plot, but the break of the two-segment regression is not
fixed at the teacher width d*. A continuous two-segment linear function
(hinge) is fitted by least squares to the per-width means of all thresholds
for every candidate breakpoint d_b, on a grid of 400 points between the
smallest and largest width plus the observed widths, and the d_b with the
smallest SSE is kept. A 90% confidence interval comes from 200 bootstrap
resamples: the seeds are resampled with replacement independently at each
width and for each threshold, and the search is repeated on the resampled
means.

Writes paper_ts_rank_piecewise_fit_estimated_breakpoint_<split>_thresholds_...
(top: fit, breakpoint and CI; bottom: SSE per candidate) and
estimated_breakpoint_fit.json.

    python src/plot_ts_breakpoint.py --root-dir results/ts_in10_out10 --probe-source train \
        --pvec-mode absolute --split test --out-dir figures/ts/in10_out10/breakpoint
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ts_plot_utils import (
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


FIT_MIN_POINTS = 2       # points required on each side of a candidate breakpoint
GRID_POINTS = 400
N_BOOTSTRAP = 200
CI_LEVEL = 0.90
BOOTSTRAP_SEED = 0


def _fit_continuous_hinge(x: np.ndarray, y: np.ndarray, bp: float, min_points: int) -> dict[str, Any] | None:
    """y = a + b*x + c*relu(x - bp) by least squares; the two lines meet at x = bp."""
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    n_left = int(np.sum(xf <= bp))
    n_right = int(np.sum(xf >= bp))
    if n_left < min_points or n_right < min_points:
        return None
    design = np.column_stack([np.ones_like(xf), xf, np.clip(xf - bp, 0.0, None)])
    coef, _, rank_, _ = np.linalg.lstsq(design, yf, rcond=None)
    if rank_ < design.shape[1]:
        return None
    a, b, c = (float(v) for v in coef)
    return {
        "m_left": b,
        "b_left": a,
        "m_right": b + c,
        "b_right": a - c * bp,
        "sse": float(np.sum((yf - design @ coef) ** 2)),
        "n_left": n_left,
        "n_right": n_right,
    }


def _search_breakpoint(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Grid search over candidate breakpoints, keeping the fit with the lowest SSE."""
    xf = x[np.isfinite(x) & np.isfinite(y)]
    if xf.size == 0:
        raise ValueError("No finite (d, rank) pairs to search a breakpoint over.")
    lo, hi = float(np.min(xf)), float(np.max(xf))
    if not (hi > lo):
        raise ValueError("Need at least two distinct widths to estimate a breakpoint.")
    candidates = np.unique(np.concatenate([np.linspace(lo, hi, GRID_POINTS), np.unique(xf)]))

    fits = [_fit_continuous_hinge(x, y, float(bp), FIT_MIN_POINTS) for bp in candidates]
    sse = np.asarray([f["sse"] if f is not None else np.nan for f in fits], dtype=float)
    valid = np.where(np.isfinite(sse))[0]
    if valid.size == 0:
        raise ValueError(f"No candidate breakpoint leaves >= {FIT_MIN_POINTS} points on both sides.")
    best_i = int(valid[np.argmin(sse[valid])])
    return {"breakpoint": float(candidates[best_i]), **fits[best_i],
            "candidates": candidates.astype(float), "sse_curve": sse}


def _bootstrap_breakpoint_ci(payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Percentile interval of the breakpoint over bootstrap resamples of the seeds."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(N_BOOTSTRAP):
        x_all: list[float] = []
        y_all: list[float] = []
        for p in payloads:
            for d, vals in p["by_d"].items():
                arr = np.asarray(vals, dtype=float)
                if arr.size == 0:
                    continue
                x_all.append(float(d))
                y_all.append(float(np.mean(rng.choice(arr, size=arr.size, replace=True))))
        if len(x_all) < 2:
            continue
        try:
            estimates.append(_search_breakpoint(np.asarray(x_all, dtype=float),
                                                np.asarray(y_all, dtype=float))["breakpoint"])
        except ValueError:
            continue
    if not estimates:
        return None
    est = np.asarray(estimates, dtype=float)
    alpha = (1.0 - CI_LEVEL) / 2.0
    return {
        "n_used": int(est.size),
        "ci_level": CI_LEVEL,
        "ci_low": float(np.percentile(est, 100.0 * alpha)),
        "ci_high": float(np.percentile(est, 100.0 * (1.0 - alpha))),
        "median": float(np.median(est)),
    }


def _plot_rank_with_estimated_breakpoint(payloads, bp_fit, bootstrap, d_star_ref, out_dir, split,
                                         hollow_partial: bool) -> list[str]:
    tau_vals, q_vals, tau_colors, q_colors = threshold_colors(payloads)

    fig, (ax, ax_sse) = plt.subplots(
        2, 1, figsize=(PAPER_FIGSIZE[0] + LEGEND_W, PAPER_FIGSIZE[1] * 1.5),
        height_ratios=[3, 1], sharex=False,
    )
    fig.subplots_adjust(right=PAPER_FIGSIZE[0] / (PAPER_FIGSIZE[0] + LEGEND_W), hspace=0.45)

    draw_rank_curves(ax, payloads, tau_colors, q_colors, hollow_partial)

    bp_hat = float(bp_fit["breakpoint"])
    d_union = sorted({int(d) for p in payloads for d in p["d_vals"]})
    x0, yhat0, left0, right0 = piecewise_line(bp_fit, float(min(d_union)), float(max(d_union)), bp_hat)
    ax.plot(x0[left0], yhat0[left0], linestyle="-", linewidth=2.4, color="black", zorder=3)
    ax.plot(x0[right0], yhat0[right0], linestyle="-", linewidth=2.4, color="black", zorder=3)

    if bootstrap is not None:
        ax.axvspan(bootstrap["ci_low"], bootstrap["ci_high"], color="black", alpha=0.08, zorder=0)
    ax.axvline(bp_hat, linestyle="--", linewidth=1.5, color="black")
    if d_star_ref is not None:
        ax.axvline(float(d_star_ref), linestyle=":", linewidth=1.2, color="0.5")

    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel(r"$\hat r_q$ or $r_\tau$")
    # Ticks through d* (when known), as in the fixed-breakpoint plot.
    ticks = anchored_ticks(ax, float(d_star_ref) if d_star_ref is not None else float(round(bp_hat)))

    empty = Line2D([], [], linestyle="none")
    annot: list[tuple[Any, str]] = [
        (empty, r"$\mathbf{Fit}$"),
        (Line2D([0], [0], color="black", linewidth=2.4, linestyle="-"), "two-segment fit"),
        (Line2D([0], [0], color="black", linewidth=1.5, linestyle="--"), r"breakpoint $\hat d_b$"),
    ]
    if bootstrap is not None:
        annot.append((plt.Rectangle((0, 0), 1, 1, facecolor="black", alpha=0.08),
                      f"{int(round(bootstrap['ci_level'] * 100))}% CI of $\\hat d_b$"))
    if d_star_ref is not None:
        annot.append((Line2D([0], [0], color="0.5", linewidth=1.2, linestyle=":"), r"teacher $d^*$"))
    side_legend(ax, threshold_legend_entries(tau_vals, q_vals, tau_colors, q_colors) + annot)

    # Bottom panel: SSE of the best fit at each candidate breakpoint.
    order = np.argsort(bp_fit["candidates"])
    ax_sse.plot(bp_fit["candidates"][order], bp_fit["sse_curve"][order], color="black", linewidth=1.2)
    ax_sse.axvline(bp_hat, linestyle="--", linewidth=1.2, color="black", alpha=0.6)
    if bootstrap is not None:
        ax_sse.axvspan(bootstrap["ci_low"], bootstrap["ci_high"], color="black", alpha=0.08, zorder=0)
    if d_star_ref is not None:
        ax_sse.axvline(float(d_star_ref), linestyle=":", linewidth=1.0, color="0.5")
    ax_sse.set_xlim(ax.get_xlim())
    if ticks:
        ax_sse.set_xticks(ticks)
    ax_sse.set_xlabel(r"Candidate breakpoint $d_b$")
    ax_sse.set_ylabel("SSE")

    base = out_dir / f"paper_ts_rank_piecewise_fit_estimated_breakpoint_{split}_thresholds_{threshold_tag(payloads)}_vs_width"
    return save_figure(fig, base)


def _json_scalar(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()

    records_variant, payloads, split = load_threshold_payloads(args)
    d_star_ref = teacher_d_star(records_variant)   # drawn for reference only

    x_all, y_all = pooled_points(payloads)
    bp_fit = _search_breakpoint(x_all, y_all)
    bootstrap = _bootstrap_breakpoint_ci(payloads)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.out_dir / "estimated_breakpoint_fit.json"
    with fit_path.open("w") as fh:
        json.dump({
            "split": str(split),
            "relative_ref": str(args.relative_ref),   # reference loss of the relative ranks
            "breakpoint_model": "continuous",
            "d_star_reference": (float(d_star_ref) if d_star_ref is not None else None),
            "taus": sorted({p["value"] for p in payloads if p["kind"] == "tau"}),
            "qs": sorted({p["value"] for p in payloads if p["kind"] == "q"}),
            "breakpoint_hat": _json_scalar(bp_fit["breakpoint"]),
            "sse": _json_scalar(bp_fit["sse"]),
            "n_left": int(bp_fit["n_left"]),
            "n_right": int(bp_fit["n_right"]),
            **{k: _json_scalar(bp_fit[k]) for k in ("m_left", "b_left", "m_right", "b_right")},
            "bootstrap": (None if bootstrap is None else {
                "n_used": bootstrap["n_used"],
                "ci_level": bootstrap["ci_level"],
                "ci_low": _json_scalar(bootstrap["ci_low"]),
                "ci_high": _json_scalar(bootstrap["ci_high"]),
                "median": _json_scalar(bootstrap["median"]),
            }),
        }, fh, indent=1)

    paths = finalize_paths(
        _plot_rank_with_estimated_breakpoint(payloads, bp_fit, bootstrap, d_star_ref, args.out_dir, split,
                                             hollow_partial=(args.relative_ref == "theta_ref")), args)

    print(f"Estimated breakpoint d_b = {bp_fit['breakpoint']:.4g} (SSE={bp_fit['sse']:.6g}, "
          f"n_left={bp_fit['n_left']}, n_right={bp_fit['n_right']})")
    if bootstrap is not None:
        print(f"Bootstrap {int(round(CI_LEVEL * 100))}% CI (n={bootstrap['n_used']}): "
              f"[{bootstrap['ci_low']:.4g}, {bootstrap['ci_high']:.4g}], median={bootstrap['median']:.4g}")
    if d_star_ref is not None:
        print(f"Teacher d* (reference only, not used in the fit): {d_star_ref}")
    print(f"wrote {fit_path}")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
