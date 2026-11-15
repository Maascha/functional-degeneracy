"""Shared figure style and helpers for the teacher-student plots.

Importing this module sets the shared rcParams (TrueType fonts, font sizes).

The second part holds the pieces shared by the recovery-rank-vs-width plots
(plot_ts_rank.py: break fixed at the teacher width d*; plot_ts_breakpoint.py:
break estimated from the data). For each threshold (absolute tau and relative q)
the rank of every seed is read from the run records and averaged per width
(mean, std with ddof=1 as the error bar); the scripts then fit a two-segment
linear regression over the per-width means of all thresholds together.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import ts_rank_data as _trd
from ts_rank_data import maybe_float, maybe_int, sanitize_float_token

# (probe_source, projection_vector_mode) variants of the probe-set /
# reference-point ablation; "absolute" is theta_ref = 0, "displacement" theta_ref = theta_0.
VARIANTS: list[tuple[str, str]] = [
    ("train", "displacement"),
    ("train", "absolute"),
    ("test", "displacement"),
    ("test", "absolute"),
]

# Figures are drawn at twice their placed size: small panels (three across,
# ~1.7in each) at 3.4in, large "appendix" panels (two across, ~2.64in each) at 5.3in.
# With the same drawn-to-placed ratio, the point sizes below land at the same
# size on the page in both cases.
PAPER_FIGSIZE_MAIN = (3.4, 2.9)
PAPER_FIGSIZE_APPENDIX = (5.3, 4.5)
PAPER_FIGSIZE = PAPER_FIGSIZE_MAIN


LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 12

plt.rcParams.update(
    {
        # TrueType rather than Type 3 fonts.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": LABEL_FS,
        "xtick.labelsize": TICK_FS,
        "ytick.labelsize": TICK_FS,
        "legend.fontsize": LEGEND_FS,
    }
)


def save_figure(fig: plt.Figure, base_path_no_ext: Path) -> list[str]:
    """Save as .png and .pdf (tight bounding box) and close the figure."""
    png_path = base_path_no_ext.with_suffix(".png")
    pdf_path = base_path_no_ext.with_suffix(".pdf")
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [str(png_path), str(pdf_path)]


def add_suffix_to_plot_paths(plot_paths: list[str], suffix: str) -> list[str]:
    """Rename saved files, appending suffix to the stem."""
    updated: list[str] = []
    for p in plot_paths:
        src = Path(p)
        dst = src.with_name(f"{src.stem}{suffix}{src.suffix}")
        src.rename(dst)
        updated.append(str(dst))
    return updated


def filter_variant(records: list[dict[str, Any]], probe_source: str, pvec_mode: str) -> list[dict[str, Any]]:
    return [
        r
        for r in records
        if str(r.get("probe_source")) == probe_source and str(r.get("projection_vector_mode")) == pvec_mode
    ]


def teacher_d_star(records: list[dict[str, Any]]) -> int | None:
    for r in records:
        v = maybe_int(r.get("teacher_hidden_dim"))
        if v is not None:
            return v
    return None


def aggregate_main_rows_by_d(records: list[dict[str, Any]], scalar_keys: list[str]) -> list[dict[str, Any]]:
    """Median and interquartile half-ranges of each key over the seeds at each width d.

    Expects the records of a single (probe_source, projection_vector_mode) variant.
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        d = maybe_int(r.get("d"))
        if d is not None:
            grouped[d].append(r)

    out: list[dict[str, Any]] = []
    for d, grp in sorted(grouped.items()):
        row: dict[str, Any] = {"d": int(d)}
        for key in scalar_keys:
            vals = [maybe_float(x.get(key)) for x in grp]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                arr = np.asarray(vals, dtype=float)
                med = float(np.median(arr))
                row[key] = med
                row[f"{key}_iqr_low"] = med - float(np.percentile(arr, 25))
                row[f"{key}_iqr_high"] = float(np.percentile(arr, 75)) - med
            else:
                row[key] = row[f"{key}_iqr_low"] = row[f"{key}_iqr_high"] = math.nan
        out.append(row)
    return out


def _eb(
    ax: plt.Axes,
    xs: list[int],
    rows_agg: list[dict[str, Any]],
    y_key: str,
    label: str,
    color: Any = None,
    yerr_low_key: str | None = None,
    yerr_high_key: str | None = None,
) -> bool:
    """Plot y_key with faint error bars behind it. Returns True if anything was plotted."""
    ys = np.asarray([maybe_float(r.get(y_key)) if maybe_float(r.get(y_key)) is not None else np.nan for r in rows_agg], dtype=float)
    lo_key = yerr_low_key or f"{y_key}_iqr_low"
    hi_key = yerr_high_key or f"{y_key}_iqr_high"
    lo = np.asarray([maybe_float(r.get(lo_key)) or 0.0 for r in rows_agg], dtype=float)
    hi = np.asarray([maybe_float(r.get(hi_key)) or 0.0 for r in rows_agg], dtype=float)
    if not np.any(np.isfinite(ys)):
        return False
    kw: dict[str, Any] = {"marker": "o", "markersize": 4.0, "linewidth": 1.2,
                          "label": label, "zorder": 2}
    if color is not None:
        kw["color"] = color
    ax.plot(xs, ys, **kw)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none",
                ecolor=(color if color is not None else None),
                elinewidth=1.0, capsize=2.0, alpha=0.25, zorder=0)
    return True


# ---------------------------------------------------------------------- recovery rank vs. width

MAX_K = 1000            # projection curves are read up to k = 1000
LEGEND_W = 1.2          # width (in) of the legend column beside the axes
ERRORBAR_ALPHA, ERRORBAR_CAPSIZE = 0.25, 2.0


def parse_int_tokens(values: list[str]) -> list[int]:
    """Integers from space- and/or comma-separated tokens."""
    out: list[int] = []
    for raw in values:
        for tok in str(raw).split(","):
            tok = tok.strip()
            if tok:
                try:
                    out.append(int(tok))
                except ValueError as exc:
                    raise SystemExit(f"Invalid integer token in list: {tok}") from exc
    return out


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-dir", type=Path, required=True,
                        help="Sweep root containing <cfg>/runs/<run>/summary.json and experiment_outputs.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe-source", type=str, default="train", choices=["train", "test"],
                        help="Probe set the behavioral Hessian was computed on.")
    parser.add_argument("--pvec-mode", type=str, default="absolute", choices=["displacement", "absolute"],
                        help="Reference point: absolute is theta_ref = 0, displacement is theta_ref = theta_0.")
    parser.add_argument("--split", type=str, default=None, choices=["test", "train"],
                        help="Split the loss M is evaluated on. Default: same as --probe-source.")
    parser.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.075, 0.1],
                        help="Absolute tolerances tau.")
    parser.add_argument("--qs", type=float, nargs="+", default=[0.8, 0.85, 0.9],
                        help="Relative recovery targets q.")
    parser.add_argument("--exclude-widths", type=str, nargs="*", default=[],
                        help="Student widths d to exclude (space- or comma-separated).")
    _trd.add_relative_ref_arg(parser)


def load_threshold_payloads(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Records of the selected variant, one payload per threshold, and the split.

    Each payload holds the per-seed ranks at every width (``by_d``) and their
    mean and std over seeds (``d_vals``, ``y_vals``, ``y_err``).
    """
    split = args.split or args.probe_source
    taus = sorted(set(args.taus))
    qs = sorted(set(args.qs))
    records = _trd.gather_records(
        root_dir=args.root_dir,
        test_thresholds=taus,
        relative_q=qs,
        max_k=MAX_K,
        exclude_widths=set(parse_int_tokens(args.exclude_widths)),
        relative_ref=args.relative_ref,
    )
    if not records:
        raise SystemExit(f"No valid runs found under {args.root_dir}")
    records_variant = filter_variant(records, args.probe_source, args.pvec_mode)
    if not records_variant:
        raise SystemExit("No records found for variant "
                         f"probe_source={args.probe_source}, projection_vector_mode={args.pvec_mode}")

    # Seeds per width: the denominator of "seeds reaching the tolerance".
    n_total: dict[int, int] = defaultdict(int)
    for r in records_variant:
        d = maybe_int(r.get("d"))
        if d is not None:
            n_total[int(d)] += 1

    payloads: list[dict[str, Any]] = []
    for kind, value in [("tau", t) for t in taus] + [("q", q) for q in qs]:
        tok = sanitize_float_token(value)
        rank_key = f"k_{split}_tau_{tok}" if kind == "tau" else f"r_{split}_q_{tok}"
        by_d: dict[int, list[float]] = defaultdict(list)
        for r in records_variant:
            d, v = maybe_int(r.get("d")), maybe_float(r.get(rank_key))
            if d is not None and v is not None and np.isfinite(v):
                by_d[int(d)].append(float(v))
        if not by_d:
            continue
        d_vals = sorted(by_d)
        arrs = [np.asarray(by_d[d], dtype=float) for d in d_vals]
        payloads.append({
            "kind": kind,
            "value": float(value),
            "rank_key": rank_key,
            "by_d": by_d,
            "d_vals": d_vals,
            "y_vals": np.asarray([float(np.mean(a)) for a in arrs], dtype=float),
            "y_err": np.asarray([float(np.std(a, ddof=1)) if a.size > 1 else 0.0 for a in arrs], dtype=float),
            "n_per_d": np.asarray([a.size for a in arrs], dtype=float),
            "n_total_per_d": np.asarray([n_total[d] for d in d_vals], dtype=float),
        })
    if not payloads:
        raise SystemExit("No rank values available for the selected thresholds.")
    return records_variant, payloads, split


def pooled_points(payloads: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """(d, mean rank) pairs of all thresholds, pooled for the shared regression."""
    x_all: list[float] = []
    y_all: list[float] = []
    for p in payloads:
        xv = np.asarray(p["d_vals"], dtype=float)
        yv = np.asarray(p["y_vals"], dtype=float)
        m = np.isfinite(xv) & np.isfinite(yv)
        x_all.extend(xv[m].tolist())
        y_all.extend(yv[m].tolist())
    if not x_all:
        raise SystemExit("No finite points for the shared regression.")
    return np.asarray(x_all, dtype=float), np.asarray(y_all, dtype=float)


def piecewise_line(fit: dict[str, Any], d_min: float, d_max: float, d_break: float):
    """The two fitted segments on a fine grid: x, y and the left/right masks."""
    x0 = np.linspace(d_min, d_max, 500)
    yhat0 = np.full_like(x0, np.nan, dtype=float)
    left0 = np.isfinite(x0) & (x0 <= d_break)
    right0 = np.isfinite(x0) & (x0 >= d_break)
    if np.isfinite(fit["m_left"]) and np.isfinite(fit["b_left"]):
        yhat0[left0] = fit["m_left"] * x0[left0] + fit["b_left"]
    if np.isfinite(fit["m_right"]) and np.isfinite(fit["b_right"]):
        yhat0[right0] = fit["m_right"] * x0[right0] + fit["b_right"]
    return x0, yhat0, left0, right0


def threshold_colors(payloads: list[dict[str, Any]]) -> tuple[list[float], list[float], dict, dict]:
    """tau and q values with their colours, from one shared tab10 stream."""
    tau_vals = sorted({float(p["value"]) for p in payloads if p["kind"] == "tau"})
    q_vals = sorted({float(p["value"]) for p in payloads if p["kind"] == "q"})
    cmap = plt.cm.get_cmap("tab10")
    tau_colors = {v: cmap(j % 10) for j, v in enumerate(tau_vals)}
    q_colors = {v: cmap((len(tau_vals) + j) % 10) for j, v in enumerate(q_vals)}
    return tau_vals, q_vals, tau_colors, q_colors


def draw_rank_curves(ax: plt.Axes, payloads: list[dict[str, Any]], tau_colors: dict, q_colors: dict,
                     hollow_partial: bool = False) -> None:
    """Mean rank per width for every threshold, with faint std error bars.

    hollow_partial: hollow marker where only some of the seeds reach the
    tolerance (the mean is then over those seeds), filled where all do. On for
    --relative-ref theta_ref; off draws every point filled.
    """
    for p in payloads:
        val = float(p["value"])
        if p["kind"] == "tau":
            color, ls, mk = tau_colors[val], "-", "o"
        else:
            color, ls, mk = q_colors[val], "--", "s"
        x = np.asarray(p["d_vals"], dtype=float)
        y = np.asarray(p["y_vals"], dtype=float)
        err = np.asarray(p["y_err"], dtype=float)
        obs = np.isfinite(x) & np.isfinite(y)
        if not np.any(obs):
            continue
        if hollow_partial:
            full = (np.asarray(p["n_per_d"], dtype=float) >= np.asarray(p["n_total_per_d"], dtype=float))[obs]
            ax.plot(x[obs], y[obs], linewidth=1.2, alpha=1.0, color=color, linestyle=ls, zorder=1)
            for msk, face in ((full, color), (~full, "white")):
                if np.any(msk):
                    ax.plot(x[obs][msk], y[obs][msk], linestyle="none", marker=mk, markersize=4.0,
                            color=color, markerfacecolor=face, markeredgewidth=1.0, zorder=2)
        else:
            ax.plot(x[obs], y[obs], linewidth=1.2, alpha=1.0, color=color, linestyle=ls,
                    marker=mk, markersize=4.0, zorder=1)
        e = np.where(np.isfinite(err[obs]), np.maximum(0.0, err[obs]), 0.0)
        if np.any(e > 0):
            ax.errorbar(x[obs], y[obs], yerr=np.vstack([e, e]), fmt="none", ecolor=color,
                        elinewidth=1.0, capsize=ERRORBAR_CAPSIZE, alpha=ERRORBAR_ALPHA, zorder=0)


def threshold_legend_entries(tau_vals, q_vals, tau_colors, q_colors) -> list[tuple[Any, str]]:
    """Legend block: 'Absolute' header with the taus, 'Relative' header with the qs."""
    empty = Line2D([], [], linestyle="none")
    return (
        [(empty, r"$\mathbf{Absolute}$")]
        + [(Line2D([0], [0], color=tau_colors[t], linewidth=2.0, linestyle="-", marker="o", markersize=5.0),
            rf"$\tau={t:g}$") for t in tau_vals]
        + [(empty, r"$\mathbf{Relative}$")]
        + [(Line2D([0], [0], color=q_colors[q], linewidth=2.0, linestyle="--", marker="s", markersize=5.0),
            rf"$q={q:g}$") for q in q_vals]
    )


def side_legend(ax: plt.Axes, entries: list[tuple[Any, str]]) -> None:
    """Single-column legend in the column to the right of the axes."""
    ax.legend([h for h, _ in entries], [l for _, l in entries], loc="upper left",
              bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.0, labelspacing=0.15,
              columnspacing=0.9, borderpad=0.2)


def anchored_ticks(ax: plt.Axes, anchor: float) -> list[int]:
    """Evenly spaced integer x-ticks (step 4, or 2 on short ranges) through anchor."""
    x_min, x_max = ax.get_xlim()
    step = 4 if (x_max - x_min) > 12 else 2
    start = float(anchor)
    while start - step >= x_min:
        start -= step
    ticks: list[int] = []
    t = start
    while t <= x_max:
        if t >= x_min:
            ticks.append(int(round(t)))
        t += step
    if ticks:
        ax.set_xticks(ticks)
    return ticks


def threshold_tag(payloads: list[dict[str, Any]]) -> str:
    return "_".join(f"{p['kind']}{sanitize_float_token(float(p['value']))}" for p in payloads)


def finalize_paths(paths: list[str], args: argparse.Namespace) -> list[str]:
    """Append the variant (probe set, reference point) and seed aggregation to the file names."""
    return add_suffix_to_plot_paths(paths, f"_probesrc_{args.probe_source}_mode_{args.pvec_mode}_seed_all")
