"""Diagnostic and spectrum figures of the teacher-student sweep, from the saved run outputs.

Suites (``--plot-suite``):
  diagnostics  probe-set / reference-point ablation of r_tau (tau = 0.05), and
               the final train/test MSE against width;
  spectra      behavioral-Hessian eigenvalue spectra, cumulative eigenvalue
               mass, and the spectral rank against width.

Final MSE and spectral rank use the runs with probe = train, theta_ref = theta_0
(both are independent of theta_ref). Spectra are drawn for that variant, or for
all four (probe, theta_ref) variants with ``--variant-scope all``.

    python src/plot_ts_diagnostics.py --root-dir results/ts_in10_out10 \
        --plot-suite spectra diagnostics --variant-scope all --spectra-max-k 40 \
        --out-dir figures/ts/in10_out10/diagnostics
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import ts_rank_data as _trd
from ts_rank_data import maybe_float, maybe_int, sanitize_float_token, to_numpy_1d
from ts_plot_utils import (
    LEGEND_FS,
    PAPER_FIGSIZE,
    PAPER_FIGSIZE_APPENDIX,
    VARIANTS,
    _eb,
    aggregate_main_rows_by_d,
    filter_variant,
    parse_int_tokens,
    save_figure,
    teacher_d_star,
)


# ------------------------------------------------------------ probe-set / reference-point ablation

def plot_probe_pvec_ablation_seed_agg(
    records: list[dict[str, Any]],
    thresholds: list[float],
    d_star: int | None,
    out_dir: Path,
    split: str = "test",
) -> list[str]:
    """r_tau at the first threshold for all four (probe set, theta_ref) variants.

    Each variant is aggregated over seeds separately (median, interquartile
    error bars). ``split`` is the split the loss M is evaluated on.
    """
    if not thresholds:
        return []
    tau = thresholds[0]
    key = f"k_{split}_tau_{sanitize_float_token(tau)}"

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    cmap = plt.cm.get_cmap("tab10")
    plotted = False

    for i, (probe_source, pvec_mode) in enumerate(VARIANTS):
        subset = filter_variant(records, probe_source=probe_source, pvec_mode=pvec_mode)
        rows = aggregate_main_rows_by_d(subset, scalar_keys=[key])
        if not rows:
            continue
        xs = [int(r["d"]) for r in rows]
        ref_label = (
            r"$\theta_{\mathrm{ref}}=\theta_0$"
            if pvec_mode == "displacement"
            else r"$\theta_{\mathrm{ref}}=0$"
        )
        if _eb(ax, xs, rows, key, f"{probe_source}, {ref_label}", color=cmap(i % 10)):
            plotted = True

    if not plotted:
        plt.close(fig)
        print(f"  [ablation] nothing plotted: no finite '{key}' in the records.")
        return []

    if d_star is not None:
        ax.axvline(d_star, linestyle="--", linewidth=1.2, color="black")

    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel(rf"$r_{{\tau}},\ \tau={tau:g}$")
    ax.grid(alpha=0.25)
    # Legend beside the axes; the canvas is widened by its column so the axes
    # keep their full size.
    fig.set_size_inches(PAPER_FIGSIZE[0] + 1.6, PAPER_FIGSIZE[1])
    fig.subplots_adjust(right=PAPER_FIGSIZE[0] / (PAPER_FIGSIZE[0] + 1.6))
    ax.legend(fontsize=LEGEND_FS - 1, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0,
              handlelength=1.4, handletextpad=0.4, labelspacing=0.2)
    stem = "appendix_ts_probe_pvec_ablation_seed_agg"
    if split != "test":
        stem = f"{stem}_{split}"
    return save_figure(fig, out_dir / stem)


# ------------------------------------------------------------ final MSE

def _median_iqr(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return math.nan, math.nan, math.nan
    arr = np.asarray(vals, dtype=float)
    med = float(np.median(arr))
    if len(arr) == 1:
        return med, 0.0, 0.0
    return med, med - float(np.percentile(arr, 25)), float(np.percentile(arr, 75)) - med


def _final_mse_by_width(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per width: median and interquartile half-ranges of the final MSE over seeds."""
    by_seed: dict[tuple[int, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        td, sd = maybe_int(r.get("teacher_hidden_dim")), maybe_int(r.get("d"))
        if td is None or sd is None:
            print(f"[warn] missing teacher/student width for run {r.get('run_dir')}; skipped")
            continue
        by_seed[(td, sd, maybe_int(r.get("seed")))].append(r)

    by_width: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: {"train": [], "test": []})
    for (td, sd, _seed), grp in sorted(by_seed.items(), key=lambda x: (x[0][0], x[0][1], -1 if x[0][2] is None else x[0][2])):
        for split in ("train", "test"):
            vals = [maybe_float(x.get(f"{split}_mse_star")) for x in grp]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                by_width[(td, sd)][split].append(float(np.median(np.asarray(vals, dtype=float))))

    rows: list[dict[str, Any]] = []
    for (_td, sd), vals in sorted(by_width.items(), key=lambda x: (x[0][1], x[0][0])):
        row: dict[str, Any] = {"student_hidden_dim": sd}
        for split in ("train", "test"):
            med, low, high = _median_iqr(vals[split])
            row[f"{split}_mse_star_median"] = med
            row[f"{split}_mse_star_iqr_low"] = low
            row[f"{split}_mse_star_iqr_high"] = high
        rows.append(row)
    return rows


def plot_final_mse_vs_d_seed_agg(
    records: list[dict[str, Any]],
    d_star: int | None,
    out_dir: Path,
) -> list[str]:
    """Final train/test MSE, median over seeds with interquartile error bars."""
    agg_rows = _final_mse_by_width(records)
    if not agg_rows:
        return []

    xs = [int(r["student_hidden_dim"]) for r in agg_rows]
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    cmap = plt.cm.get_cmap("tab10")
    for i, split in enumerate(("train", "test")):
        _eb(ax, xs, agg_rows, f"{split}_mse_star_median", f"{split} MSE", color=cmap(i),
            yerr_low_key=f"{split}_mse_star_iqr_low", yerr_high_key=f"{split}_mse_star_iqr_high")

    if d_star is not None:
        ax.axvline(d_star, linestyle="--", linewidth=1.2, color="black")

    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel("Final MSE")
    ax.set_yscale("log")
    # Integer ticks stepping outward from the teacher width.
    lo, hi = ax.get_xlim()
    step = 4 if (hi - lo) > 12 else 2
    anchor = float(d_star) if d_star is not None else float(int(lo) + 1)
    while anchor - step >= lo:
        anchor -= step
    ticks, t = [], anchor
    while t <= hi:
        if t >= lo:
            ticks.append(int(round(t)))
        t += step
    if ticks:
        ax.set_xticks(ticks)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=LEGEND_FS, framealpha=0.9, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.15, borderpad=0.15)
    return save_figure(fig, out_dir / "paper_ts_final_mse_vs_d_seed_agg")


# ------------------------------------------------------------ spectra

def plot_behavioral_spectral_rank_vs_d(
    records_main: list[dict[str, Any]],
    d_star: int | None,
    out_dir: Path,
) -> list[str]:
    """Number of eigenvalues needed to reach a fraction q of the spectral mass, median over seeds."""
    keys = ["spectral_rank_90", "spectral_rank_95", "spectral_rank_99"]
    rows = aggregate_main_rows_by_d(records_main, scalar_keys=keys)
    if not rows:
        return []

    xs = [int(r["d"]) for r in rows]
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    for q, key in [(0.90, "spectral_rank_90"), (0.95, "spectral_rank_95"), (0.99, "spectral_rank_99")]:
        ys = [maybe_float(r.get(key)) for r in rows]
        ys = [np.nan if y is None else y for y in ys]
        if np.any(np.isfinite(np.asarray(ys, dtype=float))):
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=f"q={q:.2f}")

    if d_star is not None:
        ax.axvline(d_star, linestyle="--", zorder=0, linewidth=1.2, color="black")

    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel("Number of eigenvalues")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=LEGEND_FS, framealpha=0.9, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.15, borderpad=0.15)
    return save_figure(fig, out_dir / "paper_ts_behavioral_spectral_rank_vs_d")


def collect_eigenvalue_curves(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median eigenvalue curve over seeds per (probe_source, projection_vector_mode, d)."""
    grouped: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    for r in records:
        d = maybe_int(r.get("d"))
        if d is None:
            continue
        eigs = to_numpy_1d(r.get("_eigs_pos"))
        eigs = eigs[np.isfinite(eigs) & (eigs > 0.0)]
        if len(eigs) == 0:
            continue
        key = (str(r.get("probe_source", "unknown")), str(r.get("projection_vector_mode", "unknown")), int(d))
        grouped[key].append(np.sort(eigs)[::-1])

    aggregated: list[dict[str, Any]] = []
    for (probe_source, pvec_mode, d) in sorted(grouped):
        curves = grouped[(probe_source, pvec_mode, d)]
        mat = np.full((len(curves), max(len(c) for c in curves)), np.nan, dtype=float)
        for i, c in enumerate(curves):
            mat[i, : len(c)] = c
        med = np.nanmedian(mat, axis=0)
        med = med[np.isfinite(med) & (med > 0.0)]
        if len(med):
            aggregated.append({"d": d, "probe_source": probe_source,
                               "projection_vector_mode": pvec_mode, "eigs": med})
    return aggregated


def _width_colormap(rows: list[dict[str, Any]]) -> tuple[Any, mcolors.Normalize]:
    d_vals = np.asarray([int(r["d"]) for r in rows], dtype=float)
    d_min, d_max = float(np.min(d_vals)), float(np.max(d_vals))
    if d_max <= d_min:
        d_max = d_min + 1.0
    return plt.cm.get_cmap("viridis"), mcolors.Normalize(vmin=d_min, vmax=d_max)


def _add_width_colorbar(fig: plt.Figure, ax: plt.Axes, cmap: Any, norm: mcolors.Normalize) -> None:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"Hidden width $d$")


def plot_behavioral_eigenvalue_spectra(
    aggregated_curves: list[dict[str, Any]],
    out_base: Path,
    d_star: int | None,
    max_k: int,
) -> list[str]:
    """Leading max_k eigenvalues, one curve per width."""
    rows = sorted(aggregated_curves, key=lambda x: int(x["d"]))
    if not rows:
        return []

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    cmap, norm = _width_colormap(rows)
    for r in rows:
        y = np.asarray(r["eigs"], dtype=float)[: int(max_k)]
        ax.plot(np.arange(1, len(y) + 1, dtype=float), y, linewidth=1.8, color=cmap(norm(float(r["d"]))))

    if d_star is not None and d_star > 0:
        ax.axvline(float(d_star), linestyle="--", linewidth=1.2, color="black")

    ax.set_xlabel("Behavioral-Hessian eigenvalue index k")
    ax.set_ylabel("Eigenvalue")
    ax.set_yscale("log")
    ax.set_xlim(1, max_k)
    ax.grid(alpha=0.25)
    _add_width_colorbar(fig, ax, cmap, norm)
    return save_figure(fig, out_base)


def plot_cumulative_eigenvalue_mass(
    aggregated_curves: list[dict[str, Any]],
    out_base: Path,
) -> list[str]:
    """Share of the total eigenvalue mass in the top k eigenvalues, one curve per width."""
    rows = sorted(aggregated_curves, key=lambda x: int(x["d"]))
    if not rows:
        return []

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    cmap, norm = _width_colormap(rows)
    for r in rows:
        cum = np.cumsum(np.asarray(r["eigs"], dtype=float))
        y = cum / float(cum[-1])
        ax.plot(np.arange(1, len(y) + 1, dtype=float), y, linewidth=1.8, color=cmap(norm(float(r["d"]))))

    for q in [0.90, 0.95, 0.99]:
        ax.axhline(float(q), color="0.6", linewidth=1.1, linestyle=":")

    ax.set_xlabel("Top eigenvalues k")
    ax.set_ylabel("Fraction of eigenvalue sum")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    _add_width_colorbar(fig, ax, cmap, norm)
    return save_figure(fig, out_base)


# ------------------------------------------------------------ main

MAIN_VARIANT = ("train", "displacement")
ABLATION_TAU = 0.05
MAX_K = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", type=Path, required=True,
                        help="Sweep root containing <cfg>/runs/<run>/summary.json and experiment_outputs.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plot-suite", type=str, nargs="+", default=["diagnostics"],
                        choices=["spectra", "diagnostics"])
    parser.add_argument("--variant-scope", type=str, default="main", choices=["main", "all"],
                        help="Spectra for the main variant only, or for all four probe/theta_ref variants.")
    parser.add_argument("--spectra-max-k", type=int, default=20,
                        help="Number of leading eigenvalues shown in the spectrum figure.")
    parser.add_argument("--ablation-split", type=str, default="test", choices=["train", "test"],
                        help="Split the loss is evaluated on in the probe/theta_ref ablation.")
    parser.add_argument("--exclude-widths", type=str, nargs="*", default=[],
                        help="Student widths d to exclude (space- or comma-separated).")
    parser.add_argument("--for-appendix", action="store_true",
                        help="Draw at the appendix geometry (5.3 x 4.5in instead of 3.4 x 2.9).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.for_appendix:
        global PAPER_FIGSIZE
        PAPER_FIGSIZE = PAPER_FIGSIZE_APPENDIX
        print(f"  [figsize] appendix geometry {PAPER_FIGSIZE}")

    records = _trd.gather_records(
        root_dir=args.root_dir,
        test_thresholds=[ABLATION_TAU],
        relative_q=[],
        max_k=MAX_K,
        exclude_widths=set(parse_int_tokens(args.exclude_widths)),
    )
    if not records:
        raise SystemExit(f"No valid runs found under {args.root_dir}")
    d_star = teacher_d_star(records)
    records_main = filter_variant(records, *MAIN_VARIANT)
    if not records_main:
        raise SystemExit(f"No records found for variant probe_source, projection_vector_mode = {MAIN_VARIANT}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    if "diagnostics" in args.plot_suite:
        saved += plot_final_mse_vs_d_seed_agg(records=records_main, d_star=d_star, out_dir=out_dir)
        saved += plot_probe_pvec_ablation_seed_agg(records=records, thresholds=[ABLATION_TAU],
                                                   d_star=d_star, out_dir=out_dir,
                                                   split=args.ablation_split)

    if "spectra" in args.plot_suite:
        saved += plot_behavioral_spectral_rank_vs_d(records_main=records_main, d_star=d_star, out_dir=out_dir)
        eig_rows = collect_eigenvalue_curves(records)
        for probe_source, pvec_mode in (VARIANTS if args.variant_scope == "all" else [MAIN_VARIANT]):
            subset = [r for r in eig_rows
                      if r["probe_source"] == probe_source and r["projection_vector_mode"] == pvec_mode]
            if not subset:
                continue
            tag = f"{probe_source}_{pvec_mode}"
            saved += plot_behavioral_eigenvalue_spectra(
                subset, out_dir / f"appendix_ts_behavioral_eigenvalues_{tag}_max{args.spectra_max_k}",
                d_star=d_star, max_k=args.spectra_max_k)
            saved += plot_cumulative_eigenvalue_mass(
                subset, out_dir / f"appendix_ts_cumulative_eigenvalue_mass_{tag}")

    print(f"Records: {len(records)} (main variant: {len(records_main)})")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
