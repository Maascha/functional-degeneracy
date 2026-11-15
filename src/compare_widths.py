"""MNIST figures across widths: projection curves, training curves and spectra.

Loads experiment_outputs.pt of the width-scaled MNIST runs (h1 = h2 = w) and
draws, for all widths in one panel each:

  * test / train accuracy of the projected model theta_0 + Pi_k (theta* - theta_0)
    over the number k of leading behavioral-Hessian eigendirections, with a
    zoomed inset on the tail and the final accuracy of each model as a reference
    line;
  * training cross-entropy of the projected model over k;
  * test accuracy and test loss over training;
  * the behavioral eigenvalues, their cumulative mass, and the spectral rank
    s_q = min{k : sum_{j<=k} lambda_j >= q sum_j lambda_j} over parameter count
    all within the computed top-M spectrum;
  * the behavioral recovery rank over width (mnist_rank_vs_width), from the
    test-accuracy curve A(theta_k):
        absolute  r_tau   = min{ k : A(theta_k) >= tau }
        relative  r_hat_q = min{ k : (A(theta_k) - A(theta_0)) / (A(theta*) - A(theta_0)) >= q }
    on the running maximum of the curve; NaN if not reached within the computed k.

    python src/compare_widths.py --base-dir results/mnist --probe-source train \
        --out-dir figures/mnist/curves --topm 12000 \
        --plot-spectral-appendix --plot-test-accuracy-zoom --max-eig-k 12000
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

# Panel geometry and font sizes, shared with the rank-vs-width plot below and
# matching ts_plot_utils: drawn at 3.4in and placed at about 1.7in on the page.
FIGSIZE = (3.4, 2.9)
# TrueType fonts in the PDF instead of Type 3.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
LABEL_FS = 14
TICK_FS = 12
LEGEND_FS = 12
# One dash pattern per width for the final-accuracy reference lines, since
# several final accuracies are too close to be told apart by colour alone.
_FINAL_REF_DASHES = [(1, 1), (4, 1.5), (1, 2.5), (4, 1.5, 1, 1.5), (6, 2)]
# Fraction of the x range shown in the zoomed inset of the accuracy panels.
ZOOM_TAIL_FRACTION = 0.2
# Cumulative eigenvalue-mass thresholds q for the spectral rank s_q.
SPECTRAL_Q = [0.90, 0.95, 0.99]


def _save_figure_both(fig: plt.Figure, out_base: Path) -> None:
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


def _draw_final_refs(ax: plt.Axes, refs: list[tuple[float, Any]], inset: bool = False) -> None:
    """One dashed horizontal reference line per width; refs are (y, colour)."""
    for i, (y, color) in enumerate(refs):
        ax.axhline(y, color=color, linewidth=1.4 if inset else 1.2,
                   dashes=_FINAL_REF_DASHES[i % len(_FINAL_REF_DASHES)], alpha=0.9)


def _clear_inset_ticks(axins: plt.Axes) -> None:
    axins.set_xticks([])
    axins.set_yticks([])
    axins.minorticks_off()


# ------------------------------------------------------------------ loading

def load_experiment(exp_dir: Path) -> dict[str, Any]:
    outputs_path = exp_dir / "experiment_outputs.pt"
    if not outputs_path.exists():
        raise FileNotFoundError(f"No experiment_outputs.pt in {exp_dir}")
    return torch.load(outputs_path, map_location="cpu", weights_only=False)


def get_width_label(width_str: str) -> str:
    """'32_32_bs_64' -> '32,32'."""
    return re.sub(r"_bs_\d+", "", width_str).replace("_", ",")


def _width_sort_key(width_label: str) -> tuple[int, int]:
    h1, h2 = width_label.split(",")
    return (int(h1), int(h2))


def _sorted_items(experiments: dict[str, dict[str, Any]]):
    return sorted(experiments.items(), key=lambda kv: _width_sort_key(kv[0]))


def _make_color_map(experiments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One tab10 colour per width, in width order."""
    labels = sorted(experiments.keys(), key=_width_sort_key)
    cmap = plt.cm.get_cmap("tab10")
    return {label: cmap(i % cmap.N) for i, label in enumerate(labels)}


def _compute_param_count(outputs: dict[str, Any]) -> int:
    cfg = outputs["model_config"]
    dims = [int(cfg["input_dim"])] + [int(h) for h in cfg["hidden_dims"]] + [int(cfg["output_dim"])]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


# ------------------------------------------------------------------- legend

def _width_legend(
    ax: plt.Axes,
    add_final_ref: bool = False,
    final_ref_label: str = "Final accuracy",
    loc: str = "lower right",
    final_ref_dashes: tuple | None = None,
) -> None:
    """Two-column legend of the widths in numeric order ('32,32' shown as '32'),
    optionally with a proxy entry for the final-value reference lines."""
    handles, labels = ax.get_legend_handles_labels()

    def _compact(label: str) -> str:
        parts = [s.strip() for s in label.split(",")]
        return parts[0] if len(parts) == 2 and parts[0] == parts[1] else label

    ordered = sorted(zip(handles, labels), key=lambda hl: _width_sort_key(hl[1]))
    ordered_handles = [h for h, _ in ordered]
    ordered_labels = [_compact(l) for _, l in ordered]
    if add_final_ref:
        if final_ref_dashes is None:
            ordered_handles.append(Line2D([0], [0], color="0.5", linewidth=1.8, linestyle=":"))
        else:   # drawn like the reference lines themselves (_draw_final_refs)
            ordered_handles.append(Line2D([0], [0], color="0.5", linewidth=1.2, alpha=0.9,
                                          dashes=final_ref_dashes))
        ordered_labels.append(final_ref_label)
    ax.legend(
        ordered_handles,
        ordered_labels,
        fontsize=LEGEND_FS,
        title="Widths",
        title_fontsize=LEGEND_FS,
        loc=loc,
        ncol=2,
        frameon=True,
        framealpha=0.9,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.05,
        handleheight=0.65,
        columnspacing=0.7,
        borderpad=0.2,
        borderaxespad=0.15,
    )


# ---------------------------------------------------------- training curves

def _plot_training_metric_on_ax(ax, exps, color_map, curve_key: str) -> None:
    for width_label, exp in _sorted_items(exps):
        curve = exp["outputs"][curve_key]
        if isinstance(curve, torch.Tensor):
            curve = curve.cpu().numpy()
        ax.plot(np.arange(len(curve)), curve, label=width_label, linewidth=2,
                linestyle="-", color=color_map[width_label])


def plot_training_curves(experiments, out_dir: Path, probe_tag: str, plot_accuracy_zoom: bool) -> None:
    """Test loss over training and test accuracy over training with
    a zoomed inset on the last ZOOM_TAIL_FRACTION of steps."""
    color_map = _make_color_map(experiments)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_training_metric_on_ax(ax, experiments, color_map, "test_loss_curve")
    ax.set_xlabel("Training step", fontsize=LABEL_FS)
    ax.set_ylabel("Cross-entropy loss", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    _width_legend(ax, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure_both(fig, out_dir / f"training_test_loss_probe_{probe_tag}_same")

    if not plot_accuracy_zoom:
        return
    curve_key = "test_acc_curve"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_training_metric_on_ax(ax, experiments, color_map, curve_key)
    refs = [(float(exp["outputs"]["test_acc_star"]), color_map[width_label])
            for width_label, exp in _sorted_items(experiments)]
    _draw_final_refs(ax, refs)

    curves = [exp["outputs"][curve_key].detach().cpu().numpy().astype(float)
              for exp in experiments.values()]
    max_len = max(len(v) for v in curves)
    tail_start = max(0, int(math.floor((1.0 - ZOOM_TAIL_FRACTION) * max_len)))
    tail_values = [v[tail_start:] for v in curves if len(v) > tail_start]
    y_tail = np.concatenate(tail_values + [np.array([y for y, _ in refs])])
    y_min = max(0.0, float(np.min(y_tail)) - 0.015)
    y_max = min(1.0, float(np.max(y_tail)) + 0.015)
    if y_max - y_min < 0.02:
        center = 0.5 * (y_min + y_max)
        y_min = max(0.0, center - 0.01)
        y_max = min(1.0, center + 0.01)

    ax.set_xlabel("Training step", fontsize=LABEL_FS)
    ax.set_ylabel("Test accuracy", fontsize=LABEL_FS)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    _width_legend(ax)
    ax.grid(True, alpha=0.3)

    axins = zoomed_inset_axes(ax, zoom=4, loc="center", bbox_to_anchor=(0.12, 0.1, 1, 1),
                              bbox_transform=ax.transAxes)
    _plot_training_metric_on_ax(axins, experiments, color_map, curve_key)
    _draw_final_refs(axins, refs, inset=True)
    axins.set_xlim(float(tail_start), float(max_len - 1))
    axins.set_ylim(y_min, y_max)
    _clear_inset_ticks(axins)
    axins.grid(True, alpha=0.2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.tight_layout()
    _save_figure_both(fig, out_dir / f"training_test_accuracy_probe_{probe_tag}_same_zoom_tail")


# ------------------------------------------------- projection curves over k

def _metric_reference_value(outputs: dict[str, Any], metric_key: str) -> float:
    """Reference line for a projection curve: the metric of the trained model
    theta* on the data the curve is evaluated on (full training set for the
    cross-entropy and train accuracy, full test set for test accuracy)."""
    if "train_accuracy" in metric_key:
        return float(outputs["train_acc_star"])
    if "test_accuracy" in metric_key:
        return float(outputs["test_acc_star"])
    if metric_key == "task_ce_by_k_behavioral":
        return float(outputs["train_ce_star"])
    raise KeyError(f"no reference value for {metric_key}")


def _plot_metric_on_ax(ax, experiments, color_map, metric_key: str, is_inset: bool = False) -> None:
    """Curves over k = 1..K, plus the per-width reference lines."""
    refs: list[tuple[float, Any]] = []
    for width_label, exp in _sorted_items(experiments):
        outputs = exp["outputs"]
        values = outputs[metric_key].cpu().numpy()
        k_values = np.arange(1, len(values) + 1, dtype=float)
        color = color_map[width_label]
        ax.plot(k_values, values, label=width_label, linewidth=2, linestyle="-", color=color)
        refs.append((_metric_reference_value(outputs, metric_key), color))
    _draw_final_refs(ax, refs, inset=is_inset)


def _plot_accuracy_zoom(exps, color_map, metric_key: str, y_label: str, out_base: Path) -> None:
    """Accuracy over k with a zoomed inset over the last ZOOM_TAIL_FRACTION of k."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_metric_on_ax(ax, exps, color_map, metric_key)

    curves = [exp["outputs"][metric_key].detach().cpu().numpy().astype(float)
              for exp in exps.values()]
    max_len = max(len(v) for v in curves)
    tail_start = max(1, int(math.floor((1.0 - ZOOM_TAIL_FRACTION) * max_len)))
    tail_values = [v[tail_start - 1:] for v in curves if len(v) >= tail_start]
    # Keep the final-accuracy references inside the zoom, not on its edge.
    refs_y = [_metric_reference_value(e["outputs"], metric_key) for e in exps.values()]
    y_tail = np.concatenate([np.concatenate(tail_values), refs_y])
    y_min = max(0.0, float(np.min(y_tail)) - 0.015)
    y_max = min(1.0, float(np.max(y_tail)) + 0.006)
    if y_max - y_min < 0.02:
        center = 0.5 * (y_min + y_max)
        y_min = max(0.0, center - 0.01)
        y_max = min(1.0, center + 0.01)

    ax.set_xlabel(r"Leading eigendirections $k$", fontsize=LABEL_FS)
    ax.set_ylabel(y_label, fontsize=LABEL_FS)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    _width_legend(ax)
    ax.grid(True, alpha=0.3)

    axins = zoomed_inset_axes(ax, zoom=4, loc="center", bbox_to_anchor=(0.12, 0.1, 1, 1),
                              bbox_transform=ax.transAxes)
    _plot_metric_on_ax(axins, exps, color_map, metric_key, is_inset=True)
    axins.set_xlim(float(tail_start), float(max_len))
    axins.set_ylim(y_min, y_max)
    _clear_inset_ticks(axins)
    axins.grid(True, alpha=0.2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.tight_layout()
    _save_figure_both(fig, out_base)


def plot_metrics_over_k(experiments, out_dir: Path, probe_tag: str, plot_accuracy_zoom: bool) -> None:
    """Projection curves with theta_ref = theta_0: training cross-entropy
    and, with zoomed insets, test and train accuracy."""
    color_map = _make_color_map(experiments)
    tag = f"displacement_probe_{probe_tag}_same"

    metric_key = "task_ce_by_k_behavioral"
    fig, ax = plt.subplots(figsize=FIGSIZE)
    _plot_metric_on_ax(ax, experiments, color_map, metric_key)
    ax.set_xlabel(r"Leading eigendirections $k$", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    # The loss falls with k, so the upper right is empty.
    _width_legend(ax, add_final_ref=True, final_ref_label="Final loss", loc="upper right",
                  final_ref_dashes=_FINAL_REF_DASHES[1])
    ax.set_ylabel("Cross-entropy loss", fontsize=LABEL_FS)
    # Headroom for the legend above the tails.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + 1.3 * (hi - lo))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure_both(fig, out_dir / f"task_ce_comparison_{tag}")

    if plot_accuracy_zoom:
        for metric, y_label in (("test_accuracy", "Test accuracy"), ("train_accuracy", "Train accuracy")):
            _plot_accuracy_zoom(experiments, color_map, f"{metric}_by_k_behavioral", y_label,
                                out_dir / f"{metric}_comparison_{tag}_zoom_tail")


# ------------------------------------------------------------------ spectra

def _positive_eigenvalues(outputs: dict[str, Any]) -> np.ndarray:
    """Finite positive behavioral eigenvalues, in descending order."""
    evals = np.asarray(outputs["top_eigenvalues_behavioral"].detach().cpu().numpy(), dtype=float).reshape(-1)
    evals = evals[np.isfinite(evals)]
    evals = evals[evals > 0]
    return np.sort(evals)[::-1]


def _spectral_q_label(q: float) -> str:
    q_pct = q * 100.0
    if np.isclose(q_pct, round(q_pct)):
        return str(int(round(q_pct)))
    return str(q_pct).replace(".", "_")


def plot_eigenvalues_over_k(experiments, out_dir: Path, probe_tag: str, max_eig_k: int) -> None:
    """Eigenvalues on a log scale and their cumulative mass, with the
    thresholds SPECTRAL_Q as dotted lines."""
    color_map = _make_color_map(experiments)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for width_label, exp in _sorted_items(experiments):
        evals = _positive_eigenvalues(exp["outputs"])[:max_eig_k]
        ax.semilogy(np.arange(1, len(evals) + 1), evals, label=width_label, linewidth=2,
                    linestyle="-", color=color_map[width_label])
    ax.set_xlabel("Behavioral Hessian\n" r"eigenvalue index $k$", fontsize=LABEL_FS)
    ax.set_ylabel("Eigenvalues", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    _width_legend(ax, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save_figure_both(fig, out_dir / f"eigenvalues_over_k_log_probe_{probe_tag}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for width_label, exp in _sorted_items(experiments):
        evals = _positive_eigenvalues(exp["outputs"])[:max_eig_k]
        cum_mass = np.cumsum(evals) / float(np.sum(evals))
        ax.plot(np.arange(1, len(evals) + 1), cum_mass, label=width_label, linewidth=2,
                linestyle="-", color=color_map[width_label])
    for q in SPECTRAL_Q:
        ax.axhline(q, color="0.6", linewidth=1.2, linestyle=":")
    ax.set_xlabel(r"Top eigenvalues $k$", fontsize=LABEL_FS)
    ax.set_ylabel("Fraction of\neigenvalue sum", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    _width_legend(ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure_both(fig, out_dir / f"eigenvalues_cumulative_mass_over_k_probe_{probe_tag}")


def compute_spectral_records(experiments) -> list[dict[str, Any]]:
    """Spectral rank s_q per width, over the full computed positive spectrum."""
    records: list[dict[str, Any]] = []
    for width_label, exp in _sorted_items(experiments):
        outputs = exp["outputs"]
        evals = _positive_eigenvalues(outputs)
        cum_mass = np.cumsum(evals) / float(np.sum(evals))
        h1, h2 = _width_sort_key(width_label)
        rec: dict[str, Any] = {
            "width_label": width_label,
            "h1": h1,
            "h2": h2,
            "param_count": _compute_param_count(outputs),
            "num_positive_eigs": int(len(evals)),
        }
        for q in SPECTRAL_Q:
            idx = np.where(cum_mass >= q)[0]
            rec[f"spectral_rank_{_spectral_q_label(q)}"] = float(idx[0] + 1) if len(idx) else math.nan
        records.append(rec)
    return records


def save_spectral_rank_csv(records: list[dict[str, Any]], out_dir: Path, probe_tag: str) -> None:
    fname = out_dir / f"spectral_rank_over_width_{probe_tag}.csv"
    with fname.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {fname}")


def plot_mnist_spectral_rank_vs_params(records, out_dir: Path, probe_tag: str) -> None:
    """Spectral rank s_q over parameter count, ticks labelled 'count (width)'."""
    recs = sorted(records, key=lambda r: (r["param_count"], r["h1"], r["h2"]))
    x_param = np.array([r["param_count"] for r in recs], dtype=float)
    # Taller than FIGSIZE for the two-line ticks and x-label.
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], FIGSIZE[1] + 0.5))
    for q in SPECTRAL_Q:
        y = np.array([r[f"spectral_rank_{_spectral_q_label(q)}"] for r in recs], dtype=float)
        ax.plot(x_param, y, marker="o", linewidth=2, label=f"$q={q:.2f}$")
    ax.set_xlabel("Parameter count $p$\n" r"(width $w$)", fontsize=LABEL_FS)
    ax.set_ylabel("Leading eigenvalues", fontsize=LABEL_FS)
    ticks, labels = [], []
    for r in recs:
        p_val = float(r["param_count"])
        # Round half up (55,050 -> 55.1k).
        kp = Decimal(int(round(p_val))) / 1000
        p_txt = (f"{kp.quantize(Decimal('0.1'), ROUND_HALF_UP)}k" if p_val < 1e5
                 else f"{kp.quantize(Decimal('1'), ROUND_HALF_UP)}k")
        ticks.append(p_val)
        labels.append(f"{p_txt}\n({r['h1']})")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.tick_params(axis="x", which="major", labelsize=TICK_FS - 2)
    # Headroom so the upper-left legend sits above the q = 0.99 line.
    ax.set_ylim(0, 1.6 * ax.get_ylim()[1])
    ax.legend(loc="upper left", ncol=1, fontsize=LEGEND_FS, frameon=True, framealpha=0.9,
              handlelength=1.4, handletextpad=0.4, labelspacing=0.15, borderpad=0.3,
              borderaxespad=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure_both(fig, out_dir / f"appendix_mnist_spectral_rank_vs_params_{probe_tag}")

# ------------------------------------------------------------ recovery rank

RANK_TAUS = [0.85, 0.90, 0.92]
RANK_QS = [0.90, 0.95]


def _accuracy_endpoints(outputs: dict[str, Any], data_root: Path) -> tuple[float, float]:
    """(A(theta*), A(theta_0)) on the test set.

    Taken from the run when it stored both; otherwise both are recomputed on the
    CPU from the MNIST test set with the run's input normalisation (A(theta*) is
    checked against the stored value).
    """
    if outputs.get("test_acc_theta0") is not None:
        return float(outputs["test_acc_star"]), float(outputs["test_acc_theta0"])
    from torchvision import datasets
    from run_mnist_mlp_lazy import SmallMLP
    te = datasets.MNIST(root=str(data_root), train=False, download=False)
    x = (te.data.float().view(len(te), -1) / 255.0 - outputs["normalization"]["input_mean"]) \
        / outputs["normalization"]["input_std"]
    y = te.targets.long()

    def acc(theta: torch.Tensor) -> float:
        model = SmallMLP(hidden_dims=tuple(int(h) for h in outputs["model_config"]["hidden_dims"]))
        torch.nn.utils.vector_to_parameters(theta.float(), model.parameters())
        with torch.no_grad():
            return float((model(x).argmax(1) == y).float().mean())

    a_star, a_0 = acc(outputs["theta_star"]), acc(outputs["theta_0"])
    if abs(a_star - float(outputs["test_acc_star"])) > 1e-6:
        raise SystemExit(f"recomputed A(theta*)={a_star:.4f} != stored {float(outputs['test_acc_star']):.4f}; "
                         "the test set does not match the run")
    return a_star, a_0


def _first_k(curve: np.ndarray, thr: float) -> float:
    idx = np.where(np.maximum.accumulate(curve) >= thr)[0]
    return float(idx[0] + 1) if len(idx) else math.nan   # curves start at k = 1


def plot_rank_vs_width(experiments, out_dir: Path, data_root: Path) -> None:
    """Absolute and relative recovery rank over width (CSV and figure)."""
    rows = []
    for width_label, exp in _sorted_items(experiments):
        o = exp["outputs"]
        a_star, a_ref = _accuracy_endpoints(o, data_root)
        curve = o["test_accuracy_by_k_behavioral"].numpy().astype(float)
        r = {"width": _width_sort_key(width_label)[0], "param_count": o["theta_star"].numel(),
             "A_star": a_star, "A_ref": a_ref, "K": len(curve), "A_K": float(curve[-1])}
        for t in RANK_TAUS:
            r[f"r_tau_{t:.2f}"] = _first_k(curve, t)
        for q in RANK_QS:
            thr = q * a_star + (1 - q) * a_ref
            r[f"thr_q_{q:.2f}"] = thr
            r[f"r_q_{q:.2f}"] = _first_k(curve, thr)
        rows.append(r)
        print(f"w={r['width']:3d} A*={a_star:.4f} A(theta_0)={a_ref:.4f}  "
              + "  ".join(f"q={q:.2f}: {r[f'r_q_{q:.2f}']:.0f}" for q in RANK_QS))

    with (out_dir / "mnist_rank_vs_width.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # Legend in a strip beside the axes, so the axes keep the width of the other
    # panels. x positions are the parameter counts.
    x = np.array([r["param_count"] for r in rows], float)
    legend_w = 1.2
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] + legend_w, FIGSIZE[1]))
    fig.subplots_adjust(right=FIGSIZE[0] / (FIGSIZE[0] + legend_w))
    tau_l, q_l = [], []
    for t in RANK_TAUS:
        (ln,) = ax.plot(x, [r[f"r_tau_{t:.2f}"] for r in rows], marker="o", lw=2, ls="-")
        tau_l.append((ln, rf"$\tau={t:.2f}$"))
    for q in RANK_QS:
        (ln,) = ax.plot(x, [r[f"r_q_{q:.2f}"] for r in rows], marker="s", lw=2, ls="--")
        q_l.append((ln, rf"$q={q:.2f}$"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['width']}" for r in rows], fontsize=TICK_FS)
    ax.set_xlabel(r"Hidden width $w$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\hat r_q$ or $r_\tau$", fontsize=LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    # One column: an "Absolute" block, then a "Relative" block, padded to equal height.
    empty = Line2D([], [], ls="none")
    n = max(len(tau_l), len(q_l)) + 1
    a_col = [(empty, r"$\mathbf{Absolute}$")] + tau_l
    r_col = [(empty, r"$\mathbf{Relative}$")] + q_l
    a_col += [(empty, "")] * (n - len(a_col))
    r_col += [(empty, "")] * (n - len(r_col))
    ent = a_col + r_col
    ax.legend([h for h, _ in ent], [l for _, l in ent], fontsize=LEGEND_FS, ncol=1,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, handlelength=1.4,
              handletextpad=0.4, labelspacing=0.15, borderaxespad=0.0)
    ax.grid(True, alpha=0.3)
    _save_figure_both(fig, out_dir / "mnist_rank_vs_width")


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Directory holding the width_*_topm_*_curv_behavioral_pmode_both_probesrc_* runs.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--topm", type=int, default=12000, help="Number of computed eigenpairs M.")
    parser.add_argument("--probe-source", type=str, choices=["test", "train"], default="train")
    parser.add_argument("--exclude-widths", type=str, nargs="+", default=[], metavar="WIDTH",
                        help="Widths to exclude, e.g. 32_32. Reflected in the file names.")
    parser.add_argument("--max-eig-k", type=int, default=12000,
                        help="Number of leading eigenvalues shown in the spectrum plots.")
    parser.add_argument("--plot-test-accuracy-zoom", action="store_true",
                        help="Also draw the accuracy panels with a zoomed inset.")
    parser.add_argument("--data-root", type=str, default="data",
                        help="MNIST location; only needed for runs that did not store A(theta_0).")
    parser.add_argument("--plot-spectral-appendix", action="store_true",
                        help="Also draw the spectral rank over parameter count.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded = {get_width_label(w.strip()) for w in args.exclude_widths}
    suffix = f"_topm_{args.topm}_curv_behavioral_pmode_both_probesrc_{args.probe_source}"
    experiments: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(base_dir.glob(f"width_*{suffix}")):
        if not run_dir.is_dir():
            continue
        width_label = get_width_label(run_dir.name[len("width_"):-len(suffix)])
        h1, h2 = _width_sort_key(width_label)
        if h1 != h2 or width_label in excluded:
            continue
        print(f"Loading {width_label} from: {run_dir.name}")
        experiments[width_label] = {"outputs": load_experiment(run_dir)}
    if not experiments:
        raise ValueError(f"No runs with h1 = h2 matching *{suffix} in {base_dir}")

    probe_tag = f"{args.probe_source}_topm_{args.topm}"
    if args.exclude_widths:
        probe_tag += "_excl_" + "_".join(args.exclude_widths)
    spectral_tag = f"{probe_tag}_spectral_same"

    plot_training_curves(experiments, out_dir, probe_tag, args.plot_test_accuracy_zoom)
    plot_eigenvalues_over_k(experiments, out_dir, spectral_tag, args.max_eig_k)
    plot_metrics_over_k(experiments, out_dir, probe_tag, args.plot_test_accuracy_zoom)
    spectral_records = compute_spectral_records(experiments)
    save_spectral_rank_csv(spectral_records, out_dir, spectral_tag)
    if args.plot_spectral_appendix:
        plot_mnist_spectral_rank_vs_params(spectral_records, out_dir, spectral_tag)
    plot_rank_vs_width(experiments, out_dir, Path(args.data_root))
    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
