"""Recovery rank versus one-shot pruning on the teacher-student checkpoints.

Every number is obtained by deleting parameters from an already trained network
and evaluating it: no gradient steps, no fine-tuning. All methods are read
against the same criterion, test MSE (against the teacher) reaching an absolute
threshold tau, so their degrees of freedom are directly comparable:

  r_tau   recovery rank: the number of behavioral eigendirections needed when
          theta* is projected onto the top-k eigenvectors of H_beh (reference
          point 0). Reported for two orderings of the directions, descending
          lambda_j ("eigenvalue") and descending lambda_j * c_j^2
          ("loss-optimal"), computed from one eigendecomposition.
  structure pruning
          greedy neuron removal: repeatedly delete the hidden unit whose removal
          increases the training loss least. Counted as parameters,
          p(n) = n * (d_in + 1 + d_out) + d_out.
  magnitude pruning
          global one-shot threshold on |theta_i|.
  OBD     Optimal Brain Damage (LeCun et al., 1990): global one-shot threshold
          on the saliency 0.5 * H_ii * theta_i^2, with H the behavioral Hessian
          on the probe set.

With --refit-output lstsq the output layer is refitted in closed form (least
squares on the training set) after pruning, for structure and magnitude pruning.
--include-teacher runs the teacher through the same pipeline, once per seed, and
draws its counts as horizontal reference lines.

    python src/prune_vs_recovery_rank.py \\
        --checkpoint-root results/ts_in10_out10 --widths 3 4 ... 20 \\
        --tau-abs 0.075 --refit-output none lstsq --show-obd \\
        --piecewise-fit figures/ts/in10_out10/rank/piecewise_fit.json \\
        --out figures/ts/in10_out10/pruning/pruning

Writes <out>_<tag>.json with all curves and, per refit mode,
<out>_<tag>_<refit>_dof.pdf and a matching .csv table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# torch is imported before numpy on purpose: the import order decides which MKL
# runtime settings are loaded first, and the Lanczos eigendecomposition behind the
# recovery curves is sensitive to the resulting floating-point round-off.
import torch
import numpy as np  # noqa: E402,I001

from curvature_projection import (
    compute_behavioral_eigenspace_dual,
    recover_behavioral_param_eigenvector_lazy,
)
from run_toy_teacher_student import StudentSingleHidden, rebuild_run_data, student_for_run
from train import flatten_params, get_param_layout, set_params_from_flat

DEVICE = torch.device("cpu")
CFG_RE = re.compile(r"in(\d+)_th(\d+)_out(\d+)_sd(\d+)_(bias|nobias)_seed(\d+)$")
DIRECTION_ORDERS = ("eigenvalue", "optimal")


# --------------------------------------------------------------------- loading

def _pick_summary(cfg_dir: Path, probe_source: str, pvec: str) -> dict | None:
    """Return the summary of the run with the requested probe set and reference point."""
    for path in sorted(cfg_dir.glob("runs/*/summary.json")):
        try:
            with path.open() as f:
                summ = json.load(f)
        except Exception:
            continue
        if (summ.get("probe_source") == probe_source
                and summ.get("projection_vector_mode", "displacement") == pvec):
            return summ
    return None


def load_run(cfg_dir: Path, probe_source: str = "train", pvec: str = "absolute"):
    """Rebuild a run: same teacher, same data, same trained weights."""
    summ = _pick_summary(cfg_dir, probe_source, pvec)
    if summ is None:
        return None
    ckpt_path = cfg_dir / "model" / "checkpoint.pt"
    if not ckpt_path.exists():
        return None

    ti = summ["teacher_info"]
    seed = int(summ["seed"])
    teacher, (x_tr, _, x_te, y_te, x_pr, _), dtype = rebuild_run_data(summ, DEVICE)
    probe = {"train": x_tr, "test": x_te, "synthetic": x_pr}[summ["probe_source"]]
    model = student_for_run(summ, dtype, DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict({
        k: (v.to(dtype=dtype) if torch.is_floating_point(v) else v)
        for k, v in ckpt["model_state_dict"].items()
    })

    # Sanity check: the regenerated test set must reproduce the reported test MSE.
    recomputed = test_mse(model, x_te, y_te)
    reported = float(summ.get("test_mse_star", recomputed))
    if reported > 0 and abs(recomputed - reported) / reported > 1e-3:
        print(f"    WARNING {cfg_dir.name}: recomputed test MSE {recomputed:.4e} != "
              f"summary {reported:.4e}. Data regeneration does not match the run.")
    with torch.no_grad():
        y_tr_clean = teacher(x_tr)
    return dict(model=model, probe=probe, x_te=x_te, y_te=y_te,
                x_fit=x_tr, y_fit=y_tr_clean, teacher=teacher,
                dtype=dtype, d_in=int(ti["input_dim"]), d_out=int(ti["output_dim"]),
                hidden=int(summ["student_hidden_dim"]), seed=seed)


# ------------------------------------------------------------------- teacher

def wrap_teacher_as_student(teacher, d_in, d_out, dtype):
    """Express the two-layer teacher as a StudentSingleHidden.

    The teacher computes x @ W1 + b1 -> act -> h @ W2 + b2, nn.Linear computes
    x @ weight.T + bias, so both weights are transposed. Missing biases become
    zero biases, which leaves the function unchanged.
    """
    m = StudentSingleHidden(input_dim=d_in, output_dim=d_out,
                            hidden_dim=teacher.W1.shape[1],
                            activation=teacher.activation,
                            bias=True).to(dtype=dtype)
    with torch.no_grad():
        m.fc1.weight.copy_(teacher.W1.T.to(dtype))
        m.fc2.weight.copy_(teacher.W2.T.to(dtype))
        m.fc1.bias.copy_(teacher.b1.to(dtype) if teacher.b1 is not None
                         else torch.zeros_like(m.fc1.bias))
        m.fc2.bias.copy_(teacher.b2.to(dtype) if teacher.b2 is not None
                         else torch.zeros_like(m.fc2.bias))
    return m


def teacher_reference(run, refit_modes):
    """Prune the teacher, and compute its recovery rank, with the student pipeline."""
    teacher, dtype = run["teacher"], run["dtype"]
    if not (hasattr(teacher, "W1") and hasattr(teacher, "W2")):
        raise RuntimeError(
            f"--include-teacher only supports the two-layer teacher; this run "
            f"has a {type(teacher).__name__}, which has no hidden units to prune.")
    tm = wrap_teacher_as_student(teacher, run["d_in"], run["d_out"], dtype)

    with torch.no_grad():
        dev = float((teacher(run["x_te"]) - tm(run["x_te"])).abs().max())
    if dev > 1e-9:
        raise RuntimeError(f"teacher wrapper does not reproduce the teacher "
                           f"(max abs dev {dev:.3e})")

    curves = {m: greedy_neuron_curve(tm, run["x_te"], run["y_te"], dtype,
                                     run["x_fit"], run["y_fit"], refit=m)
              for m in refit_modes}
    mag = {m: magnitude_curve(tm, run["x_te"], run["y_te"],
                              run["x_fit"], run["y_fit"], refit=m)
           for m in ("none", "lstsq")}
    obd = obd_curve(tm, run["x_te"], run["y_te"], dtype,
                    behavioral_hessian_dense(tm, run["probe"]))
    rec = recovery_curves(tm, run["probe"], run["x_te"], run["y_te"], dtype)

    return {
        "hidden": int(teacher.W1.shape[1]),
        "p": int(sum(q.numel() for q in tm.parameters())),
        "neuron_by_refit": {k: list(v) for k, v in curves.items()},
        "magnitude_by_refit": {k: [[int(a), float(b)] for a, b in v]
                               for k, v in mag.items()},
        "obd": [[int(a), float(b)] for a, b in obd],
        "recovery_by_order": {k: list(v) for k, v in rec.items()},
        "equiv_max_dev": dev,
    }


# ------------------------------------------------------------------- pruning

def test_mse(model, x, y) -> float:
    with torch.no_grad():
        return float(((model(x) - y) ** 2).mean())


def n_params(n_neurons, d_in, d_out) -> int:
    return n_neurons * (d_in + 1 + d_out) + d_out


def _hidden(model, x, mask):
    pre = model.fc1(x)
    act = getattr(model, "activation", "relu")
    if act == "relu":
        h = torch.relu(pre)
    elif act == "tanh":
        h = torch.tanh(pre)
    elif act == "linear":
        h = pre
    else:
        raise ValueError(f"unsupported activation for pruning: {act}")
    return h * mask


def fit_output_layer(model, x_fit, y_fit, mask, mode: str):
    """Output layer to use after masking hidden units.

      mode='none'   keep the trained (W2, b2)
      mode='lstsq'  least-squares refit of W2[:, kept] and b2 on (x_fit, y_fit)

    The output is linear in the surviving activations, so the refit is closed
    form. It is fitted on the training set, never on the evaluation split.
    """
    W2 = model.fc2.weight.detach().clone()
    b2 = model.fc2.bias.detach().clone()
    if mode == "none":
        return W2, b2
    if mode != "lstsq":
        raise ValueError(f"unknown refit mode {mode!r}")
    with torch.no_grad():
        h_full = _hidden(model, x_fit, torch.ones_like(mask))
        keep = (mask != 0)
        if int(keep.sum()) == 0:
            return W2 * 0, y_fit.mean(dim=0)
        A = torch.cat([h_full[:, keep], torch.ones_like(h_full[:, :1])], dim=1)
        sol = torch.linalg.lstsq(A, y_fit).solution        # (|keep|+1) x d_out
        W2_new = torch.zeros_like(W2)
        W2_new[:, keep] = sol[:-1].T
        return W2_new, sol[-1]


def masked_mse(model, x_fit, y_fit, x_eval, y_eval, mask, mode) -> float:
    """MSE on (x_eval, y_eval) with the masked units removed and the output
    layer given by fit_output_layer(mode)."""
    with torch.no_grad():
        W2, b2 = fit_output_layer(model, x_fit, y_fit, mask, mode)
        h = _hidden(model, x_eval, mask)
        return float(((h @ W2.T + b2 - y_eval) ** 2).mean())


def greedy_neuron_curve(model, x_te, y_te, dtype, x_fit, y_fit,
                        refit: str = "none") -> list[float]:
    """curve[n] = test MSE with n hidden units left, removed greedily.

    At each step every remaining unit is tried, and the one whose removal (after
    the output-layer refit, if any) gives the lowest training MSE is deleted.
    The selection uses the training set only; the recorded curve is test MSE.
    """
    d = model.fc1.weight.shape[0]
    mask = torch.ones(d, dtype=dtype)

    def sel(m):
        return masked_mse(model, x_fit, y_fit, x_fit, y_fit, m, refit)

    def ev(m):
        return masked_mse(model, x_fit, y_fit, x_te, y_te, m, refit)

    curve = [0.0] * (d + 1)
    curve[d] = ev(mask)
    for remaining in range(d, 0, -1):
        best_j, best = None, float("inf")
        for j in range(d):
            if mask[j] == 0:
                continue
            mask[j] = 0
            loss = sel(mask)
            mask[j] = 1
            if loss < best:
                best_j, best = j, loss
        mask[best_j] = 0
        curve[remaining - 1] = ev(mask)
    return curve


def magnitude_curve(model, x_te, y_te, x_fit, y_fit, refit: str = "none",
                    steps: int = 60):
    """Global one-shot magnitude pruning: list of (kept params, test MSE).

    With refit='lstsq' the surviving output weights are refitted by least
    squares, each output row on its own surviving support (and its bias only
    if the bias survived), so no pruned parameter is restored. W1 is left as
    pruned.
    """
    w = [model.fc1.weight, model.fc1.bias, model.fc2.weight, model.fc2.bias]
    orig = [t.detach().clone() for t in w]
    flat = torch.cat([t.reshape(-1) for t in orig]).abs().sort().values
    total = flat.numel()
    out = []
    for frac in torch.linspace(0.0, 0.98, steps):
        n_prune = int(float(frac) * total)
        thresh = -1.0 if n_prune == 0 else float(flat[n_prune - 1])
        with torch.no_grad():
            kept = 0
            for t, o in zip(w, orig):
                keep = o.abs() > thresh
                t.copy_(o * keep)
                kept += int(keep.sum())

            if refit == "lstsq":
                # Features are computed with the pruned W1.
                h = _hidden(model, x_fit, torch.ones(model.fc1.weight.shape[0],
                                                     dtype=x_fit.dtype))
                support = model.fc2.weight.detach() != 0
                bias_support = model.fc2.bias.detach() != 0
                W_new = torch.zeros_like(model.fc2.weight)
                b_new = torch.zeros_like(model.fc2.bias)
                ones = torch.ones_like(h[:, :1])
                for o_idx in range(W_new.shape[0]):
                    cols = support[o_idx]
                    use_bias = bool(bias_support[o_idx])
                    blocks = ([h[:, cols]] if int(cols.sum()) else [])
                    if use_bias:
                        blocks.append(ones)
                    if not blocks:
                        continue        # row fully pruned: it stays at zero
                    A = torch.cat(blocks, dim=1)
                    sol = torch.linalg.lstsq(A, y_fit[:, o_idx:o_idx + 1]).solution
                    if use_bias:
                        b_new[o_idx] = sol[-1, 0]
                        sol = sol[:-1]
                    if int(cols.sum()):
                        W_new[o_idx, cols] = sol[:, 0]
                model.fc2.weight.copy_(W_new)
                model.fc2.bias.copy_(b_new)

            out.append((kept, test_mse(model, x_te, y_te)))
            for t, o in zip(w, orig):
                t.copy_(o)
    return out


def behavioral_hessian_dense(model, x_probe) -> torch.Tensor:
    """H_beh = J_P^T J_P / |P|, the Hessian of the behavioral loss at theta*.

    Built from the explicit Jacobian of the outputs on the probe set (forward
    mode, p <= 430 here). Parameter order is that of flatten_params.
    """
    names = [n for n, q in model.named_parameters() if q.requires_grad]
    layout = get_param_layout(model)
    theta = flatten_params(model).clone()

    def f(th):
        chunks = th.split(layout.sizes)
        params = {n: c.view(s) for n, c, s in zip(names, chunks, layout.shapes)}
        return torch.func.functional_call(model, params, (x_probe,)).reshape(-1)

    J = torch.func.jacfwd(f)(theta)
    return J.T @ J / x_probe.shape[0]


def obd_curve(model, x_te, y_te, dtype, H, steps: int = 60) -> list[tuple[int, float]]:
    """Optimal Brain Damage: global one-shot pruning in ascending order of the
    saliency 0.5 * H_ii * theta_i^2. Returns (kept params, test MSE) pairs."""
    layout = get_param_layout(model)
    theta = flatten_params(model).clone()
    p = theta.numel()
    saliency = 0.5 * torch.diagonal(H).clone() * theta ** 2
    order = torch.argsort(saliency)

    out = []
    for frac in torch.linspace(0.0, 0.98, steps):
        n_prune = int(float(frac) * p)
        mask = torch.ones(p, dtype=dtype, device=theta.device)
        if n_prune:
            mask[order[:n_prune]] = 0.0
        set_params_from_flat(model, theta * mask, layout)
        out.append((p - n_prune, test_mse(model, x_te, y_te)))
    set_params_from_flat(model, theta, layout)
    return out


# ------------------------------------------------------------- recovery rank

def recovery_curves(model, x_probe, x_te, y_te, dtype) -> dict[str, list[float]]:
    """Test MSE of Pi_k(theta*) for k = 1..K, for both direction orderings.

      'eigenvalue'  directions in descending lambda_j.
      'optimal'     directions in descending lambda_j * c_j^2, c_j = <v_j, theta*>.
                    Under the quadratic model the behavioral loss left after
                    keeping a set S is 0.5 * sum_{j not in S} lambda_j c_j^2, so
                    this ordering is optimal for every budget k.

    Both orderings use one orthonormal basis, built in eigenvalue order.
    """
    layout = get_param_layout(model)
    theta_star = flatten_params(model).clone()
    evals, evecs, state = compute_behavioral_eigenspace_dual(
        model=model, x_probe=x_probe, top_m=-1, device=DEVICE, dtype=dtype,
        min_relative_eval=1e-6,
    )

    cols, lams = [], []
    for j in range(len(evals)):
        v = recover_behavioral_param_eigenvector_lazy(evecs[:, j], evals[j], state, DEVICE, dtype)
        for _ in range(2):
            for b in cols:
                v = v - torch.dot(b, v) * b
        if float(v.norm()) < 1e-6:
            continue
        cols.append(v / v.norm())
        lams.append(float(evals[j]))
    if not cols:
        return {name: [] for name in DIRECTION_ORDERS}

    coeffs = [float(torch.dot(v, theta_star)) for v in cols]
    cost = [lams[j] * coeffs[j] ** 2 for j in range(len(cols))]

    out: dict[str, list[float]] = {}
    for name in DIRECTION_ORDERS:
        if name == "optimal":
            order = sorted(range(len(cols)), key=lambda j: -cost[j])
        else:
            order = list(range(len(cols)))
        acc, curve = torch.zeros_like(theta_star), []
        for j in order:
            acc = acc + coeffs[j] * cols[j]
            set_params_from_flat(model, acc, layout)
            curve.append(test_mse(model, x_te, y_te))
        out[name] = curve
    set_params_from_flat(model, theta_star, layout)
    return out


def _first_crossing(curve, tau):
    """Smallest kept-parameter count whose MSE is at or below tau."""
    return next((k for k, v in sorted(curve) if v <= tau), float("nan"))


def analyse(run, tau, refit_modes):
    model, dtype = run["model"], run["dtype"]
    x_te, y_te, x_fit, y_fit = run["x_te"], run["y_te"], run["x_fit"], run["y_fit"]
    star = test_mse(model, x_te, y_te)

    neuron_curves = {mode: greedy_neuron_curve(model, x_te, y_te, dtype,
                                               x_fit, y_fit, refit=mode)
                     for mode in refit_modes}
    nc = neuron_curves[refit_modes[0]]
    n_tau = next((n for n in range(len(nc)) if nc[n] <= tau), float("nan"))
    p_of_n = (n_params(int(n_tau), run["d_in"], run["d_out"])
              if n_tau == n_tau else float("nan"))

    magnitude_curves = {m: magnitude_curve(model, x_te, y_te, x_fit, y_fit, refit=m)
                        for m in ("none", "lstsq")}
    m_tau = _first_crossing(magnitude_curves["none"], tau)

    oc = obd_curve(model, x_te, y_te, dtype,
                   behavioral_hessian_dense(model, run["probe"]))
    o_tau = _first_crossing(oc, tau)

    rc_by_order = recovery_curves(model, run["probe"], x_te, y_te, dtype)
    rc = rc_by_order["eigenvalue"]
    r_tau = next((i + 1 for i, v in enumerate(rc) if v <= tau), float("nan"))

    p = sum(q.numel() for q in model.parameters())
    return dict(p=p, star=star, tau=tau, r_tau=r_tau, n_tau=n_tau, p_of_n=p_of_n,
                m_tau=m_tau, o_tau=o_tau, neuron_curves=neuron_curves,
                recovery_by_order=rc_by_order,
                magnitude_curves={k: [[int(a), float(b)] for a, b in v]
                                  for k, v in magnitude_curves.items()},
                obd_curve=[[int(a), float(b)] for a, b in oc],
                r_floor=(min(rc) if rc else float("nan")))


# -------------------------------------------------------------------- figure

REFIT_SLUG = {"none": "norefit", "lstsq": "lsqrefit"}

# Error-bar style shared with plot_ts_rank.py.
_EB = dict(fmt="none", elinewidth=1.0, capsize=2.0, alpha=0.25, zorder=0)


def _style():
    """Shared rcParams (from ts_plot_utils) with TrueType font embedding."""
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    from ts_plot_utils import PAPER_FIGSIZE      # importing applies the rcParams
    return PAPER_FIGSIZE


def _make_legend_headroom(fig, ax, leg, ymax, max_iter=5):
    """Raise the top of the axes until the legend clears the data.

    Iterated because the legend is placed in axes coordinates and moves up
    with the limit.
    """
    for _ in range(max_iter):
        try:
            fig.canvas.draw()
            box = leg.get_window_extent().transformed(ax.transData.inverted())
        except Exception:
            return
        if box.y0 >= ymax:
            return
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (ymax - box.y0))


def _integer_ticks(ax, anchor):
    """Evenly spaced integer x ticks that include the teacher width."""
    lo, hi = ax.get_xlim()
    step = 4 if (hi - lo) > 12 else 2
    start = float(anchor)
    while start - step >= lo:
        start -= step
    ticks, t = [], start
    while t <= hi:
        if t >= lo:
            ticks.append(int(round(t)))
        t += step
    if ticks:
        ax.set_xticks(ticks)


def _run_tag(args) -> str:
    """Filename tag; dots are written as 'p' so LaTeX's graphicx accepts it."""
    tag = "tau" + f"{args.tau_abs:g}".replace(".", "p")
    if args.include_teacher:
        tag += "_withteacher"
    if args.show_obd:
        tag += "_obd"
    return tag


def _draw_piecewise(ax, widths, fit_path):
    """Overlay the piecewise-linear fit of the eigenvalue-ordered recovery rank,
    read from the piecewise_fit.json written by
    plot_ts_rank.py."""
    try:
        with open(fit_path) as fh:
            fit = json.load(fh)
    except Exception as exc:
        print(f"  [warn] no piecewise fit drawn ({exc})")
        return
    d_star = float(fit.get("d_star", float("nan")))
    x = np.linspace(float(min(widths)), float(max(widths)), 500)
    for side, mask in (("left", x <= d_star), ("right", x >= d_star)):
        m, b = fit.get(f"m_{side}"), fit.get(f"b_{side}")
        if m is None or b is None or not np.isfinite(m) or not np.isfinite(b):
            continue
        ax.plot(x[mask], m * x[mask] + b, linestyle="-", linewidth=2.4,
                color="black", zorder=3)


def _crossing(curve, tau, which, d_in, d_out):
    """Degrees of freedom at which `curve` (of method `which`) first reaches tau."""
    if which == "neuron":
        n = next((n for n, y in enumerate(curve) if y <= tau), None)
        return n_params(n, d_in, d_out) if n is not None else None
    if which in ("recovery", "recovery_opt"):
        return next((i + 1 for i, y in enumerate(curve) if y <= tau), None)
    return next((k for k, y in sorted(curve) if y <= tau), None)


def _curve(entry, which, refit):
    """The stored curve of method `which` in a per-run (or teacher) entry."""
    if which == "recovery":
        return entry["recovery_by_order"].get("eigenvalue")
    if which == "recovery_opt":
        return entry["recovery_by_order"].get("optimal")
    if which == "neuron":
        return entry["neuron_by_refit"].get(refit)
    if which == "magnitude":
        return entry["magnitude_by_refit"].get(refit)
    return entry.get(which)


def _crossings(res, d, tau, which, refit, d_in, d_out):
    """Per-seed degrees of freedom of method `which` at width d (nan if unreached)."""
    out = []
    for key, c in res["curves"].items():
        if int(key.partition("_")[0]) != d:
            continue
        curve = _curve(c, which, refit)
        v = _crossing(curve, tau, which, d_in, d_out) if curve is not None else None
        out.append(np.nan if v is None else v)
    return out


DOF_METHODS = ("recovery", "recovery_opt", "neuron", "magnitude", "obd")


def write_dof_table(res, path: Path, d_in, d_out, refit) -> None:
    """Degrees of freedom at the tau crossing per width and method: mean, std, n."""
    tau = res["tau"]
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tau", "width", "p", "method", "refit", "mean", "std",
                     "n_reached", "n_seeds", "values"])
        for d in res["widths"]:
            for which in DOF_METHODS:
                vals = _crossings(res, d, tau, which, refit, d_in, d_out)
                if not vals:
                    continue
                fin = [v for v in vals if v == v]
                wr.writerow([tau, d, res["p"].get(str(d)), which, refit,
                             float(np.mean(fin)) if fin else "",
                             float(np.std(fin)) if fin else "",
                             len(fin), len(vals),
                             " ".join(str(v) for v in vals)])
    print(f"  wrote {path}")


def figure_dof(res, out: Path, teacher_width, d_in, d_out, refit, piecewise_fit=None,
               show_obd=False):
    """Degrees of freedom at the tau crossing vs. hidden width, mean +- std over
    seeds, with the teacher's values (if computed) as dash-dotted lines."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figsize = _style()
    widths = res["widths"]
    tau = res["tau"]
    fig, ax = plt.subplots(figsize=figsize)

    # (key, colour, marker): circles for the recovery ranks, squares for pruning.
    methods = [("recovery", "C1", "o"), ("recovery_opt", "C3", "o"),
               ("neuron", "C0", "s"), ("magnitude", "C2", "s")]
    if show_obd:
        methods.append(("obd", "C5", "s"))
    ymax = 0.0

    for which, colour, mk in methods:
        vals = [_crossings(res, d, tau, which, refit, d_in, d_out) for d in widths]
        m = np.array([np.nanmean(v) if np.any(np.isfinite(v)) else np.nan
                      for v in vals], float)
        sd = np.array([np.nanstd(v) if np.any(np.isfinite(v)) else np.nan
                       for v in vals], float)
        ax.plot(widths, m, linewidth=1.2, linestyle="-", marker=mk, markersize=4.0,
                color=colour, alpha=1.0, zorder=1)
        ax.errorbar(widths, m, yerr=sd, ecolor=colour, **_EB)
        if np.isfinite(m).any():
            ymax = max(ymax, float(np.nanmax(m + np.nan_to_num(sd))))

    # Teacher control: the same methods applied to the teacher, in the colour of
    # the corresponding student curve.
    tref = res.get("teacher_ref") or {}
    for which, colour, _ in methods:
        vals = []
        for _sd, tr in sorted(tref.items()):
            curve = _curve(tr, which, refit)
            if not curve:
                continue
            v = _crossing(curve, tau, which, d_in, d_out)
            if v is not None:
                vals.append(int(v))
        if not vals:
            continue
        mv, sv = float(np.mean(vals)), float(np.std(vals))
        ax.axhline(mv, color=colour, ls="-.", lw=1.0, alpha=0.75)
        if sv > 0:
            ax.axhspan(mv - sv, mv + sv, color=colour, alpha=0.08, lw=0)
        ymax = max(ymax, mv + sv)

    if piecewise_fit:
        _draw_piecewise(ax, widths, piecewise_fit)
    ax.axvline(float(teacher_width), linestyle=":", linewidth=1.5, color="black")
    ax.set_ylim(0, ymax * 1.02 if ymax > 0 else None)

    # Two-column legend: recovery ranks on the left, pruning methods on the right.
    empty = Line2D([], [], linestyle="none")
    style = {k: (c, mk) for k, c, mk in methods}

    def _proxy(key):
        colour, mk = style[key]
        return Line2D([0], [0], color=colour, marker=mk, markersize=4.0,
                      linewidth=1.2, linestyle="-")

    left = [(empty, rf"$\mathbf{{r_\tau}}$, $\tau={tau:g}$"),
            (_proxy("recovery"), "eigenvalue"),
            (_proxy("recovery_opt"), "loss-optimal")]
    right = [(empty, r"$\mathbf{Pruning}$"),
             (_proxy("neuron"), "structure"),
             (_proxy("magnitude"), "magnitude")]
    if show_obd:
        right.append((_proxy("obd"), "OBD"))
    while len(left) < len(right):
        left.append((empty, ""))
    while len(right) < len(left):
        right.append((empty, ""))
    h = [x for x, _ in left] + [x for x, _ in right]
    l = [s for _, s in left] + [s for _, s in right]
    leg = ax.legend(h, l, ncol=2, loc="upper left", frameon=True,
                    framealpha=0.9, handlelength=1.0, handletextpad=0.25,
                    labelspacing=0.05, handleheight=0.5, borderpad=0.15,
                    borderaxespad=0.15, columnspacing=0.5)
    _make_legend_headroom(fig, ax, leg, ymax)

    _integer_ticks(ax, anchor=teacher_width)
    ax.set_xlabel(r"Hidden width $d$")
    ax.set_ylabel("degrees of freedom")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-root", type=str, required=True,
                    help="sweep directory with the trained teacher-student runs")
    ap.add_argument("--widths", type=int, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--teacher-width", type=int, default=10)
    ap.add_argument("--tau-abs", type=float, default=0.075,
                    help="absolute test-MSE threshold, identical for every width")
    ap.add_argument("--refit-output", type=str, nargs="+", default=["none"],
                    choices=["none", "lstsq"],
                    help="output-layer refit after structure/magnitude pruning: "
                         "'none' (pure deletion) or 'lstsq' (closed-form least "
                         "squares on the training set). One figure per mode.")
    ap.add_argument("--include-teacher", action="store_true",
                    help="also prune the teacher (once per seed) and draw its "
                         "counts as reference lines")
    ap.add_argument("--show-obd", action="store_true",
                    help="add Optimal Brain Damage to the figure")
    ap.add_argument("--piecewise-fit", type=str, default=None,
                    help="piecewise_fit.json written by "
                         "plot_ts_rank.py; "
                         "its trend lines are overlaid on the recovery rank")
    ap.add_argument("--out", type=str, default="figures/compression",
                    help="output directory and filename prefix")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tau = args.tau_abs
    refit_modes = tuple(args.refit_output)

    root = Path(args.checkpoint_root)
    found = []
    for cfg in sorted(root.glob("in*_th*_out*_sd*_seed*")):
        m = CFG_RE.search(cfg.name)
        if not m:
            continue
        th, d, seed = int(m.group(2)), int(m.group(4)), int(m.group(6))
        if th != args.teacher_width:
            continue
        if args.widths and d not in args.widths:
            continue
        if args.seeds and seed not in args.seeds:
            continue
        found.append((d, seed, cfg))
    if not found:
        ap.error(f"no runs matching teacher width {args.teacher_width} under {root}")
    widths = sorted({d for d, _, _ in found})
    seeds = sorted({s for _, s, _ in found})
    teacher_width = args.teacher_width
    print(f"found {len(found)} run(s) under {root}")
    print(f"teacher width {teacher_width}; tau = {tau:g}; refit = {list(refit_modes)}\n")

    res = {"widths": widths, "seeds": seeds, "teacher_width": teacher_width,
           "source": args.checkpoint_root, "tau_abs": [tau],
           "p": {}, "r_tau": {}, "n_tau": {}, "p_of_n": {}, "m_tau": {}, "o_tau": {},
           "test_mse_star": {}, "r_floor": {}, "curves": {}}
    for d in widths:
        for key in ("r_tau", "n_tau", "p_of_n", "m_tau", "o_tau",
                    "test_mse_star", "r_floor"):
            res[key][str(d)] = []

    print("   d   seed     p   test_mse*   r_floor     r_tau   neurons  params(n)"
          "   magnitude     o_tau")
    d_in = d_out = None
    teacher_cfg: dict[int, Path] = {}
    for d, seed, cfg in sorted(found):
        teacher_cfg.setdefault(seed, cfg)
        run = load_run(cfg)
        if run is None:
            print(f"  {d:3d}  {seed:5d}   -- skipped (no checkpoint/summary)")
            continue
        d_in, d_out = run["d_in"], run["d_out"]
        a = analyse(run, tau, refit_modes)
        res["p"][str(d)] = a["p"]
        for key in ("r_tau", "n_tau", "p_of_n", "m_tau", "o_tau"):
            res[key][str(d)].append(a[key])
        res["test_mse_star"][str(d)].append(a["star"])
        res["r_floor"][str(d)].append(a["r_floor"])
        res["curves"][f"{d}_{seed}"] = {
            "tau": a["tau"], "test_mse_star": a["star"],
            "neuron_by_refit": a["neuron_curves"],
            "recovery_by_order": a["recovery_by_order"],
            "magnitude_by_refit": a["magnitude_curves"],
            "obd": a["obd_curve"],
        }
        print(f"  {d:3d}  {seed:5d}  {a['p']:4d}   {a['star']:.2e}  {a['r_floor']:.2e}   "
              f"{a['r_tau']:7}   {a['n_tau']:7}   {a['p_of_n']:8}   {a['m_tau']:9}"
              f"   {a['o_tau']:7}", flush=True)

    if args.include_teacher:
        print(f"\nteacher, pruned with the same pipeline ({len(teacher_cfg)} seed(s))")
        res["teacher_ref"] = {}
        for sd, cfg in sorted(teacher_cfg.items()):
            r = load_run(cfg)
            if r is None:
                continue
            tr = teacher_reference(r, refit_modes)
            res["teacher_ref"][str(sd)] = tr
            cells = [f"{w}={_crossing(_curve(tr, w, mode), tau, w, d_in, d_out)}"
                     for mode in refit_modes
                     for w in ("recovery", "recovery_opt", "neuron", "magnitude", "obd")
                     if _curve(tr, w, mode)]
            print(f"  seed {sd}: " + "  ".join(cells))

    stem = f"{out.name}_{_run_tag(args)}"
    json_path = out.parent / f"{stem}.json"
    with json_path.open("w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  wrote {json_path}")
    res["tau"] = tau
    for mode in refit_modes:
        slug = REFIT_SLUG[mode]
        figure_dof(res, out.parent / f"{stem}_{slug}_dof.pdf", teacher_width,
                   d_in, d_out, refit=mode, piecewise_fit=args.piecewise_fit,
                   show_obd=args.show_obd)
        write_dof_table(res, out.parent / f"{stem}_{slug}_dof.csv",
                        d_in, d_out, refit=mode)


if __name__ == "__main__":
    main()
