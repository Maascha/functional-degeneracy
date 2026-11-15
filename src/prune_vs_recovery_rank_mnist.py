"""Behavioral recovery rank vs. one-shot pruning on width-scaled MNIST MLPs.

For each trained width-w MLP (784 -> w -> w -> 10) and an absolute test-accuracy
threshold tau, reports the number of degrees of freedom needed to keep the test
accuracy >= tau:

  recovery     r_tau with eigenvalue ordering: the smallest k such that
               theta_0 + Pi_k (theta* - theta_0) reaches tau. Read from the
               projection curve already stored with the run.
  recovery_opt r_tau with directions ordered by lambda_j <v_j, theta* - theta_0>^2,
               computed in probe-output (dual) space from the cached eigenpairs,
               evaluated at every k until tau is reached (--optimal-grid N
               evaluates N evenly spaced k instead: faster, gives an upper bound on the rank).
  neuron       greedy structural pruning: hidden units are removed one at a time
               from a pool spanning both hidden layers, each step removing the unit
               whose removal costs least accuracy on the probe set; the count is
               all parameters attached to surviving units.
  magnitude    global one-shot magnitude pruning over all weights and biases.
  obd          Optimal Brain Damage: global one-shot pruning by the saliency
               0.5 * diag(H_beh)_q * theta_q^2.

No retraining or refitting: every number comes from deleting parts of the trained
network and evaluating it.

    python src/prune_vs_recovery_rank_mnist.py --run-root results/mnist \
        --widths 32 64 96 128 160 --device cuda --tau-abs 0.90 --show-obd \
        --out figures/mnist/pruning/mnist_dof_tau090

--run-root holds the base run folders (with model/checkpoint.pt and the cached
dual spectrum in model/eigen_cache/) next to their eigen-analysis folders
<base>_topm_<M>_curv_*_pmode_*_probesrc_<s> (with experiment_outputs.pt).
Writes <out>.pdf, <out>.json and <out>_dof.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# torch is imported before numpy on purpose: the import order decides which MKL
# runtime settings load, and the Lanczos eigensolver is sensitive to the
# resulting round-off (see prune_vs_recovery_rank.py).
import torch  # noqa: I001
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from curvature_projection import compute_behavioral_eigenspace_dual
from run_mnist_mlp_lazy import SmallMLP, load_mnist_subsets, set_seed
from train import flatten_params, get_param_layout, set_params_from_flat
from ts_plot_utils import PAPER_FIGSIZE, TICK_FS

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DEVICE = torch.device("cpu")
PROBE_SOURCE = "train"   # probe set: training images
PROBE_SIZE = 10000       # must match the cached spectrum
TOP_M = 12000            # leading eigenpairs used
MAGNITUDE_STEPS = 2000   # sparsity levels swept by magnitude pruning and OBD


# --------------------------------------------------------------------- metric

def test_accuracy(model, x, y) -> float:
    with torch.no_grad():
        return float((model(x).argmax(dim=1) == y).float().mean())


def first_crossing(values, tau: float):
    """Smallest budget whose accuracy is >= tau, or None.

    `values` is a list of (budget, accuracy) pairs in increasing budget order.
    """
    for budget, score in values:
        if score >= tau:
            return budget
    return None


# ------------------------------------------------------------------- loading

def find_runs(root: Path, widths=None):
    """MNIST base runs with h1 = h2, as (width, seed, run_dir).

    Runs are found by their model/checkpoint.pt; width and seed are read from
    the checkpoint. The immediate children are walked explicitly in addition to
    rglob, which does not follow symlinked directories.
    """
    candidates = set(root.rglob("model/checkpoint.pt"))
    for child in root.iterdir():
        ckpt = child / "model" / "checkpoint.pt"
        if ckpt.is_file():
            candidates.add(ckpt)

    found = []
    for ckpt_path in sorted(candidates):
        cfg = ckpt_path.parent.parent
        meta = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        h1, h2 = (int(h) for h in meta["model_config"]["hidden_dims"])
        if h1 != h2:
            print(f"  [skip] {cfg.name}: hidden widths differ ({h1},{h2})")
            continue
        if widths and h1 not in widths:
            continue
        found.append((h1, int(meta["seed"]), cfg))
    return found


def resolve_cache_path(cfg: Path) -> Path | None:
    """Cached dual spectrum written by run_mnist_mlp_lazy for this probe set, or None."""
    path = cfg / "model" / "eigen_cache" / f"behavioral_dual_top_{PROBE_SOURCE}_n{PROBE_SIZE}.pt"
    if not path.is_file():
        print(f"    [warn] no cached spectrum {path}; the eigensolve is run from scratch")
        return None
    return path


def load_run(cfg_dir: Path, data_root: Path):
    """Rebuild one MNIST run: the same data subsets, normalization and weights.

    The data are regenerated with the run's seed and subset sizes; the recomputed
    test accuracy is checked against the logged one.
    """
    ckpt = torch.load(cfg_dir / "model" / "checkpoint.pt",
                      map_location=DEVICE, weights_only=False)
    hidden = tuple(int(h) for h in ckpt["model_config"]["hidden_dims"])
    seed = int(ckpt["seed"])

    set_seed(seed)
    x_tr, y_tr, x_te, y_te, x_pr, y_pr, _stats = load_mnist_subsets(
        data_root=data_root,
        train_subset_size=int(ckpt["train_subset_size"]),
        test_subset_size=int(ckpt["test_subset_size"]),
        probe_size=PROBE_SIZE,
        probe_source=PROBE_SOURCE,
        seed=seed,
        device=DEVICE,
    )

    model = SmallMLP(hidden_dims=hidden).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    acc = test_accuracy(model, x_te, y_te)
    reported = ckpt.get("accs_test")
    if isinstance(reported, (list, tuple)) and reported:
        if abs(acc - float(reported[-1])) > 1e-6:
            print(f"    WARNING {cfg_dir.name}: recomputed test accuracy {acc:.4f} "
                  f"!= logged {float(reported[-1]):.4f}; the regenerated data do "
                  f"not match the run.")
    return dict(model=model, hidden=hidden, seed=seed, probe=x_pr, y_probe=y_pr,
                x_te=x_te, y_te=y_te, theta_0=ckpt["theta_0"].to(DEVICE),
                theta_star=flatten_params(model).clone(), acc_star=acc)


# ------------------------------------------------------------------- pruning

def _layers(model):
    """The three Linear layers of SmallMLP, in order."""
    return model.net[0], model.net[2], model.net[4]


def _ones(model, n: int):
    """Unit mask on the model's device and dtype."""
    ref = next(model.parameters())
    return torch.ones(n, device=ref.device, dtype=ref.dtype)


def n_params(n1: int, n2: int, d_in: int = 784, d_out: int = 10) -> int:
    """Parameters attached to n1 and n2 surviving hidden units."""
    return n1 * (d_in + 1) + n2 * (n1 + 1) + d_out * (n2 + 1)


def _masked_accuracy(model, x, y, m1, m2) -> float:
    """Accuracy with hidden units masked out of both hidden layers."""
    l1, l2, l3 = _layers(model)
    with torch.no_grad():
        h1 = torch.relu(l1(x)) * m1
        h2 = torch.relu(l2(h1)) * m2
        logits = h2 @ l3.weight.T + l3.bias
        return float((logits.argmax(dim=1) == y).float().mean())


def greedy_neuron_curve(run):
    """Greedy unit removal from a pool spanning both hidden layers.

    Each step removes the surviving unit whose removal keeps the probe-set
    accuracy highest (tried exactly for every unit). Units in the two layers
    carry different numbers of parameters, so the choice is scored on accuracy
    and converted to parameters afterwards. Returns (retained parameters,
    test accuracy) in increasing budget order.
    """
    model, x, y = run["model"], run["x_te"], run["y_te"]
    xs, ys = run["probe"], run["y_probe"]
    h1, h2 = run["hidden"]
    m1, m2 = _ones(model, h1), _ones(model, h2)

    curve = [(n_params(int(m1.sum()), int(m2.sum())), _masked_accuracy(model, x, y, m1, m2))]
    while m1.sum() + m2.sum() > 1:
        best = None
        for layer, mask in ((1, m1), (2, m2)):
            for j in range(len(mask)):
                if mask[j] == 0:
                    continue
                mask[j] = 0
                score = _masked_accuracy(model, xs, ys, m1, m2)
                mask[j] = 1
                if best is None or score > best[0]:
                    best = (score, layer, j)
        _, layer, j = best
        (m1 if layer == 1 else m2)[j] = 0
        curve.append((n_params(int(m1.sum()), int(m2.sum())), _masked_accuracy(model, x, y, m1, m2)))
    curve.reverse()
    return curve


def magnitude_curve(run, steps: int = MAGNITUDE_STEPS):
    """Global one-shot magnitude pruning across all weight and bias tensors.

    Returns (surviving nonzeros, test accuracy) at `steps` sparsity levels in [0, 0.98].
    """
    model = run["model"]
    l1, l2, l3 = _layers(model)
    w = [l1.weight, l1.bias, l2.weight, l2.bias, l3.weight, l3.bias]
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
            out.append((kept, test_accuracy(model, run["x_te"], run["y_te"])))
            for t, o in zip(w, orig):
                t.copy_(o)
    out.sort()
    return out


def behavioral_hessian_diag(run, batch: int = 64) -> torch.Tensor:
    """diag(H_beh) = (1/|P|) sum_{x in P} sum_c (d f_c(x) / d theta)^2, f = logits.

    H_beh = J_P^T J_P / |P| is the matrix whose eigenpairs define r_tau. The
    per-sample, per-logit gradients come from vmap over jacrev, a batch of
    probe images at a time.
    """
    model = run["model"]
    names = [n for n, _ in model.named_parameters()]
    params = {n: q.detach() for n, q in model.named_parameters()}

    def f_single(prm, x):
        return torch.func.functional_call(model, prm, (x[None],))[0]

    jac = torch.func.vmap(torch.func.jacrev(f_single), in_dims=(None, 0))
    diag = {n: torch.zeros_like(q) for n, q in params.items()}
    x_pr = run["probe"]
    for i in range(0, x_pr.shape[0], batch):
        J = jac(params, x_pr[i:i + batch])          # name -> (B, 10, *shape)
        for n in names:
            diag[n] += (J[n] ** 2).sum(dim=(0, 1))
    return torch.cat([diag[n].reshape(-1) for n in names]) / x_pr.shape[0]


def obd_curve(run, steps: int = MAGNITUDE_STEPS):
    """Optimal Brain Damage (LeCun et al. 1989) on diag(H_beh): global one-shot
    pruning of the parameters with the smallest saliency 0.5 * [H_beh]_qq * theta_q^2,
    no compensation. Same protocol as magnitude_curve; only the ranking differs.
    """
    model = run["model"]
    layout = get_param_layout(model)
    theta = flatten_params(model).clone()
    saliency = 0.5 * behavioral_hessian_diag(run) * theta ** 2
    order = torch.argsort(saliency)
    total = theta.numel()

    out = []
    for frac in torch.linspace(0.0, 0.98, steps):
        n_prune = int(float(frac) * total)
        keep = torch.ones_like(theta)
        if n_prune:
            keep[order[:n_prune]] = 0.0
        set_params_from_flat(model, theta * keep, layout)
        kept = int((theta * keep != 0).sum())
        out.append((kept, test_accuracy(model, run["x_te"], run["y_te"])))
    set_params_from_flat(model, theta, layout)
    out.sort()
    return out


# ------------------------------------------------------------------ recovery

def load_archived_recovery(cfg: Path) -> list[tuple[int, float]]:
    """Eigenvalue-ordered projection curve (k, test accuracy) with theta_ref = theta_0,
    as stored by run_mnist_mlp_lazy in <base>_topm_<M>_curv_*_pmode_*_probesrc_<s>/."""
    hits = sorted(cfg.parent.glob(f"{cfg.name}_topm_{TOP_M}_curv_*_pmode_*_probesrc_{PROBE_SOURCE}"))
    if not hits:
        raise SystemExit(f"no eigen-analysis outputs for {cfg.name} (topm={TOP_M}, "
                         f"probe source {PROBE_SOURCE}) in {cfg.parent}")
    d = torch.load(hits[-1] / "experiment_outputs.pt", map_location="cpu", weights_only=False)
    curve = sorted(zip((int(k) for k in d["k_values_behavioral"].tolist()),
                       (float(a) for a in d["test_accuracy_by_k_behavioral"].tolist())))
    print(f"    recovery curve from {hits[-1].name} "
          f"({len(curve)} points, k<={curve[-1][0]}, max acc={max(a for _, a in curve):.4f})")
    return curve


def recovery_curve_optimal(run, cache_path: Path, tau: float, n_points: int | None = None):
    """Projection curve with directions ordered by lambda_j c_j^2, c_j = <v_j, delta>,
    delta = theta* - theta_0, computed in dual (probe-output) space.

    With g = J delta and v_j = J^T u_j / sqrt(|P| lambda_j) for the dual
    eigenvectors u_j of J J^T / |P|:

        lambda_j c_j^2 = <u_j, g>^2 / |P|,
        Pi_S delta     = J^T ( sum_{j in S} <u_j, g> / (|P| lambda_j) u_j ),

    so the ordering is by |<u_j, g>| and J^T is applied once per evaluated k.

    n_points=None: every k = 1, 2, ..., stopping at the first k with test
    accuracy >= tau (all that first_crossing needs). n_points=N: N values of k
    spread evenly over [1, M] (faster; the first crossing on this grid is an
    upper bound on the rank).
    """
    model = run["model"]
    layout = get_param_layout(model)
    evals, evecs, state = compute_behavioral_eigenspace_dual(
        model=model, x_probe=run["probe"], top_m=TOP_M,
        device=DEVICE, dtype=torch.float32, cache_path=cache_path,
    )
    params_dict = state["params_dict"]
    logits_fn = state["logits_fn"]
    param_names = state["param_names"]
    batch_size = int(state["batch_size"])
    output_dim = int(state["output_dim"])

    theta_ref = run["theta_0"]
    delta = run["theta_star"] - theta_ref

    # g = J delta, one forward-mode pass.
    tangent, i = {}, 0
    for name in param_names:
        ref = params_dict[name]
        n = ref.numel()
        tangent[name] = delta[i:i + n].view_as(ref).to(ref.dtype)
        i += n
    _, jd = torch.func.jvp(logits_fn, (params_dict,), (tangent,))
    g = jd.reshape(-1).to(evecs.dtype)

    num = evecs.T @ g                                   # <u_j, g>
    a = num / torch.clamp(evals * float(batch_size), min=1e-30)
    idx = torch.argsort(-(num ** 2))

    _, vjp_fn = torch.func.vjp(logits_fn, params_dict)

    def evaluate(w):
        jt_w = vjp_fn(w.view(batch_size, output_dim))[0]
        flat = torch.cat([jt_w[name].reshape(-1) for name in param_names])
        set_params_from_flat(model, theta_ref + flat, layout)
        return test_accuracy(model, run["x_te"], run["y_te"])

    m = len(evals)
    curve, w = [], torch.zeros_like(g)
    if n_points is None:
        for k in range(1, m + 1):
            j = idx[k - 1]
            w = w + evecs[:, j] * a[j]
            curve.append((k, evaluate(w)))
            if curve[-1][1] >= tau:
                break
    else:
        ks = sorted({int(k) for k in torch.linspace(1, m, min(n_points, m)).tolist()})
        prev = 0
        for k in ks:
            sel = idx[prev:k]
            w = w + (evecs[:, sel] * a[sel]).sum(dim=1)
            prev = k
            curve.append((k, evaluate(w)))
    set_params_from_flat(model, run["theta_star"], layout)
    return curve


# -------------------------------------------------------------------- figure

def figure(res, out: Path, tau: float, show_obd: bool) -> None:
    """Retained degrees of freedom at threshold tau over hidden width.

    Circles are recovery ranks, squares pruning baselines; mean over seeds, with
    the seed standard deviation as faint error bars.
    """
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    widths = res["widths"]
    xs = list(widths)
    methods = [("recovery", "C1", "o"),
               ("recovery_opt", "C3", "o"),
               ("neuron", "C0", "s"),
               ("magnitude", "C2", "s")]
    if show_obd:
        methods.append(("obd", "C5", "s"))
    ymax = 0.0

    for key, colour, mk in methods:
        m, sd = [], []
        for w in widths:
            vals = [v for v in res["cross"].get(f"{w}|{key}|{tau:g}", []) if v is not None]
            m.append(np.mean(vals) if vals else np.nan)
            sd.append(np.std(vals) if len(vals) > 1 else 0.0)
        m = np.asarray(m, dtype=float)
        sd = np.asarray(sd, dtype=float)
        ax.plot(xs, m, linewidth=1.2, linestyle="-", marker=mk, markersize=4.0, color=colour)
        ax.errorbar(xs, m, yerr=sd, fmt="none", ecolor=colour,
                    elinewidth=1.0, capsize=2.0, alpha=0.25, zorder=0)
        if np.isfinite(m).any():
            ymax = max(ymax, float(np.nanmax(m + np.nan_to_num(sd))))

    if ymax > 0:
        ax.set_ylim(0, ymax * 1.12)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(w) for w in widths], fontsize=TICK_FS)
    ax.set_xlabel(r"Hidden width $w$")
    ax.set_ylabel("degrees of freedom")

    # Two-column key: recovery ranks on the left, pruning baselines on the right.
    empty = Line2D([], [], linestyle="none")
    style = {k: (c, mk) for k, c, mk in methods}
    drawn_keys = {k for k in style if any(f"{w}|{k}|{tau:g}" in res["cross"] for w in widths)}

    def _proxy(key):
        colour, mk = style[key]
        return Line2D([0], [0], color=colour, marker=mk, markersize=4.0,
                      linewidth=1.2, linestyle="-")

    left = [(empty, rf"$\mathbf{{r_\tau}}$, $\tau={tau:g}$")]
    for key, lab in (("recovery", "eigenvalue"), ("recovery_opt", "loss-optimal")):
        if key in drawn_keys:
            left.append((_proxy(key), lab))
    right = [(empty, r"$\mathbf{Pruning}$")]
    for key, lab in (("neuron", "structure"), ("magnitude", "magnitude"), ("obd", "OBD")):
        if key in drawn_keys:
            right.append((_proxy(key), lab))
    while len(left) < len(right):
        left.append((empty, ""))
    while len(right) < len(left):
        right.append((empty, ""))
    h = [x for x, _ in left] + [x for x, _ in right]
    l = [s for _, s in left] + [s for _, s in right]
    leg = ax.legend(h, l, ncol=2, loc="upper left", frameon=True,
                    framealpha=0.9, borderpad=0.15, borderaxespad=0.15,
                    handlelength=1.0, handletextpad=0.25,
                    labelspacing=0.05, handleheight=0.5,
                    columnspacing=0.5)
    # Raise the upper y-limit until the legend box clears the curves.
    for _ in range(5):
        fig.canvas.draw()
        box = leg.get_window_extent().transformed(ax.transData.inverted())
        if box.y0 >= ymax:
            break
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (ymax - box.y0))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def write_dof_table(res, path: Path) -> None:
    """Retained count at the tau crossing, one row per (tau, width, method)."""
    with path.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tau", "width", "p", "method", "values"])
        for key in sorted(res["cross"], key=lambda k: (float(k.split("|")[2]),
                                                       int(k.split("|")[0]),
                                                       k.split("|")[1])):
            w, method, tau = key.split("|")
            wr.writerow([tau, w, res["p"].get(w), method,
                         " ".join(str(v) for v in res["cross"][key])])
    print(f"  wrote {path}")


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=str, required=True,
                    help="directory containing the MNIST run folders")
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--widths", type=int, nargs="+", default=None,
                    help="hidden widths to include (default: all h1 = h2 runs found)")
    ap.add_argument("--tau-abs", type=float, default=0.90,
                    help="absolute test-accuracy threshold tau")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--show-obd", action="store_true",
                    help="draw the OBD curve (always computed and saved)")
    ap.add_argument("--out", type=str, default="mnist_dof",
                    help="output path prefix for the .pdf, .json and _dof.csv")
    ap.add_argument("--optimal-grid", type=int, default=None, metavar="N",
                    help="evaluate the loss-optimal curve at N evenly spaced k instead of every k "
                         "(faster, gives an upper bound on the rank)")
    args = ap.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)
    tau = args.tau_abs

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = find_runs(Path(args.run_root), args.widths)
    if not runs:
        ap.error(f"no MNIST runs with model/checkpoint.pt under {args.run_root}")
    print(f"found {len(runs)} run(s)\n")

    res = {"widths": sorted({w for w, _, _ in runs}), "p": {}, "hidden": {}, "cross": {},
           "recovery_opt_k": ("every k" if args.optimal_grid is None
                              else f"{args.optimal_grid} evenly spaced k")}
    keys = ("recovery", "recovery_opt", "neuron", "magnitude", "obd")
    print("    w   seed        p   acc*   " + "  ".join(f"{k:>7s}" for k in keys))
    for w, seed, cfg in runs:
        run = load_run(cfg, Path(args.data_root))
        res["p"][str(w)] = int(sum(q.numel() for q in run["model"].parameters()))
        res["hidden"][str(w)] = f"{run['hidden'][0]},{run['hidden'][1]}"

        curves = {
            "neuron": greedy_neuron_curve(run),
            "magnitude": magnitude_curve(run),
            "obd": obd_curve(run),
            "recovery": load_archived_recovery(cfg),
            "recovery_opt": recovery_curve_optimal(run, resolve_cache_path(cfg), tau, args.optimal_grid),
        }
        for key, curve in curves.items():
            res["cross"].setdefault(f"{w}|{key}|{tau:g}", []).append(first_crossing(curve, tau))
        row = [res["cross"][f"{w}|{k}|{tau:g}"][-1] for k in keys]
        print(f"  {w:4d}  {seed:5d}  {res['p'][str(w)]:7d}  {run['acc_star']:.4f}   "
              + "  ".join(f"{str(v):>7s}" for v in row))

    with (out.parent / f"{out.name}.json").open("w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  wrote {out.parent / (out.name + '.json')}")
    write_dof_table(res, out.parent / f"{out.name}_dof.csv")
    figure(res, out.parent / f"{out.name}.pdf", tau, args.show_obd)


if __name__ == "__main__":
    main()
