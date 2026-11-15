"""Load teacher-student sweep runs and compute behavioral recovery ranks.

A sweep root contains one directory per run, ``<root>/<cfg>/runs/<run>/``, with
``summary.json`` and ``experiment_outputs.pt`` (projection curves over k and the
behavioral-Hessian eigenvalues). ``gather_records`` turns every run into a flat
record holding

  - absolute ranks   k_<split>_tau_<tau>: first k with best_{j<=k} M(theta_hat_j) <= tau,
  - relative ranks   r_<split>_q_<q>:     first k with (L_ref - best_k) / (L_ref - L*) >= q,
  - spectral ranks   spectral_rank_<90|95|99>: eigenvalues needed for that share of the mass,
  - the number of retained eigenpairs, lambda_j >= 1e-6 lambda_1 (``STABLE_EIG_COL``),
  - the raw curves and eigenvalues (``_curves``, ``_eigs_pos``) for heatmaps and spectra.

A rank is NaN when the tolerance is not reached within the retained directions.

Reference loss of the relative rank (``relative_ref``, ``--relative-ref``):
  "theta_ref" (default): L_ref = M(theta_ref), the loss at the
              projection's own reference point: M(0) for projection_vector_mode =
              absolute (summary *_mse_zero; recomputed from the run's seed if the
              summary does not have it); M(theta_0) for displacement.
  "theta0"    L_ref = M(theta_0) in both modes (summary *_mse_theta0).
Absolute ranks r_tau do not depend on it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

STABLE_EIG_COL = "k_lambda_rel_1e-06"

RELATIVE_REF_CHOICES = ("theta_ref", "theta0")
RELATIVE_REF_DEFAULT = "theta_ref"


def add_relative_ref_arg(parser) -> None:
    """--relative-ref, for every script that computes or plots relative ranks."""
    parser.add_argument(
        "--relative-ref", default=RELATIVE_REF_DEFAULT, choices=list(RELATIVE_REF_CHOICES),
        help="Reference loss of the relative rank. theta_ref (default): loss at the projection's "
             "reference point, M(0) in absolute and M(theta_0) in displacement mode. "
             "theta0: M(theta_0) in both modes.")


def mse_at_zero_for_run(summary: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """M(0) of one run: from its summary, or recomputed (run_toy_teacher_student)."""
    if summary.get("train_mse_zero") is not None and summary.get("test_mse_zero") is not None:
        return {"train_mse_zero": summary["train_mse_zero"], "test_mse_zero": summary["test_mse_zero"]}
    from run_toy_teacher_student import recompute_mse_at_zero
    return recompute_mse_at_zero(summary, payload)


def maybe_int(x: Any) -> int | None:
    try:
        return None if x is None else int(x)
    except (TypeError, ValueError):
        return None


def maybe_float(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def sanitize_float_token(x: float) -> str:
    return f"{float(x):g}".replace("+", "")


def parse_seed_from_path(path_like: str) -> int | None:
    for pat in (r"(?:^|[_\-/])seed[_\-=]?(\d+)(?:$|[_\-/])",
                r"(?:^|[_\-/])run[_\-=]?(\d+)(?:$|[_\-/])"):
        m = re.search(pat, str(path_like), flags=re.IGNORECASE)
        if m:
            return maybe_int(m.group(1))
    return None


def to_numpy_1d(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = np.asarray([float(arr)], dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1)


def parse_eigenvalues_txt(path: Path) -> np.ndarray:
    """Eigenvalues from the ``index,value`` text file written next to each run."""
    vals: list[float] = []
    if not path.exists():
        return np.asarray(vals, dtype=float)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 2 and maybe_float(parts[1]) is not None:
                vals.append(float(parts[1]))
    return np.asarray(vals, dtype=float)


def get_behavioral_eigenvalues(payload: dict[str, Any], eig_txt_path: Path) -> np.ndarray:
    """Positive behavioral-Hessian eigenvalues, sorted in decreasing order."""
    eigvals = to_numpy_1d(payload.get("top_eigenvalues_behavioral"))
    if len(eigvals) == 0:
        eigvals = parse_eigenvalues_txt(eig_txt_path)
    eigvals = eigvals[np.isfinite(eigvals) & (eigvals > 0.0)]
    return np.sort(eigvals)[::-1] if len(eigvals) else np.array([], dtype=float)


def get_k_values(payload: dict[str, Any], curve_len: int, max_k: int | None) -> np.ndarray:
    """The k of each point of a projection curve, capped at max_k."""
    if curve_len <= 0:
        return np.array([], dtype=float)
    arr = np.arange(1, curve_len + 1, dtype=float)
    for key in ("k_values_behavioral", "k_values",
                "k_values_behavioral_displacement", "k_values_behavioral_absolute"):
        stored = to_numpy_1d(payload.get(key))
        if len(stored) >= curve_len:
            arr = stored[:curve_len]
            break
    if max_k is not None and max_k > 0:
        arr = arr[arr <= float(max_k)]
    return arr


def _first_k(mask: np.ndarray, k_values: np.ndarray) -> tuple[float, bool]:
    idx = np.where(mask)[0]
    return (float(k_values[int(idx[0])]), True) if len(idx) else (math.nan, False)


def _mse_theta0(summary: dict[str, Any], payload: dict[str, Any], split: str) -> float | None:
    """M(theta_0) on ``split``, as stored by run_toy_teacher_student."""
    for src in (summary, payload):
        v = maybe_float(src.get(f"{split}_mse_theta0"))
        if v is not None:
            return v
    return None


def load_run_record(
    run_dir: Path,
    test_thresholds: list[float],
    relative_q: list[float],
    max_k: int | None,
    relative_ref: str = RELATIVE_REF_DEFAULT,
) -> tuple[dict[str, Any], list[str]]:
    """Load one run directory into a flat record, plus any warnings."""
    if relative_ref not in RELATIVE_REF_CHOICES:
        raise ValueError(f"relative_ref must be one of {RELATIVE_REF_CHOICES}, got {relative_ref!r}")
    warnings: list[str] = []
    summary_path = run_dir / "summary.json"
    outputs_path = run_dir / "experiment_outputs.pt"

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        return {}, [f"failed to load {summary_path}: {exc}"]
    if not isinstance(summary, dict):
        return {}, [f"summary is not a dict: {summary_path}"]

    payload: dict[str, Any] = {}
    if outputs_path.exists():
        try:
            loaded = torch.load(outputs_path, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                payload = loaded
            else:
                warnings.append(f"experiment_outputs.pt is not a dict: {outputs_path}")
        except Exception as exc:
            warnings.append(f"failed to load {outputs_path}: {exc}")
    else:
        warnings.append(f"missing experiment_outputs.pt at {outputs_path}")

    d = maybe_int(summary.get("student_hidden_dim"))
    if d is None:
        warnings.append(f"missing student_hidden_dim in {summary_path}")
        return {}, warnings

    teacher_info = summary.get("teacher_info") if isinstance(summary.get("teacher_info"), dict) else {}
    teacher_hidden_dim = maybe_int(teacher_info.get("hidden_dim"))
    if teacher_hidden_dim is None:
        teacher_hidden_dim = maybe_int(summary.get("teacher_hidden_dim"))

    seed = maybe_int(summary.get("seed"))
    if seed is None:
        seed = parse_seed_from_path(str(run_dir))
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "d": int(d),
        "teacher_hidden_dim": teacher_hidden_dim,
        "seed": 0 if seed is None else seed,
        "probe_source": str(summary.get("probe_source", "unknown")),
        "projection_vector_mode": str(summary.get("projection_vector_mode", "displacement")),
        "train_mse_star": maybe_float(summary.get("train_mse_star")),
        "test_mse_star": maybe_float(summary.get("test_mse_star")),
        "relative_ref": relative_ref,
    }
    mse_zero = (mse_at_zero_for_run(summary, payload)
                if relative_q and relative_ref == "theta_ref"
                and record["projection_vector_mode"] == "absolute" else None)

    curves: dict[str, np.ndarray] = {}
    k_vals: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        raw = to_numpy_1d(payload.get(f"{split}_mse_by_k_behavioral"))
        k_vals[split] = get_k_values(payload, len(raw), max_k)
        curves[split] = raw[: len(k_vals[split])]
    max_ks = [float(np.max(k)) for k in k_vals.values() if len(k)]
    record["max_k_available"] = max(max_ks) if max_ks else math.nan

    for split in ("test", "train"):
        curve, ks = curves[split], k_vals[split]
        best = np.minimum.accumulate(curve) if len(curve) else np.array([], dtype=float)
        if len(best) == 0:
            warnings.append(f"missing {split}_mse_by_k_behavioral for run {run_dir}")

        # Absolute rank r_tau: first k whose running-best loss is <= tau.
        for tau in test_thresholds:
            tok = sanitize_float_token(tau)
            k, reached = _first_k(best <= float(tau), ks) if len(best) else (math.nan, False)
            record[f"k_{split}_tau_{tok}"] = k
            record[f"reached_{split}_tau_{tok}"] = reached

        # Relative rank r_q: first k recovering a fraction q of L_ref - L*.
        l_ref = (_mse_theta0(summary, payload, split) if mse_zero is None
                 else maybe_float(mse_zero.get(f"{split}_mse_zero")))
        record[f"{split}_mse_ref"] = l_ref
        l_star = maybe_float(record.get(f"{split}_mse_star"))
        for q in relative_q:
            tok = sanitize_float_token(q)
            k, reached = math.nan, False
            if len(best) and l_ref is not None and l_star is not None:
                denom = float(l_ref - l_star)
                if denom > 0:
                    k, reached = _first_k((float(l_ref) - best) / denom >= float(q), ks)
                else:
                    warnings.append(f"non-positive L_ref - L* for {split} in run {run_dir}")
            record[f"r_{split}_q_{tok}"] = k
            record[f"reached_r_{split}_q_{tok}"] = reached

    eig_pos = get_behavioral_eigenvalues(payload, run_dir / "eigenvalues_behavioral.txt")
    if len(eig_pos) == 0:
        warnings.append(f"missing behavioral eigenvalues for run {run_dir}")
    eig_sum = float(np.sum(eig_pos)) if len(eig_pos) else 0.0
    for frac in (90, 95, 99):
        if len(eig_pos) == 0 or eig_sum <= 0.0:
            record[f"spectral_rank_{frac}"] = math.nan
        else:
            cum = np.cumsum(eig_pos) / eig_sum
            record[f"spectral_rank_{frac}"] = float(int(np.searchsorted(cum, frac / 100, side="left")) + 1)
    record[STABLE_EIG_COL] = (float(np.sum(eig_pos >= 1e-6 * float(eig_pos[0])))
                              if len(eig_pos) else math.nan)

    record["_curves"] = curves
    record["_eigs_pos"] = eig_pos
    return record, warnings


def gather_records(
    root_dir: Path,
    test_thresholds: list[float],
    relative_q: list[float],
    max_k: int | None,
    exclude_widths: set[int],
    relative_ref: str = RELATIVE_REF_DEFAULT,
) -> list[dict[str, Any]]:
    """Load every ``<cfg>/runs/<run>/summary.json`` under root_dir (at any depth), skipping
    excluded widths. Records are sorted by (d, seed, probe set, projection mode)."""
    records: list[dict[str, Any]] = []
    warned: set[str] = set()
    # The shallow pattern also catches <cfg> directories that are symlinks,
    # which the recursive pattern does not descend into.
    paths = set(root_dir.glob("*/runs/*/summary.json")) | set(root_dir.glob("**/runs/*/summary.json"))
    for summary_path in sorted(paths, key=str):
        record, warnings = load_run_record(summary_path.parent, test_thresholds, relative_q, max_k,
                                            relative_ref)
        for w in warnings:
            if w not in warned:
                print(f"[warn] {w}")
                warned.add(w)
        if record and int(record["d"]) not in exclude_widths:
            records.append(record)
    records.sort(key=lambda r: (r["d"], r["seed"], r["probe_source"], r["projection_vector_mode"]))
    return records
