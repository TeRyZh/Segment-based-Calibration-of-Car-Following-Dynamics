#!/usr/bin/env python3
"""
regenerate_figures.py
=====================
Rebuild the three ``run_experiment.py`` figures from an existing output folder
**without** re-running Differential Evolution.  Use this to re-do figure polish
(labels, colours, DPI) cheaply after the expensive pooled calibration is done.

Inputs (all read from ``--out``, the folder ``run_experiment.py`` wrote to)
    test_per_pair.csv   ->  fig_test_spacing.png  and  fig_paired_diff.png
    summary.json        ->  non-inferiority margins + model/units (as defaults)
    theta.json          ->  pooled theta* used for the example overlays

fig_examples.png additionally re-simulates the held-out test pairs.  That is a
free (open-loop) simulation only -- no calibration -- but it needs the pair
trajectories, so pass ``--root`` pointing at the same manifest folder used by
run_experiment.py.  If ``--root`` is omitted, only the two distribution figures
are regenerated (a note is printed).

The two distribution figures need nothing from the car-following stack, so they
regenerate even if ``cf_models`` / ``cf_data`` / ``evaluate`` are not importable
here (those are imported lazily, only for the example overlays).

Labels/colours match run_experiment.py (segment-based terminology).  The output
keys in the files are the legacy keys ("phase", "sample_spacing",
"sample_speed"); they are mapped to display labels via _NICE.

Examples
--------
    # all three figures (needs the manifest folder for the overlays)
    python regenerate_figures.py --out experiment_out_ --root /path/to/pairs

    # only the two distribution figures, from the CSV alone, at 600 dpi
    python regenerate_figures.py --out experiment_out_ --dpi 600
    
    python regenerate_figures.py --out experiment_out_idm --root cf_ngsim_I80
    
    python regenerate_figures.py --out experiment_out_ovm --root cf_ngsim_I80
    
    python regenerate_figures.py --out experiment_out_gipps --root cf_ngsim_I80
    
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Display constants -- kept identical to run_experiment.py
# --------------------------------------------------------------------------- #
ORDER = ["phase", "sample_spacing", "sample_speed"]      # legacy output keys
_COLORS = {"phase": "#c0392b", "sample_spacing": "#2471a3",
           "sample_speed": "#7f8c8d"}
_NICE = {"phase": "segment-based", "sample_spacing": "sample / spacing",
         "sample_speed": "sample / speed"}


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_per_pair(out_dir: str) -> Tuple[
        "OrderedDict[str, np.ndarray]", "OrderedDict[str, Dict[str, float]]",
        "OrderedDict[str, List[str]]"]:
    """Parse test_per_pair.csv.

    Returns three parallel views, each an OrderedDict keyed by objective in the
    canonical ORDER (only objectives actually present are included):
        rmse_by_obj[obj]  -> np.ndarray of spacing_rmse in *file order*
        rmse_map[obj]     -> {pair_name: spacing_rmse}  (for name alignment)
        names_by_obj[obj] -> [pair_name, ...] in file order
    """
    path = os.path.join(out_dir, "test_per_pair.csv")
    if not os.path.isfile(path):
        sys.exit(f"[error] not found: {path}")

    rows_by_obj: "OrderedDict[str, List[Tuple[str, float]]]" = OrderedDict()
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        needed = {"objective", "pair", "spacing_rmse"}
        missing_cols = needed - set(rdr.fieldnames or [])
        if missing_cols:
            sys.exit(f"[error] {path} is missing columns: {sorted(missing_cols)}")
        for r in rdr:
            obj = r["objective"]
            try:
                val = float(r["spacing_rmse"])
            except (TypeError, ValueError):
                continue
            rows_by_obj.setdefault(obj, []).append((r["pair"], val))

    if not rows_by_obj:
        sys.exit(f"[error] no rows parsed from {path}")

    # order objectives by ORDER first, then any extras in first-seen order
    ordered_keys = [k for k in ORDER if k in rows_by_obj] + \
                   [k for k in rows_by_obj if k not in ORDER]

    rmse_by_obj: "OrderedDict[str, np.ndarray]" = OrderedDict()
    rmse_map: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    names_by_obj: "OrderedDict[str, List[str]]" = OrderedDict()
    for k in ordered_keys:
        pairs = rows_by_obj[k]
        names_by_obj[k] = [p for p, _ in pairs]
        rmse_by_obj[k] = np.array([v for _, v in pairs], dtype=float)
        rmse_map[k] = {p: v for p, v in pairs}      # last wins on dup names
    return rmse_by_obj, rmse_map, names_by_obj


def read_json(out_dir: str, fname: str) -> Optional[dict]:
    path = os.path.join(out_dir, fname)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Figures (array-based; visuals identical to run_experiment.py)
# --------------------------------------------------------------------------- #
def fig_test_spacing(rmse_by_obj: "OrderedDict[str, np.ndarray]",
                     path: str, dpi: int) -> None:
    labels = list(rmse_by_obj.keys())
    data = [rmse_by_obj[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.boxplot(data, showmeans=True, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([_NICE.get(k, k) for k in labels])
    for patch, k in zip(parts["boxes"], labels):
        patch.set_facecolor(_COLORS.get(k, "#95a5a6")); patch.set_alpha(0.35)
    for i, (k, arr) in enumerate(zip(labels, data), start=1):
        xj = np.random.default_rng(1).normal(i, 0.05, len(arr))
        ax.scatter(xj, arr, s=14, color=_COLORS.get(k, "#95a5a6"),
                   alpha=0.7, zorder=3)
    ax.set_ylabel("held-out spacing RMSE  (m)")
    ax.set_title("Test-set spacing error by objective (per-pair)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_paired_diff(a: np.ndarray, b: np.ndarray, path: str,
                    delta_abs: Optional[float], delta_rel: Optional[float],
                    dpi: int) -> None:
    d = a - b
    order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#c0392b" if x > 0 else "#27ae60" for x in d[order]]
    ax.bar(range(len(d)), d[order], color=colors, alpha=0.8)
    ax.axhline(0, color="k", lw=1)
    if delta_abs is not None:
        ax.axhline(delta_abs, color="#8e44ad", ls="--", lw=1.2,
                   label=f"+δ_abs = {delta_abs:g} m")
    if delta_rel is not None:
        m = delta_rel * float(np.mean(b))
        ax.axhline(m, color="#e67e22", ls=":", lw=1.2,
                   label=f"+δ_rel = {m:.3g} m ({delta_rel:g}·sample)")
    ax.set_xlabel("test pair (sorted by difference)")
    ax.set_ylabel("segment-based − sample  spacing RMSE  (m)")
    ax.set_title("Paired difference (negative ⇒ segment-based better)",
                 fontweight="bold")
    if delta_abs is not None or delta_rel is not None:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_examples(model, thetas: "OrderedDict[str, Sequence[float]]",
                 test: Sequence, path: str, dpi: int, k: int = 3) -> None:
    # lazy: only the overlays need the CF stack
    from evaluate import free_simulate
    order = np.argsort([np.mean(p.spacing) for p in test])
    picks = [test[order[i]] for i in
             np.linspace(0, len(test) - 1, k).astype(int)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.2 * len(picks)),
                             squeeze=False)
    for ax, pair in zip(axes[:, 0], picks):
        ax.plot(pair.t, pair.spacing, color="#222", lw=1.6, label="observed gap")
        for key, th in thetas.items():
            sim = free_simulate(model, th, pair)
            ax.plot(pair.t, sim.s, color=_COLORS.get(key, "#95a5a6"), lw=1.3,
                    ls="-" if key == "phase" else "--", label=_NICE.get(key, key))
        ax.set_ylabel("gap (m)")
        ax.set_title(f"{pair.name}  (mean gap {np.mean(pair.spacing):.1f} m)",
                     fontsize=9)
        ax.legend(fontsize=7, loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Example-overlay support: reload exactly the graded test pairs
# --------------------------------------------------------------------------- #
def load_test_subset(root: str, units: str, pairs_order: Sequence[str],
                     limit: Optional[int]) -> List:
    """Reload test pairs and restrict to the graded set, in CSV order.

    ``pairs_order`` are the pair names from the CSV (the exact pairs the run
    graded, already reflecting any --limit-test), so name-restriction rebuilds
    the same list and the same example selection regardless of --limit-test.
    """
    from cf_data import load_folder
    reloaded = load_folder(root, split="test", units=units, limit=limit)
    by_name = {p.name: p for p in reloaded}
    subset = [by_name[n] for n in pairs_order if n in by_name]
    missing = [n for n in pairs_order if n not in by_name]
    if missing:
        print(f"[warn] {len(missing)} graded test pair(s) not found under "
              f"--root (first few: {missing[:3]}); overlays use the rest.")
    return subset


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True,
                    help="run_experiment.py output folder (theta.json, "
                         "test_per_pair.csv, summary.json)")
    ap.add_argument("--root", default=None,
                    help="manifest folder (train/ test/) -- required only for "
                         "fig_examples overlays; omit to skip them")
    ap.add_argument("--fig-out", default=None,
                    help="where to write the PNGs (default: --out)")
    ap.add_argument("--dpi", type=int, default=300, help="figure DPI")
    ap.add_argument("--k", type=int, default=3,
                    help="number of example pairs in fig_examples")
    # defaults for these are read from summary.json; CLI overrides win
    ap.add_argument("--delta-abs", type=float, default=None,
                    help="override absolute NI margin (m); default from summary")
    ap.add_argument("--delta-rel", type=float, default=None,
                    help="override relative NI margin; default from summary")
    ap.add_argument("--model", default=None,
                    help="override model for overlays; default from summary")
    ap.add_argument("--units", default=None,
                    help="override units for pair loading; default from summary")
    ap.add_argument("--limit-test", type=int, default=None,
                    help="passthrough to load_folder; usually unnecessary "
                         "(graded pairs are matched by name)")
    args = ap.parse_args(argv)

    fig_out = args.fig_out or args.out
    os.makedirs(fig_out, exist_ok=True)

    # ---- read files ----
    rmse_by_obj, rmse_map, names_by_obj = read_per_pair(args.out)
    summary = read_json(args.out, "summary.json") or {}
    cfg = summary.get("config", {})

    delta_abs = args.delta_abs if args.delta_abs is not None else cfg.get("delta_abs")
    delta_rel = args.delta_rel if args.delta_rel is not None else cfg.get("delta_rel")
    model_name = args.model or cfg.get("model")
    units = args.units or cfg.get("units", "auto")

    present = ", ".join(f"{_NICE.get(k, k)} (n={len(v)})"
                        for k, v in rmse_by_obj.items())
    print(f"[read] objectives: {present}")

    # ---- fig 1: distribution of held-out spacing RMSE ----
    p1 = os.path.join(fig_out, "fig_test_spacing.png")
    fig_test_spacing(rmse_by_obj, p1, args.dpi)
    print(f"[fig] {p1}")

    # ---- fig 2: paired difference (needs phase + sample_spacing) ----
    p2 = os.path.join(fig_out, "fig_paired_diff.png")
    if "phase" in rmse_map and "sample_spacing" in rmse_map:
        # align element-wise by pair name, in the phase objective's file order
        a_map, b_map = rmse_map["phase"], rmse_map["sample_spacing"]
        common = [n for n in names_by_obj["phase"] if n in b_map]
        dropped = len(names_by_obj["phase"]) - len(common)
        if dropped:
            print(f"[warn] {dropped} pair(s) present for segment-based but not "
                  f"sample/spacing; excluded from the paired figure.")
        a = np.array([a_map[n] for n in common], dtype=float)
        b = np.array([b_map[n] for n in common], dtype=float)
        fig_paired_diff(a, b, p2, delta_abs, delta_rel, args.dpi)
        print(f"[fig] {p2}"
              + ("" if (delta_abs is not None or delta_rel is not None)
                 else "   (no margins in summary.json; pass --delta-abs/-rel to draw them)"))
    else:
        print(f"[skip] {p2}: need both 'phase' and 'sample_spacing' in the CSV.")

    # ---- fig 3: example overlays (needs --root + theta.json) ----
    p3 = os.path.join(fig_out, "fig_examples.png")
    if args.root is None:
        print(f"[skip] {p3}: pass --root to re-simulate the example overlays.")
        print(f"\n[out] wrote 2 figures → {fig_out}/")
        return 0

    theta_json = read_json(args.out, "theta.json")
    if not theta_json:
        print(f"[skip] {p3}: theta.json not found in {args.out}.")
        print(f"\n[out] wrote 2 figures → {fig_out}/")
        return 0
    if not model_name:
        print(f"[skip] {p3}: model unknown (no summary.json); pass --model.")
        print(f"\n[out] wrote 2 figures → {fig_out}/")
        return 0

    # thetas in ORDER, only those present in both theta.json and the CSV
    thetas: "OrderedDict[str, Sequence[float]]" = OrderedDict()
    for k in list(rmse_by_obj.keys()):
        if k in theta_json and "theta" in theta_json[k]:
            thetas[k] = theta_json[k]["theta"]
    if not thetas:
        print(f"[skip] {p3}: no usable theta vectors in theta.json.")
        print(f"\n[out] wrote 2 figures → {fig_out}/")
        return 0

    try:
        from cf_models import get_model
        model = get_model(model_name)
        canon_order = names_by_obj.get("phase") or next(iter(names_by_obj.values()))
        test = load_test_subset(args.root, units, canon_order, args.limit_test)
        if not test:
            print(f"[skip] {p3}: no graded test pairs resolved under --root.")
        else:
            fig_examples(model, thetas, test, p3, args.dpi, k=args.k)
            print(f"[fig] {p3}  (model={model_name}, {len(test)} test pairs)")
    except ImportError as e:
        print(f"[skip] {p3}: CF stack not importable here ({e}). "
              f"Run where cf_models/cf_data/evaluate are on the path.")

    n_written = 2 + (1 if os.path.isfile(p3) else 0)
    print(f"\n[out] wrote {n_written} figures → {fig_out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
