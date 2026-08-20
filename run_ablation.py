#!/usr/bin/env python3
"""
run_ablation.py
===============
Ablation of the phase-anchored calibration objective (Mechanism A) on NGSIM.

Design (one-factor-at-a-time from a single reference configuration)
-------------------------------------------------------------------
Reference =  phase | PELT+ boundaries | features={s_end, dist, phase_speed_ols}
             | z-score W | IDM.   (the manuscript's proposed feature vector)
Shared bar =  sample-spacing (conventional RMSE on the gap), the non-inferiority
             anchor every phase cell is compared against.

Each ablation cell knocks out or swaps EXACTLY ONE component of the reference,
so any change in held-out error is attributable to that component:

  Feature axis  (anchor = PELT+, everything else = reference)
    {s_end}                                    minimal position-level
    {s_end, dist}                              position-level pair
    {s_end, phase_speed_ols}                   position + averaged phase speed
    {s_end, dist, phase_speed_ols}             <-- reference (proposed)
    {v_end}                                    velocity-only  (Probe 1: does
                                               reading v at the boundary re-import
                                               differentiation noise?)
    {v_end, s_end}                             mixed
    {v_end, s_end, dist, phase_speed_ols}      "all" incl. velocity
    <reference> + duration                     duration negative control
                                               (Probe 3: under fixed observed
                                               boundaries the duration residual is
                                               identically 0, so this MUST equal
                                               the reference exactly)

  Anchor-placement axis  (features = reference, everything else = reference)
    PELT+ critical points                      <-- reference
    uniform-K   (matched phase count)          Probe 2: behavioural placement vs.
    random-K    (matched phase count, seeded)  mere sparsity.

Protocol matches run_experiment.py: a single global theta is fitted per cell by
POOLED calibration over the training pairs, then graded on held-out pairs by free
(open-loop) simulation.  NGSIM pairs are feet; PELT+ boundaries are detected on
native ft/s (indices are unit-invariant) while features and all reported numbers
are SI -- reused verbatim via ``run_experiment.segment_pair``.

Response per cell
-----------------
  * held-out spacing RMSE  (mean / median / pooled)  -- primary
  * railed-parameter count -- a stability proxy from one pooled fit
  * paired non-inferiority vs sample-spacing (Wilcoxon + bootstrap CI)
  * PER-FEATURE WEIGHT INSPECTION -- for every phase cell, the diagonal W is
    reported per feature: the observed cross-phase std (the z-score denominator),
    the resulting weight w_j = 1/std_j^2 (median across pairs), and the feature's
    EFFECTIVE CONTRIBUTION to J at theta* (pooled weighted SSE share). The weight
    is what the optimiser assigned; the contribution is what that feature actually
    drove at the optimum -- an inert feature (e.g. duration) shows ~0 contribution.

Differential evolution uses polish=False (the phase loss is piecewise-constant in
the boundary indices; the L-BFGS-B polish is actively harmful there).

Requires the feature-registry version of objectives.py / phase_segmentation.py
(the one exposing phase_speed_ols and duration).

Outputs (to --out, default ./ablation_out)
    ablation_table.csv       one row per cell (long form)
    feature_weights.csv      per-cell x per-feature weights + contribution
    ablation_summary.json    config + reference + baseline + every cell (+ weights)
    fig_ablation.png         held-out RMSE per cell + railed-count per cell
    fig_feature_weights.png  per-cell feature-contribution stacks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_models import get_model, CFModel
from cf_data import load_folder
from calibrate import calibrate_pairs, CalibResult
from evaluate import evaluate_theta, paired_compare, EvalResult
from objectives import PhaseAnchoredObjective
from run_experiment import segment_pair                      # PELT+ (ft) -> SI feats
from ablation_anchors import matched_control_segmentation


# --------------------------------------------------------------------------- #
# Feature-axis cells (edit here; the reference set comes from --ref-features).
# The duration negative control (reference + 'duration') is appended
# automatically so it always tracks whatever reference is chosen.
# --------------------------------------------------------------------------- #
FEATURE_CELLS: List[Tuple[str, ...]] = [
    ("s_end",),
    ("s_end", "dist"),
    ("s_end", "phase_speed_ols"),
    ("s_end", "dist", "phase_speed_ols"),               # proposed / reference
    ("v_end",),
    ("v_end", "s_end"),
    ("v_end", "s_end", "dist", "phase_speed_ols"),      # "all" incl. velocity
]

# stable colour per feature for the contribution figure
_FEAT_COLORS = {
    "s_end": "#2471a3", "dist": "#27ae60", "phase_speed_ols": "#e67e22",
    "v_end": "#c0392b", "phase_speed": "#8e44ad", "duration": "#7f8c8d",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _feat_label(keys: Sequence[str]) -> str:
    return "{" + ",".join(keys) + "}"


def railed_params(theta: Sequence[float], model: CFModel,
                  tol: float = 0.01) -> List[str]:
    """Names of theta components within ``tol`` * (bound range) of a bound."""
    hits: List[str] = []
    for nm, val, (lo, hi) in zip(model.param_names, theta, model.bounds()):
        rng = float(hi - lo)
        if rng <= 0:
            continue
        if (val - lo) <= tol * rng:
            hits.append(f"{nm}@lo")
        elif (hi - val) <= tol * rng:
            hits.append(f"{nm}@hi")
    return hits


def feature_weight_report(model: CFModel, pairs, segs,
                          feature_keys: Sequence[str],
                          theta: Sequence[float]) -> List[Dict]:
    """Per-feature diagonal weights and effective contribution to J at ``theta``.

    Each pair is z-scored by its OWN cross-phase spread, so weights are per-pair;
    we report the median (and IQR) across pairs. The contribution is POOLED: the
    weighted SSE  w_j * sum_k resid_{k,j}^2  is summed over pairs, then normalised
    to a fraction of the pooled J -- the share of the objective each feature
    accounts for at ``theta``. Uses the same PhaseAnchoredObjective (default
    zscore weighting) the optimiser saw, so the weights match exactly.
    """
    keys = tuple(feature_keys)
    F = len(keys)
    stds = np.empty((len(pairs), F))
    weights = np.empty((len(pairs), F))
    contrib = np.zeros(F)
    total = 0.0
    for i, (pair, seg) in enumerate(zip(pairs, segs)):
        obj = PhaseAnchoredObjective(model, pair, seg, feature_keys=keys)
        stds[i] = obj.phi_obs.std(axis=0)                 # raw z-score denominator
        weights[i] = obj.w                                # actual weight used
        resid = obj._phi_sim(obj.simulate(theta)) - obj.phi_obs   # (K, F)
        wsse_j = obj.w * np.sum(resid ** 2, axis=0)               # (F,)
        contrib += wsse_j
        total += float(np.sum(wsse_j))
    frac = contrib / total if total > 0 else np.full(F, np.nan)
    rows: List[Dict] = []
    for j, k in enumerate(keys):
        q25, q75 = np.percentile(weights[:, j], [25, 75])
        rows.append({
            "feature": k,
            "obs_std_median": float(np.median(stds[:, j])),
            "weight_median": float(np.median(weights[:, j])),
            "weight_q25": float(q25),
            "weight_q75": float(q75),
            "contrib_frac": float(frac[j]),
        })
    return rows


def _calibrate_cell(model: CFModel, train, kind: str,
                    *, segs=None, features: Optional[Sequence[str]] = None,
                    target: str = "spacing", de_kw: Dict) -> CalibResult:
    if kind == "phase":
        return calibrate_pairs(
            model, train, "phase", segmentations=list(segs),
            obj_kwargs=dict(feature_keys=tuple(features)),
            polish=False, **de_kw)
    return calibrate_pairs(
        model, train, "sample",
        obj_kwargs=dict(target=target, metric="rmse"),
        polish=False, **de_kw)


def _row_from_fit(group: str, label: str, kind: str, anchor: str,
                  features: Optional[Sequence[str]],
                  fit: CalibResult, ev: EvalResult, model: CFModel,
                  rail_tol: float, cmp: Optional[Dict]) -> Dict:
    agg = ev.aggregate
    rails = railed_params(fit.theta, model, tol=rail_tol)
    row: Dict = {
        "group": group, "label": label, "kind": kind, "anchor": anchor,
        "features": _feat_label(features) if features else "",
        "spacing_rmse_mean": agg["spacing_rmse_mean"],
        "spacing_rmse_median": agg["spacing_rmse_median"],
        "spacing_rmse_pooled": agg["spacing_rmse_pooled"],
        "collision_free_frac": agg["collision_free_frac"],
        "n_railed": len(rails), "railed": "|".join(rails),
        "obj_star": fit.fun, "n_eval": fit.n_eval,
        "elapsed_s": round(fit.elapsed_s, 2),
        "theta": {k: round(v, 5) for k, v in fit.param_dict.items()},
    }
    if cmp is not None:
        ni = cmp.get("noninferiority", {})
        row.update({
            "diff_vs_baseline_mean": cmp["diff_mean"],
            "diff_vs_baseline_median": cmp["diff_median"],
            "ci_lo": cmp["diff_ci95"][0], "ci_hi": cmp["diff_ci95"][1],
            "wilcoxon_p": cmp["wilcoxon_p"], "phase_win_frac": cmp["A_better_frac"],
            "noninferior": bool(ni.get("noninferior")) if ni else None,
        })
    else:
        row.update({"diff_vs_baseline_mean": None, "diff_vs_baseline_median": None,
                    "ci_lo": None, "ci_hi": None, "wilcoxon_p": None,
                    "phase_win_frac": None, "noninferior": None})
    return row


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
_GROUP_ORDER = {"baseline": 0, "reference": 1, "feature": 2, "anchor": 3}


def _bar_color(row: Dict) -> str:
    g = row["group"]
    if g == "baseline":
        return "#7f8c8d"
    if g == "reference":
        return "#8e44ad"
    ni = row.get("noninferior")
    return {True: "#2471a3", False: "#c0392b"}.get(ni, "#95a5a6")


def make_figure(rows: List[Dict], baseline_rmse: float, ref_rmse: float,
                path: str) -> None:
    order = sorted(range(len(rows)),
                   key=lambda i: (_GROUP_ORDER.get(rows[i]["group"], 9), i))
    rows = [rows[i] for i in order]
    y = np.arange(len(rows))[::-1]
    labels = [r["label"] for r in rows]
    means = [r["spacing_rmse_mean"] for r in rows]
    medians = [r["spacing_rmse_median"] for r in rows]
    rails = [r["n_railed"] for r in rows]
    colors = [_bar_color(r) for r in rows]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 0.55 * len(rows) + 2.2), sharey=True,
        gridspec_kw=dict(width_ratios=[2.4, 1.0]))
    axL.barh(y, means, color=colors, alpha=0.85, height=0.62)
    axL.plot(medians, y, "k|", ms=10, mew=1.5, label="median")
    axL.axvline(baseline_rmse, color="#7f8c8d", ls="--", lw=1.4,
                label="sample-spacing baseline")
    axL.axvline(ref_rmse, color="#8e44ad", ls=":", lw=1.6, label="reference")
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
    axL.set_xlabel("held-out spacing RMSE  (m)")
    axL.set_title("Predictive accuracy by ablation cell", fontweight="bold",
                  fontsize=10)
    axL.legend(fontsize=7, loc="lower right"); axL.grid(axis="x", ls=":", alpha=0.4)

    axR.barh(y, rails, color=colors, alpha=0.85, height=0.62)
    axR.set_xlabel("# railed parameters")
    axR.set_title("Parameter railing", fontweight="bold", fontsize=10)
    axR.grid(axis="x", ls=":", alpha=0.4)
    axR.set_xticks(range(0, max(rails + [1]) + 1))

    for gi in range(1, len(rows)):
        if rows[gi]["group"] != rows[gi - 1]["group"]:
            b = (y[gi] + y[gi - 1]) / 2.0
            for ax in (axL, axR):
                ax.axhline(b, color="0.85", lw=0.8)

    fig.suptitle("Phase-anchored calibration -- one-factor-at-a-time ablation",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_weight_figure(weight_by_cell: "Dict[str, List[Dict]]",
                       ordered_labels: List[str], path: str) -> None:
    """Stacked horizontal bars: per-cell feature contribution share at theta*."""
    labels = [l for l in ordered_labels if l in weight_by_cell]
    y = np.arange(len(labels))[::-1]
    feats_seen = [f for f in _FEAT_COLORS
                  if any(w["feature"] == f for lbl in labels
                         for w in weight_by_cell[lbl])]

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(labels) + 2.0))
    for yi, lbl in zip(y, labels):
        wr = {w["feature"]: w["contrib_frac"] for w in weight_by_cell[lbl]}
        left = 0.0
        for f in feats_seen:
            frac = wr.get(f, 0.0)
            if frac <= 0:
                continue
            ax.barh(yi, frac, left=left, color=_FEAT_COLORS[f], alpha=0.9,
                    height=0.62)
            if frac > 0.06:
                ax.text(left + frac / 2, yi, f"{frac*100:.0f}%",
                        ha="center", va="center", fontsize=7, color="white")
            left += frac
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("share of J at theta*  (weighted SSE per feature)")
    ax.set_xlim(0, 1)
    ax.set_title("Effective feature contribution by cell (weight x residual)",
                 fontweight="bold", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_FEAT_COLORS[f])
               for f in feats_seen]
    ax.legend(handles, feats_seen, fontsize=8, ncol=min(len(feats_seen), 5),
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="folder with manifest.csv (+ train/ test subdirs)")
    ap.add_argument("--model", default="idm", help="idm | gipps | ovm")
    ap.add_argument("--ref-features", default="s_end,dist,phase_speed_ols",
                    help="reference feature set (drives the anchor axis + "
                         "duration control)")
    ap.add_argument("--units", default="feet", help="feet | si | auto")
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anchor-seed", type=int, default=0,
                    help="RNG seed for the random-K placement control")
    ap.add_argument("--min-seg", type=int, default=20,
                    help="min samples per control phase")
    ap.add_argument("--rail-tol", type=float, default=0.01,
                    help="railing tolerance as fraction of each bound's range")
    ap.add_argument("--delta-abs", type=float, default=0.5)
    ap.add_argument("--delta-rel", type=float, default=0.05)
    ap.add_argument("--gate", choices=("and", "or"), default="and")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out", default="ablation_out")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    model = get_model(args.model)
    ref_features = tuple(s.strip() for s in args.ref_features.split(",") if s.strip())
    de_kw = dict(maxiter=args.maxiter, popsize=args.popsize,
                 workers=args.workers, seed=args.seed)

    train = load_folder(args.root, split="train", units=args.units,
                        limit=args.limit_train)
    test = load_folder(args.root, split="test", units=args.units,
                       limit=args.limit_test)
    if not train or not test:
        sys.exit(f"Loaded train={len(train)} test={len(test)} from {args.root}.")
    print(f"[load] train={len(train)} test={len(test)} pairs (units={args.units})")

    print("[segment] PELT+ on training followers (native-unit detection) ...")
    segs_pelt = [segment_pair(p) for p in train]
    nph = np.array([s.n_phases for s in segs_pelt])
    print(f"    phases/pair: min={nph.min()} median={int(np.median(nph))} "
          f"max={nph.max()}")

    print("[segment] building matched-count uniform / random controls ...")
    segs_uniform = [matched_control_segmentation(
        s.n_phases, p.t, p.x_follower, p.v_follower, p.spacing,
        mode="uniform", min_segment_length=args.min_seg)
        for p, s in zip(train, segs_pelt)]
    segs_random = [matched_control_segmentation(
        s.n_phases, p.t, p.x_follower, p.v_follower, p.spacing,
        mode="random", min_segment_length=args.min_seg, seed=args.anchor_seed)
        for p, s in zip(train, segs_pelt)]

    rows: List[Dict] = []
    weight_by_cell: Dict[str, List[Dict]] = {}
    weight_long: List[Dict] = []
    ev_base: Optional[EvalResult] = None      # set below, before any phase cell

    def run_phase_cell(group: str, label: str, anchor: str, segs, features):
        t0 = time.time()
        fit = _calibrate_cell(model, train, "phase", segs=segs,
                              features=features, de_kw=de_kw)
        ev = evaluate_theta(model, fit.theta, test, label=label)
        cmp = paired_compare(ev, ev_base, metric="spacing_rmse",
                             delta_abs=args.delta_abs, delta_rel=args.delta_rel,
                             gate=args.gate, seed=args.seed)
        wrep = feature_weight_report(model, train, segs, features, fit.theta)
        row = _row_from_fit(group, label, "phase", anchor, features,
                            fit, ev, model, args.rail_tol, cmp)
        row["feature_weights"] = wrep
        rows.append(row)
        weight_by_cell[label] = wrep
        for w in wrep:
            weight_long.append({"group": group, "cell": label, **w})
        print(f"    {label[:44]:44s} mean={ev.aggregate['spacing_rmse_mean']:.4f} m"
              f"  railed={row['n_railed']}  ({time.time()-t0:.1f}s)")
        return fit, ev, row, wrep

    # ---- baseline: sample-spacing (shared non-inferiority anchor) ----
    print("\n[cell] baseline  sample-spacing (shared bar) ...")
    t0 = time.time()
    fit_base = _calibrate_cell(model, train, "sample", target="spacing", de_kw=de_kw)
    ev_base = evaluate_theta(model, fit_base.theta, test, label="sample_spacing")
    rows.append(_row_from_fit("baseline", "sample-spacing", "sample", "-",
                              None, fit_base, ev_base, model, args.rail_tol, None))
    baseline_rmse = ev_base.aggregate["spacing_rmse_mean"]
    print(f"    held-out mean={baseline_rmse:.4f} m  railed="
          f"{len(railed_params(fit_base.theta, model, args.rail_tol))}  "
          f"({time.time()-t0:.1f}s)")

    # ---- reference: phase | PELT+ | ref_features ----
    print(f"[cell] reference  phase | PELT+ | {_feat_label(ref_features)} ...")
    _, ev_ref, row_ref, wrep_ref = run_phase_cell(
        "reference", f"PELT+ | {_feat_label(ref_features)}", "PELT+",
        segs_pelt, ref_features)
    ref_rmse = row_ref["spacing_rmse_mean"]

    print("\n    reference per-feature weights (median across pairs) + "
          "contribution at theta*:")
    print(f"      {'feature':18s}{'obs_std':>10s}{'weight':>12s}{'contrib%':>10s}")
    for w in wrep_ref:
        print(f"      {w['feature']:18s}{w['obs_std_median']:10.4f}"
              f"{w['weight_median']:12.4g}{w['contrib_frac']*100:9.1f}%")

    # ---- feature axis (anchor = PELT+); duration control tracks reference ----
    print("\n[axis] feature set (anchor = PELT+) ...")
    feature_cells: List[Tuple[str, ...]] = [tuple(c) for c in FEATURE_CELLS]
    dur_cell = tuple(ref_features) + ("duration",)
    if dur_cell not in feature_cells:
        feature_cells.append(dur_cell)
    for feats in feature_cells:
        if tuple(feats) == tuple(ref_features):
            continue                              # reference already computed
        run_phase_cell("feature", _feat_label(feats), "PELT+", segs_pelt, feats)

    # ---- anchor-placement axis (features = ref_features) ----
    print("\n[axis] anchor placement (features = reference) ...")
    for anchor, segs in (("uniform-K", segs_uniform), ("random-K", segs_random)):
        run_phase_cell("anchor", anchor, anchor, segs, ref_features)

    # ---- console summary ----
    print("\n" + "=" * 80)
    print(f"ABLATION SUMMARY  (model={model.name}, {len(train)} train / "
          f"{len(test)} test)   baseline held-out RMSE = {baseline_rmse:.4f} m")
    print("=" * 80)
    print(f"{'group':10s}{'cell':32s}{'RMSE':>8s}{'med':>8s}{'rail':>6s}"
          f"{'dVsBase':>9s}{'NI':>4s}")
    for r in rows:
        ni = "" if r["noninferior"] is None else ("Y" if r["noninferior"] else "n")
        dv = "" if r["diff_vs_baseline_mean"] is None \
            else f"{r['diff_vs_baseline_mean']:+.3f}"
        print(f"{r['group']:10s}{r['label'][:31]:32s}"
              f"{r['spacing_rmse_mean']:8.4f}{r['spacing_rmse_median']:8.4f}"
              f"{r['n_railed']:6d}{dv:>9s}{ni:>4s}")

    # ---- persist: ablation table ----
    cols = ["group", "label", "kind", "anchor", "features",
            "spacing_rmse_mean", "spacing_rmse_median", "spacing_rmse_pooled",
            "collision_free_frac", "n_railed", "railed",
            "diff_vs_baseline_mean", "diff_vs_baseline_median",
            "ci_lo", "ci_hi", "wilcoxon_p", "phase_win_frac", "noninferior",
            "obj_star", "n_eval", "elapsed_s"]
    with open(os.path.join(args.out, "ablation_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) for c in cols])

    # ---- persist: feature weights (long form) ----
    wcols = ["group", "cell", "feature", "obs_std_median",
             "weight_median", "weight_q25", "weight_q75", "contrib_frac"]
    with open(os.path.join(args.out, "feature_weights.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(wcols)
        for r in weight_long:
            w.writerow([r.get(c) for c in wcols])

    summary = {
        "config": {
            "model": model.name, "ref_features": list(ref_features),
            "feature_cells": [list(c) for c in feature_cells],
            "anchor_cells": ["PELT+", "uniform-K", "random-K"],
            "units": args.units, "maxiter": args.maxiter, "popsize": args.popsize,
            "seed": args.seed, "anchor_seed": args.anchor_seed,
            "min_seg": args.min_seg, "rail_tol": args.rail_tol,
            "delta_abs": args.delta_abs, "delta_rel": args.delta_rel,
            "gate": args.gate, "polish": False,
            "n_train": len(train), "n_test": len(test),
            "protocol": "pooled calibration -> held-out free-sim; one-factor-"
                        "at-a-time from reference; primary metric spacing RMSE",
        },
        "baseline_holdout_rmse_mean": baseline_rmse,
        "reference_holdout_rmse_mean": ref_rmse,
        "rows": rows,
    }
    with open(os.path.join(args.out, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    make_figure(rows, baseline_rmse, ref_rmse,
                os.path.join(args.out, "fig_ablation.png"))
    ordered_labels = [r["label"] for r in sorted(
        rows, key=lambda r: (_GROUP_ORDER.get(r["group"], 9),))
        if r["kind"] == "phase"]
    make_weight_figure(weight_by_cell, ordered_labels,
                       os.path.join(args.out, "fig_feature_weights.png"))

    print(f"\n[out] wrote ablation_table.csv, feature_weights.csv, "
          f"ablation_summary.json, fig_ablation.png, fig_feature_weights.png "
          f"-> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
