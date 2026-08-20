#!/usr/bin/env python3
"""
run_experiment.py
=================
End-to-end NGSIM experiment for the phase-transition calibration study.

Protocol (this run is the *pooled / transferability* experiment)
----------------------------------------------------------------
A single global parameter vector is fitted over **all** training pairs at once
(pooled calibration -- not one theta per pair), separately for each objective,
then that one theta is graded on the held-out test pairs by free simulation.
This tests generalisation/transferability and, via the two sample baselines, the
differentiation-noise argument.  (Parameter-stability / per-pair Delta-theta is a
separate experiment and is intentionally not part of this run.)

Objectives compared (IDM by default):
    - segment-based  segment objective; feature set chosen with --features, any
                     subset of {v_end, s_end, dist, phase_speed,
                     phase_speed_ols, duration}.  Default {s_end, dist} is
                     position-level; phase_speed / phase_speed_ols add a
                     segment-level mean speed taken as a secant / OLS slope of
                     position across the segment (an averaging operator, so
                     unlike v_end it does not re-import differentiation noise).
                     Swap the set freely for ablation.
    - sample-spacing conventional RMSE on the gap  (the *strong* baseline; the
                     honest bar for non-inferiority, since it shares the primary
                     evaluation metric)
    - sample-speed   conventional RMSE on speed    (the *naive* baseline the
                     noise argument predicts should transfer worse)

Units
-----
NGSIM pairs are feet.  Segment boundaries are detected on the **native ft/s**
velocity (the PELT+ CUSUM thresholds are tuned for ft/s; SI under-detects), while
the objective features and all reported quantities are SI.  Boundary indices are
unit-invariant, so detection runs in ft and only the per-segment features are
recomputed on the SI arrays.

Nothing is hard-wired: the model, DE budget, segment feature set, both
non-inferiority margins, and the margin gate are CLI flags.

Outputs (to --out, default ./experiment_out)
    theta.json              the three pooled theta* (SI) + fit diagnostics
    test_per_pair.csv       per-pair held-out metrics for every theta (long form)
    summary.json            aggregates + paired head-to-heads + non-inferiority
    fig_test_spacing.png    per-pair test spacing-RMSE distribution (3 theta)
    fig_paired_diff.png     paired (segment-based - sample) differences with margin lines
    fig_examples.png        example test-pair gap overlays (obs + 3 sims)
    
    
python run_experiment.py --root cf_ngsim_I80 --model idm --out experiment_out_idm    

python run_experiment.py --root cf_ngsim_I80 --model ovm --out experiment_out_ovm

python run_experiment.py --root cf_ngsim_I80 --model gipps --out experiment_out_gipps

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_models import get_model, CFModel
from cf_data import PairData, load_folder, FT_TO_M
from phase_segmentation import (segment_trajectory, SegmentationResult, Phase,
                                _phase_features, FEATURE_KEYS,
                                validate_feature_keys)
from calibrate import calibrate_pairs, CalibResult
from evaluate import evaluate_theta, paired_compare, free_simulate, EvalResult


# --------------------------------------------------------------------------- #
# Native-unit segmentation (detect in ft, features in SI)
# --------------------------------------------------------------------------- #
def _resegment_si(seg_native: SegmentationResult, t: np.ndarray,
                  x_si: np.ndarray, v_si: np.ndarray,
                  s_si: np.ndarray) -> SegmentationResult:
    """Carry ft-detected boundary indices onto SI arrays, recomputing features.

    Boundary *indices* are unit-invariant; only the per-segment features depend on
    units, so we recompute the whole registered set on the SI arrays.
    """
    phases_si = [
        Phase(k=ph.k, i_start=ph.i_start, i_end=ph.i_end,
              t_start=float(t[ph.i_start]), t_end=float(t[ph.i_end]),
              kind=ph.kind,
              features=_phase_features(t, x_si, v_si, s_si,
                                       ph.i_start, ph.i_end))
        for ph in seg_native.phases
    ]
    return SegmentationResult(
        critical_points=seg_native.critical_points,
        decel_points=seg_native.decel_points,
        accel_points=seg_native.accel_points,
        phases=phases_si,
        diagnostics=seg_native.diagnostics)


def segment_pair(pair: PairData) -> SegmentationResult:
    """Segment a (feet-sourced) pair: PELT+ in native ft, features in SI."""
    if pair.units_source.startswith("feet"):
        inv = 1.0 / FT_TO_M
        seg_native = segment_trajectory(
            pair.t, pair.x_follower * inv, pair.v_follower * inv,
            pair.spacing * inv)
        return _resegment_si(seg_native, pair.t, pair.x_follower,
                             pair.v_follower, pair.spacing)
    # already native SI (e.g. TGSIM / synthetic)
    return segment_trajectory(pair.t, pair.x_follower,
                              pair.v_follower, pair.spacing)


# --------------------------------------------------------------------------- #
# Pooled calibration of the three objectives
# --------------------------------------------------------------------------- #
def calibrate_pooled(model: CFModel, train: Sequence[PairData],
                     segs: Sequence[SegmentationResult],
                     features: Sequence[str], *, de_kw: Dict) -> Dict[str, CalibResult]:
    results: Dict[str, CalibResult] = {}

    print(f"\n[pooled] segment-based (features={'+'.join(features)}) over "
          f"{len(train)} pairs ...")
    t0 = time.time()
    results["phase"] = calibrate_pairs(
        model, train, "phase", segmentations=list(segs),
        obj_kwargs=dict(feature_keys=tuple(features)), **de_kw)
    print(f"    J*={results['phase'].fun:.6g}   ({time.time()-t0:.1f}s)")

    print(f"[pooled] sample-RMSE / spacing (strong baseline) ...")
    t0 = time.time()
    results["sample_spacing"] = calibrate_pairs(
        model, train, "sample",
        obj_kwargs=dict(target="spacing", metric="rmse"), **de_kw)
    print(f"    RMSE*={results['sample_spacing'].fun:.6g}   "
          f"({time.time()-t0:.1f}s)")

    print(f"[pooled] sample-RMSE / speed (naive baseline) ...")
    t0 = time.time()
    results["sample_speed"] = calibrate_pairs(
        model, train, "sample",
        obj_kwargs=dict(target="speed", metric="rmse"), **de_kw)
    print(f"    RMSE*={results['sample_speed'].fun:.6g}   "
          f"({time.time()-t0:.1f}s)")
    return results


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
_COLORS = {"phase": "#c0392b", "sample_spacing": "#2471a3",
           "sample_speed": "#7f8c8d"}
_NICE = {"phase": "segment-based", "sample_spacing": "sample / spacing",
         "sample_speed": "sample / speed"}


def fig_test_spacing(evals: Dict[str, EvalResult], path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(evals.keys())
    data = [evals[k].rmse_vector("spacing_rmse") for k in labels]
    parts = ax.boxplot(data, showmeans=True, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([_NICE[k] for k in labels])
    for patch, k in zip(parts["boxes"], labels):
        patch.set_facecolor(_COLORS[k]); patch.set_alpha(0.35)
    # jittered points
    for i, (k, arr) in enumerate(zip(labels, data), start=1):
        xj = np.random.default_rng(1).normal(i, 0.05, len(arr))
        ax.scatter(xj, arr, s=14, color=_COLORS[k], alpha=0.7, zorder=3)
    ax.set_ylabel("held-out spacing RMSE  (m)")
    ax.set_title("Test-set spacing error by objective (per-pair)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_paired_diff(evals: Dict[str, EvalResult], path: str,
                    delta_abs: Optional[float], delta_rel: Optional[float]) -> None:
    a = evals["phase"].rmse_vector("spacing_rmse")
    b = evals["sample_spacing"].rmse_vector("spacing_rmse")
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
    ax.set_title("Paired difference (negative ⇒ segment-based better)", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_examples(model, thetas: Dict[str, Sequence[float]],
                 test: Sequence[PairData], path: str, k: int = 3) -> None:
    # pick pairs spanning the gap range
    order = np.argsort([np.mean(p.spacing) for p in test])
    picks = [test[order[i]] for i in
             np.linspace(0, len(test) - 1, k).astype(int)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3.2 * len(picks)),
                             squeeze=False)
    for ax, pair in zip(axes[:, 0], picks):
        ax.plot(pair.t, pair.spacing, color="#222", lw=1.6, label="observed gap")
        for key, th in thetas.items():
            sim = free_simulate(model, th, pair)
            ax.plot(pair.t, sim.s, color=_COLORS[key], lw=1.3,
                    ls="-" if key == "phase" else "--", label=_NICE[key])
        ax.set_ylabel("gap (m)")
        ax.set_title(f"{pair.name}  (mean gap {np.mean(pair.spacing):.1f} m)",
                     fontsize=9)
        ax.legend(fontsize=7, loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="folder containing manifest.csv (+ train/ test subdirs)")
    ap.add_argument("--model", default="idm", help="idm | gipps | ovm")
    ap.add_argument("--features", default="s_end,dist,phase_speed_ols",
                    help="segment feature set, comma-separated; any subset of "
                         + " ".join(FEATURE_KEYS)
                         + "  (e.g. s_end,dist,phase_speed)")
    ap.add_argument("--units", default="feet", help="feet | si | auto")
    # DE budget
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    # non-inferiority (pre-register before looking at test!)
    ap.add_argument("--delta-abs", type=float, default=0.5,
                    help="absolute non-inferiority margin, m spacing RMSE")
    ap.add_argument("--delta-rel", type=float, default=0.05,
                    help="relative non-inferiority margin, fraction of sample")
    ap.add_argument("--gate", choices=("and", "or"), default="and")
    # scale controls for quick dry runs
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--out", default="experiment_out_")
    args = ap.parse_args(argv)
    
    os.makedirs(args.out, exist_ok=True)
    model = get_model(args.model)
    features = [s.strip() for s in args.features.split(",") if s.strip()]
    try:                       # fail before loading/segmenting 200 pairs
        validate_feature_keys(features)
    except (KeyError, ValueError) as e:
        sys.exit(f"--features: {e}")
    de_kw = dict(maxiter=args.maxiter, popsize=args.popsize,
                 workers=args.workers, seed=args.seed)

    # ---- load ----
    train = load_folder(args.root, split="train", units=args.units,
                        limit=args.limit_train)
    test = load_folder(args.root, split="test", units=args.units,
                       limit=args.limit_test)
    if not train or not test:
        sys.exit(f"Loaded train={len(train)} test={len(test)} from {args.root}. "
                 f"Check that manifest.csv paths resolve under --root.")
    print(f"[load] train={len(train)} pairs, test={len(test)} pairs "
          f"(units={args.units})")

    # ---- segment train (native ft, SI features) ----
    print("[segment] segmenting training followers (native-unit detection) ...")
    segs = [segment_pair(p) for p in train]
    nph = np.array([s.n_phases for s in segs])
    print(f"    segments/pair: min={nph.min()} median={int(np.median(nph))} "
          f"max={nph.max()}   (pairs with <4 segments: {(nph < 4).sum()})")

    # ---- pooled calibration ----
    fits = calibrate_pooled(model, train, segs, features, de_kw=de_kw)
    thetas = {k: r.theta for k, r in fits.items()}

    print("\n" + "=" * 70)
    print(f"POOLED θ*  (model={model.name}, {len(train)} training pairs)")
    print("=" * 70)
    for k, r in fits.items():
        vals = "  ".join(f"{n}={v:.4f}" for n, v in r.param_dict.items())
        print(f"  {_NICE[k]:16s}: {vals}")
    
    # ---- evaluate on held-out test ----
    print("\n[evaluate] free-simulating test pairs ...")
    evals = {k: evaluate_theta(model, th, test, label=k)
             for k, th in thetas.items()}
    print(f"\n{'objective':18s}{'spc-RMSE(mean)':>15s}{'(median)':>10s}"
          f"{'RMSPE':>9s}{'collision-free':>16s}")
    for k, ev in evals.items():
        a = ev.aggregate
        print(f"  {_NICE[k]:16s}{a['spacing_rmse_mean']:15.4f}"
              f"{a['spacing_rmse_median']:10.4f}"
              f"{a['spacing_rmspe_mean']:9.3f}"
              f"{a['collision_free_frac']:16.2f}")
    
    # ---- paired head-to-heads ----
    print("\n[paired] segment-based vs sample/spacing (primary), and the noise contrast")
    primary = paired_compare(
        evals["phase"], evals["sample_spacing"], metric="spacing_rmse",
        delta_abs=args.delta_abs, delta_rel=args.delta_rel, gate=args.gate,
        seed=args.seed)
    noise_contrast = paired_compare(
        evals["sample_spacing"], evals["sample_speed"], metric="spacing_rmse",
        seed=args.seed)
    
    ni = primary.get("noninferiority", {})
    print(f"    segment-based−sample  mean Δ={primary['diff_mean']:+.4f} m  "
          f"median Δ={primary['diff_median']:+.4f}  "
          f"CI95=[{primary['diff_ci95'][0]:+.3f},{primary['diff_ci95'][1]:+.3f}]")
    print(f"    segment-based wins on {primary['A_better_frac']*100:.0f}% of pairs; "
          f"Wilcoxon p={primary['wilcoxon_p']:.3g}")
    if ni:
        print(f"    non-inferior (gate={ni['gate']}): {ni['noninferior']}  "
              f"{ {kk: vv['noninferior'] for kk, vv in ni['verdicts'].items()} }")
    print(f"    [contrast] spacing−speed baseline mean Δ="
          f"{noise_contrast['diff_mean']:+.4f} m "
          f"(speed worse ⇒ positive), Wilcoxon p={noise_contrast['wilcoxon_p']:.3g}")

    # ---- figures ----
    fig_test_spacing(evals, os.path.join(args.out, "fig_test_spacing.png"))
    fig_paired_diff(evals, os.path.join(args.out, "fig_paired_diff.png"),
                    args.delta_abs, args.delta_rel)
    fig_examples(model, thetas, test, os.path.join(args.out, "fig_examples.png"))
    
    # ---- persist ----
    with open(os.path.join(args.out, "theta.json"), "w") as f:
        json.dump({k: {"theta": r.theta, "param_dict": r.param_dict,
                       "fun": r.fun, "success": r.success,
                       "n_eval": r.n_eval, "elapsed_s": r.elapsed_s}
                   for k, r in fits.items()}, f, indent=2)

    # long-form per-pair CSV
    import csv
    csv_path = os.path.join(args.out, "test_per_pair.csv")
    metric_keys = ["mean_gap", "spacing_rmse", "spacing_mae", "spacing_rmspe",
                   "pos_rmse", "speed_rmse", "n_barrier", "collided"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["objective", "pair", "n"] + metric_keys)
        for k, ev in evals.items():
            for r in ev.per_pair:
                w.writerow([k, r["pair"], r["n"]] + [r[m] for m in metric_keys])

    summary = {
        "config": {"model": model.name, "features": features,
                   "maxiter": args.maxiter, "popsize": args.popsize,
                   "seed": args.seed, "units": args.units,
                   "delta_abs": args.delta_abs, "delta_rel": args.delta_rel,
                   "gate": args.gate,
                   "n_train": len(train), "n_test": len(test),
                   "protocol": "pooled calibration -> held-out free-sim; "
                               "primary metric spacing RMSE (position-level); "
                               "segment-based faces non-inferiority bar on it"},
        "theta": {k: r.param_dict for k, r in fits.items()},
        "test_aggregate": {k: ev.aggregate for k, ev in evals.items()},
        "paired_primary_phase_vs_sample_spacing": primary,
        "paired_contrast_spacing_vs_speed": noise_contrast,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[out] wrote theta.json, test_per_pair.csv, summary.json, "
          f"3 figures → {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
