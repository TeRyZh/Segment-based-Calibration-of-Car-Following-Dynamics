#!/usr/bin/env python3
"""
run_robustness.py
=================
Noise-robustness experiment for the phase-transition calibration study, a
sibling of run_experiment.py. It answers two questions (decision D1 = both):

  A. Predictive degradation  -- pooled calibration on RAW-train (theta_N) vs
     PRECISE-train (theta_C), BOTH free-simulated on PRECISE-test. Does noisy
     training degrade held-out spacing RMSE, and does the phase objective
     degrade less than the sample baselines? (Problem 2)

  B. Parameter displacement  -- per-pair calibration (Protocol B) of the SAME
     pair on its raw and precise versions -> Delta-theta = theta_N - theta_C,
     normalised by bound range. Small ||Delta-theta|| = noise did not get
     absorbed into the parameters. Reported per parameter with a railing mask so
     bound-pinned parameters (IDM b, OVM kappa/s_c) can't fake stability.
     (Problem 3)

  C. Segmentation stability  -- PELT+ critical points on raw vs precise per pair
     (count, matched fraction, boundary-time MAE). A prerequisite for A/B under
     noisy training; reported honestly whether it is a strength or a weakness.

Optional (D2): --sigma-sweep injects graded synthetic noise onto PRECISE (known
signal), re-calibrates, and evaluates on clean PRECISE-test -> a degradation
vs sigma curve that side-steps the "precise is only a reconstruction" caveat.

Input
-----
A pairing manifest from merge_robustness_pairs.py binding each pair's raw and
precise per-pair csv. Only rows with aligned == True are used. This script
re-splits the matched set with its own seed, so raw and precise share the split.
Noisy data is generated on the fly (seeded), never persisted.

Units follow run_experiment: NGSIM pairs are feet; phase boundaries are detected
on native ft/s (via segment_pair) and features/reported quantities are SI.

Outputs (to --out)
    robustness_summary.json
    degradation_per_pair.csv        (A, long form: objective x pair, C and N)
    delta_theta_per_pair.csv        (B, long form: objective x pair x parameter)
    seg_stability.csv               (C)
    fig_degradation.png             (A)
    fig_delta_theta.png             (B, per-parameter violins by objective)
    fig_example_pair.png            (A/B, one test pair: obs + sim theta_C/theta_N)
    fig_sigma_curve.png             (only with --sigma-sweep)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
import zlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Pipeline symbols are imported lazily (see _import_pipeline) so --pipeline-dir
# can be honoured and so this file stays importable for --self-test.
get_model = None
load_pair = None
calibrate_pairs = None
calibrate_pair = None
evaluate_theta = None
paired_compare = None
free_simulate = None
segment_pair = None

OBJECTIVES = ("phase", "sample_spacing", "sample_speed")
_COLORS = {"phase": "#c0392b", "sample_spacing": "#2471a3", "sample_speed": "#7f8c8d"}
_NICE = {"phase": "phase", "sample_spacing": "sample / spacing",
         "sample_speed": "sample / speed"}


def _import_pipeline(pipeline_dir: Optional[str]) -> None:
    """Bind the project modules into module globals (after path injection)."""
    global get_model, load_pair, calibrate_pairs, calibrate_pair
    global evaluate_theta, paired_compare, free_simulate, segment_pair
    if pipeline_dir:
        sys.path.insert(0, os.path.abspath(pipeline_dir))
    from cf_models import get_model as _gm
    from cf_data import load_pair as _lp
    from calibrate import calibrate_pairs as _cp, calibrate_pair as _c1
    from evaluate import (evaluate_theta as _et, paired_compare as _pc,
                          free_simulate as _fs)
    from run_experiment import segment_pair as _sp
    get_model, load_pair = _gm, _lp
    calibrate_pairs, calibrate_pair = _cp, _c1
    evaluate_theta, paired_compare, free_simulate = _et, _pc, _fs
    segment_pair = _sp


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested by --self-test; no pipeline deps)
# --------------------------------------------------------------------------- #
def split_names(names: Sequence[str], test_frac: float,
                seed: int) -> Tuple[List[str], List[str]]:
    """Deterministic seeded train/test split on the matched pair names."""
    names = list(names)
    idx = np.random.default_rng(seed).permutation(len(names))
    n_test = max(1, int(round(test_frac * len(names))))
    test = sorted(names[i] for i in idx[:n_test])
    train = sorted(names[i] for i in idx[n_test:])
    return train, test


def obj_spec(objective: str, features: Sequence[str]) -> Tuple[str, Dict]:
    """Map a robustness objective label to (calibrate objective, obj_kwargs)."""
    if objective == "phase":
        return "phase", dict(feature_keys=tuple(features))
    if objective == "sample_spacing":
        return "sample", dict(target="spacing", metric="rmse")
    if objective == "sample_speed":
        return "sample", dict(target="speed", metric="rmse")
    raise ValueError(objective)


def is_railed(value: float, lo: float, hi: float, rtol: float) -> bool:
    span = max(hi - lo, 1e-12)
    return abs(value - lo) <= rtol * span or abs(value - hi) <= rtol * span


def inject_noise(pair, sigma: float, seed: int, target: str = "position"):
    """Return a noised deepcopy of a pair (leader stays exogenous).

    position: x_f -> x_f + eps ; v_f re-derived by finite difference (this is the
              differentiation-noise mechanism); spacing s -> s - eps.
    velocity: v_f -> v_f + eps only.
    Noise is seeded per pair (stable across runs) so results are reproducible.
    """
    q = copy.deepcopy(pair)
    n = len(pair.t)
    rng = np.random.default_rng(
        [int(seed), zlib.adler32(str(pair.name).encode()) & 0xFFFFFFFF])
    eps = rng.normal(0.0, sigma, n)
    t = np.asarray(pair.t, float)
    if target == "position":
        x = np.asarray(pair.x_follower, float) + eps
        v = np.gradient(x, t)
        q.x_follower = x
        q.v_follower = v
        q.spacing = np.asarray(pair.spacing, float) - eps
        if hasattr(q, "a_follower"):
            q.a_follower = np.gradient(v, t)
    else:
        q.v_follower = np.asarray(pair.v_follower, float) + eps
    return q


def match_critical_points(cps_a: Sequence[int], cps_b: Sequence[int],
                          tol_frames: int) -> Tuple[int, float]:
    """Greedy nearest matching of two critical-point index lists.

    Returns (n_matched, mean_abs_offset_frames_over_matched).
    """
    a = sorted(cps_a)
    b = sorted(cps_b)
    used = [False] * len(b)
    matched, offsets = 0, []
    for ia in a:
        best_j, best_d = -1, tol_frames + 1
        for j, ib in enumerate(b):
            if used[j]:
                continue
            d = abs(ia - ib)
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= tol_frames:
            used[best_j] = True
            matched += 1
            offsets.append(best_d)
    mae = float(np.mean(offsets)) if offsets else float("nan")
    return matched, mae


# --------------------------------------------------------------------------- #
# Data loading via the pairing manifest
# --------------------------------------------------------------------------- #
class PairStore:
    """Lazily load + cache raw/precise PairData for matched pair names."""

    def __init__(self, manifest_df: pd.DataFrame, units: str):
        self.units = units
        self._raw_path = dict(zip(manifest_df["name"], manifest_df["raw_path"]))
        self._prc_path = dict(zip(manifest_df["name"], manifest_df["precise_path"]))
        self._raw: Dict[str, object] = {}
        self._prc: Dict[str, object] = {}

    def raw(self, name: str):
        if name not in self._raw:
            self._raw[name] = load_pair(self._raw_path[name], units=self.units)
        return self._raw[name]

    def precise(self, name: str):
        if name not in self._prc:
            self._prc[name] = load_pair(self._prc_path[name], units=self.units)
        return self._prc[name]


def _segs(pairs, objective):
    return [segment_pair(p) for p in pairs] if objective == "phase" else None


def _calib_set(model, pairs, objective, features, de_kw, segs=None):
    name, ok = obj_spec(objective, features)
    return calibrate_pairs(model, pairs, name, segmentations=segs,
                           obj_kwargs=ok, **de_kw)


def _calib_one(model, pair, objective, features, de_kw, seg=None):
    name, ok = obj_spec(objective, features)
    return calibrate_pair(model, pair, name, segmentation=seg,
                          obj_kwargs=ok, **de_kw)


# --------------------------------------------------------------------------- #
# Analysis A -- predictive degradation (pooled)
# --------------------------------------------------------------------------- #
def analysis_degradation(model, store: PairStore, train_names, test_names,
                         features, de_kw) -> Dict:
    precise_train = [store.precise(n) for n in train_names]
    raw_train = [store.raw(n) for n in train_names]
    precise_test = [store.precise(n) for n in test_names]

    per_obj = {}
    for obj in OBJECTIVES:
        segs_C = _segs(precise_train, obj)
        segs_N = _segs(raw_train, obj)
        print(f"  [A] {obj}: theta_C (precise-train) ...")
        crC = _calib_set(model, precise_train, obj, features, de_kw, segs_C)
        print(f"  [A] {obj}: theta_N (raw-train) ...")
        crN = _calib_set(model, raw_train, obj, features, de_kw, segs_N)
        evC = evaluate_theta(model, crC.theta, precise_test, label=f"{obj}_C")
        evN = evaluate_theta(model, crN.theta, precise_test, label=f"{obj}_N")
        per_obj[obj] = dict(crC=crC, crN=crN, evC=evC, evN=evN)

    # phase vs baselines on the NOISY-trained models (robustness head-to-head)
    contrasts = {}
    for base in ("sample_spacing", "sample_speed"):
        contrasts[f"phaseN_vs_{base}N"] = paired_compare(
            per_obj["phase"]["evN"], per_obj[base]["evN"],
            metric="spacing_rmse")
    return dict(per_obj=per_obj, contrasts=contrasts)


# --------------------------------------------------------------------------- #
# Analysis B -- per-pair Delta-theta (Protocol B)
# --------------------------------------------------------------------------- #
def analysis_delta_theta(model, store: PairStore, train_names, features,
                         pp_de_kw, rail_rtol: float,
                         limit: Optional[int]) -> pd.DataFrame:
    bounds = model.bounds()
    span = [max(hi - lo, 1e-12) for lo, hi in bounds]
    pnames = list(model.param_names)
    names = train_names[:limit] if limit else train_names

    records = []
    for j, name in enumerate(names, 1):
        rp, pp = store.raw(name), store.precise(name)
        for obj in OBJECTIVES:
            seg_r = segment_pair(rp) if obj == "phase" else None
            seg_p = segment_pair(pp) if obj == "phase" else None
            crN = _calib_one(model, rp, obj, features, pp_de_kw, seg_r)
            crC = _calib_one(model, pp, obj, features, pp_de_kw, seg_p)
            for i, p in enumerate(pnames):
                tN, tC = float(crN.theta[i]), float(crC.theta[i])
                lo, hi = bounds[i]
                railed = (is_railed(tN, lo, hi, rail_rtol)
                          or is_railed(tC, lo, hi, rail_rtol))
                records.append(dict(
                    objective=obj, pair=name, param=p,
                    theta_N=tN, theta_C=tC, dtheta=tN - tC,
                    dtheta_norm=abs(tN - tC) / span[i], railed=railed))
        if j % 10 == 0 or j == len(names):
            print(f"  [B] {j}/{len(names)} pairs")
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Analysis C -- segmentation stability
# --------------------------------------------------------------------------- #
def analysis_seg_stability(store: PairStore, train_names, dt: float,
                           match_tol_s: float,
                           limit: Optional[int]) -> pd.DataFrame:
    tol_frames = int(round(match_tol_s / dt))
    names = train_names[:limit] if limit else train_names
    rows = []
    for name in names:
        sr = segment_pair(store.raw(name))
        sp = segment_pair(store.precise(name))
        n_matched, mae_f = match_critical_points(
            sr.critical_points, sp.critical_points, tol_frames)
        denom = max(len(sp.critical_points), 1)
        rows.append(dict(
            pair=name, n_cp_raw=len(sr.critical_points),
            n_cp_precise=len(sp.critical_points), n_matched=n_matched,
            matched_frac=n_matched / denom,
            boundary_mae_s=(mae_f * dt if mae_f == mae_f else float("nan"))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Optional sigma-sweep
# --------------------------------------------------------------------------- #
def sigma_sweep(model, store: PairStore, train_names, test_names, features,
                de_kw, sigmas, noise_seed, target) -> pd.DataFrame:
    precise_train = [store.precise(n) for n in train_names]
    precise_test = [store.precise(n) for n in test_names]
    rows = []
    for sig in sigmas:
        noised = ([precise_train[i] if sig == 0 else
                   inject_noise(precise_train[i], sig, noise_seed, target)
                   for i in range(len(precise_train))])
        for obj in OBJECTIVES:
            segs = _segs(noised, obj)
            cr = _calib_set(model, noised, obj, features, de_kw, segs)
            ev = evaluate_theta(model, cr.theta, precise_test, label=f"{obj}_s{sig}")
            rows.append(dict(sigma=sig, objective=obj,
                             spacing_rmse=ev.aggregate["spacing_rmse_mean"]))
        print(f"  [sigma] sigma={sig} done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_degradation(deg: Dict, path: str) -> None:
    import matplotlib.pyplot as plt
    per = deg["per_obj"]
    labels = list(OBJECTIVES)
    c_vals = [per[k]["evC"].aggregate["spacing_rmse_mean"] for k in labels]
    n_vals = [per[k]["evN"].aggregate["spacing_rmse_mean"] for k in labels]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, c_vals, w, label="clean-trained (theta_C)",
           color=[_COLORS[k] for k in labels], alpha=0.55)
    ax.bar(x + w / 2, n_vals, w, label="noisy-trained (theta_N)",
           color=[_COLORS[k] for k in labels], alpha=0.95)
    for xi, cv, nv in zip(x, c_vals, n_vals):
        ax.annotate(f"+{nv - cv:.2f}", (xi, max(cv, nv)),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([_NICE[k] for k in labels])
    ax.set_ylabel("held-out spacing RMSE on Precise-test  (m)")
    ax.set_title("Predictive robustness: clean- vs noisy-trained",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_delta_theta(df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    params = list(dict.fromkeys(df["param"]))
    ncols = min(3, len(params)); nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                             squeeze=False)
    for idx, prm in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        data, positions, colors, railed_counts = [], [], [], []
        for j, obj in enumerate(OBJECTIVES, 1):
            sub = df[(df["param"] == prm) & (df["objective"] == obj)]
            free = sub[~sub["railed"]]["dtheta_norm"].to_numpy()
            railed_counts.append(int(sub["railed"].sum()))
            if free.size:
                data.append(free); positions.append(j); colors.append(_COLORS[obj])
        if data:
            vp = ax.violinplot(data, positions=positions, showmedians=True,
                               widths=0.8)
            for body, c in zip(vp["bodies"], colors):
                body.set_facecolor(c); body.set_alpha(0.45)
        ax.set_title(prm, fontweight="bold")
        ax.set_xticks(range(1, len(OBJECTIVES) + 1))
        ax.set_xticklabels([_NICE[o].replace("sample / ", "s/") for o in OBJECTIVES],
                           fontsize=8)
        ax.set_ylabel("|Δθ| / bound range")
        # note railed pairs excluded from each violin
        rc = ", ".join(f"{_NICE[o].replace('sample / ','s/')}:{n}"
                       for o, n in zip(OBJECTIVES, railed_counts) if n)
        if rc:
            ax.text(0.5, 0.97, f"railed excl.  {rc}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6, color="#555")
    for k in range(len(params), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Per-pair parameter displacement raw→precise (lower = robust)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_example_pair(model, deg: Dict, example_pair, path: str) -> None:
    import matplotlib.pyplot as plt
    per = deg["per_obj"]
    fig, axes = plt.subplots(len(OBJECTIVES), 1,
                             figsize=(10, 2.8 * len(OBJECTIVES)), squeeze=False)
    for ax, obj in zip(axes[:, 0], OBJECTIVES):
        ax.plot(example_pair.t, example_pair.spacing, color="#222", lw=1.6,
                label="observed gap (Precise)")
        simC = free_simulate(model, per[obj]["crC"].theta, example_pair)
        simN = free_simulate(model, per[obj]["crN"].theta, example_pair)
        ax.plot(example_pair.t, simC.s, color=_COLORS[obj], lw=1.4,
                label="θ_C (clean-trained)")
        ax.plot(example_pair.t, simN.s, color=_COLORS[obj], lw=1.4, ls="--",
                label="θ_N (noisy-trained)")
        ax.set_ylabel("gap (m)")
        ax.set_title(f"{_NICE[obj]}  —  {getattr(example_pair, 'name', 'pair')}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="best")
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Noisy-training shift on one Precise-test pair",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_sigma_curve(df: pd.DataFrame, path: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for obj in OBJECTIVES:
        sub = df[df["objective"] == obj].sort_values("sigma")
        ax.plot(sub["sigma"], sub["spacing_rmse"], marker="o",
                color=_COLORS[obj], label=_NICE[obj])
    ax.set_xlabel("injected position noise σ  (m)")
    ax.set_ylabel("held-out spacing RMSE on clean Precise-test  (m)")
    ax.set_title("Degradation vs synthetic noise (known signal)",
                 fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Aggregation for the summary
# --------------------------------------------------------------------------- #
def _delta_aggregate(df: pd.DataFrame) -> Dict:
    """Median |Δθ_norm| per objective on the identifiable (non-railed) subset."""
    out = {}
    for obj in OBJECTIVES:
        sub = df[(df["objective"] == obj) & (~df["railed"])]
        out[obj] = {
            "median_dtheta_norm_identifiable":
                (float(sub["dtheta_norm"].median()) if len(sub) else None),
            "n_identifiable": int(len(sub)),
            "n_railed": int(((df["objective"] == obj) & df["railed"]).sum()),
            "per_param_median": {
                p: (float(g[~g["railed"]]["dtheta_norm"].median())
                    if len(g[~g["railed"]]) else None)
                for p, g in df[df["objective"] == obj].groupby("param")}}
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args) -> int:
    _import_pipeline(args.pipeline_dir)
    os.makedirs(args.out, exist_ok=True)

    man = pd.read_csv(args.manifest)
    aligned = man[man["aligned"]] if "aligned" in man.columns else man
    dropped = len(man) - len(aligned)
    if aligned.empty:
        sys.exit("Manifest has no aligned pairs.")
    store = PairStore(aligned, args.units)
    all_names = sorted(aligned["name"].tolist())

    train_names, test_names = split_names(all_names, args.test_frac, args.seed)
    if args.limit_train:
        train_names = train_names[:args.limit_train]
    if args.limit_test:
        test_names = test_names[:args.limit_test]

    model = get_model(args.model)
    features = [s.strip() for s in args.features.split(",") if s.strip()]
    de_kw = dict(maxiter=args.maxiter, popsize=args.popsize,
                 workers=args.workers, seed=args.seed, polish=args.polish)
    pp_de_kw = dict(maxiter=args.pp_maxiter, popsize=args.pp_popsize,
                    workers=args.workers, seed=args.seed, polish=args.polish)

    print(f"[load] {len(all_names)} matched pairs ({dropped} misaligned dropped) "
          f"-> train={len(train_names)} test={len(test_names)}  model={model.name}")

    summary: Dict = {"config": {
        "model": model.name, "features": features, "units": args.units,
        "test_frac": args.test_frac, "seed": args.seed,
        "n_matched": len(all_names), "n_train": len(train_names),
        "n_test": len(test_names), "pooled_de": de_kw, "perpair_de": pp_de_kw,
        "rail_rtol": args.rail_rtol,
        "protocol": "train raw vs precise, evaluate on precise; per-pair "
                    "Delta-theta on training pairs; noisy generated on the fly"}}

    deg = None
    if not args.no_degradation:
        print("[A] predictive degradation (pooled) ...")
        t0 = time.time()
        deg = analysis_degradation(model, store, train_names, test_names,
                                   features, de_kw)
        summary["degradation"] = {
            obj: {"rmse_clean_trained":
                  d["evC"].aggregate["spacing_rmse_mean"],
                  "rmse_noisy_trained":
                  d["evN"].aggregate["spacing_rmse_mean"],
                  "degradation":
                  d["evN"].aggregate["spacing_rmse_mean"]
                  - d["evC"].aggregate["spacing_rmse_mean"],
                  "theta_C": d["crC"].param_dict,
                  "theta_N": d["crN"].param_dict}
            for obj, d in deg["per_obj"].items()}
        summary["degradation_contrasts"] = deg["contrasts"]
        # per-pair long form
        with open(os.path.join(args.out, "degradation_per_pair.csv"),
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["objective", "train", "pair", "spacing_rmse"])
            for obj, d in deg["per_obj"].items():
                for tag, ev in (("clean", d["evC"]), ("noisy", d["evN"])):
                    for r in ev.per_pair:
                        w.writerow([obj, tag, r["pair"], r["spacing_rmse"]])
        fig_degradation(deg, os.path.join(args.out, "fig_degradation.png"))
        # pick an example test pair spanning the median gap
        ex = store.precise(test_names[len(test_names) // 2])
        fig_example_pair(model, deg, ex,
                         os.path.join(args.out, "fig_example_pair.png"))
        print(f"    [A] done ({time.time() - t0:.1f}s)")

    if not args.no_delta:
        print("[B] per-pair Delta-theta ...")
        t0 = time.time()
        dfd = analysis_delta_theta(model, store, train_names, features,
                                   pp_de_kw, args.rail_rtol, args.limit_pairs)
        dfd.to_csv(os.path.join(args.out, "delta_theta_per_pair.csv"),
                   index=False)
        summary["delta_theta"] = _delta_aggregate(dfd)
        fig_delta_theta(dfd, os.path.join(args.out, "fig_delta_theta.png"))
        print(f"    [B] done ({time.time() - t0:.1f}s)")

    if not args.no_seg:
        print("[C] segmentation stability ...")
        dfs = analysis_seg_stability(store, train_names, args.dt,
                                     args.match_tol_s, args.limit_pairs)
        dfs.to_csv(os.path.join(args.out, "seg_stability.csv"), index=False)
        summary["seg_stability"] = {
            "median_matched_frac": float(dfs["matched_frac"].median()),
            "median_boundary_mae_s": float(np.nanmedian(dfs["boundary_mae_s"])),
            "median_cp_raw": float(dfs["n_cp_raw"].median()),
            "median_cp_precise": float(dfs["n_cp_precise"].median())}

    if args.sigma_sweep:
        sigmas = [float(s) for s in args.sigma_sweep.split(",") if s.strip() != ""]
        print(f"[sigma] sweep {sigmas} ...")
        dfsig = sigma_sweep(model, store, train_names, test_names, features,
                            de_kw, sigmas, args.noise_seed, args.noise_target)
        dfsig.to_csv(os.path.join(args.out, "sigma_sweep.csv"), index=False)
        summary["sigma_sweep"] = dfsig.to_dict(orient="records")
        fig_sigma_curve(dfsig, os.path.join(args.out, "fig_sigma_curve.png"))

    with open(os.path.join(args.out, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[out] wrote summary + csvs + figures -> {args.out}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", help="Pairing manifest from merge_robustness_pairs.py")
    p.add_argument("--model", choices=["idm", "gipps", "ovm"], default="idm")
    p.add_argument("--features", default="s_end,dist",
                   help="Phase objective feature keys (comma-sep).")
    p.add_argument("--units", choices=["auto", "si", "feet"], default="feet")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # pooled DE budget (analysis A + sigma-sweep)
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--popsize", type=int, default=15)
    # per-pair DE budget (analysis B; many small fits)
    p.add_argument("--pp-maxiter", type=int, default=40)
    p.add_argument("--pp-popsize", type=int, default=12)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--polish", action="store_true",
                   help="Enable DE polish (off by default: harmful on the "
                        "piecewise-constant phase landscape).")
    # scope / cost
    p.add_argument("--limit-pairs", type=int, default=100,
                   help="Cap pairs used for Delta-theta and segmentation (cost).")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-test", type=int, default=None)
    # analysis toggles
    p.add_argument("--no-degradation", action="store_true")
    p.add_argument("--no-delta", action="store_true")
    p.add_argument("--no-seg", action="store_true")
    # sigma-sweep
    p.add_argument("--sigma-sweep", default=None,
                   help="Comma-sep sigma grid in metres, e.g. 0,0.1,0.25,0.5.")
    p.add_argument("--noise-target", choices=["position", "velocity"],
                   default="position")
    p.add_argument("--noise-seed", type=int, default=7)
    # segmentation-stability knobs
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--match-tol-s", type=float, default=0.5)
    p.add_argument("--rail-rtol", type=float, default=0.01)
    p.add_argument("--pipeline-dir", default="./",
                   help="Directory with cf_models.py etc. (prepended to import "
                        "path; use when running from a separate data folder).")
    p.add_argument("--out", default="robustness_out")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.manifest:
        build_parser().error("--manifest is required (or use --self-test).")
    import matplotlib
    matplotlib.use("Agg")
    return run(args)


# --------------------------------------------------------------------------- #
# Self-test: validate pure helpers + orchestration/figures with light fakes
# --------------------------------------------------------------------------- #
def self_test() -> int:
    import types
    import matplotlib
    matplotlib.use("Agg")

    # ---- pure helpers ----
    tr, te = split_names([f"p{i}" for i in range(10)], 0.2, 42)
    assert len(te) == 2 and len(tr) == 8 and not set(tr) & set(te)
    tr2, te2 = split_names([f"p{i}" for i in range(10)], 0.2, 42)
    assert tr == tr2 and te == te2, "split not deterministic"
    assert obj_spec("phase", ["s_end"])[0] == "phase"
    assert obj_spec("sample_spacing", [])[1]["target"] == "spacing"
    assert is_railed(1.001, 1.0, 3.0, 0.01) and not is_railed(2.0, 1.0, 3.0, 0.01)
    m, mae = match_critical_points([10, 40, 90], [12, 41, 300], 5)
    assert m == 2 and abs(mae - 1.5) < 1e-9, (m, mae)

    # ---- fake pair + noise reproducibility ----
    def mk_pair(name, n=120):
        t = np.arange(n) * 0.1
        v = 12 + 3 * np.sin(t / 3)
        x = np.cumsum(v) * 0.1
        return types.SimpleNamespace(
            name=name, t=t, x_follower=x, v_follower=v,
            a_follower=np.gradient(v, t), spacing=np.full(n, 25.0),
            units_source="feet")
    q1 = inject_noise(mk_pair("p0"), 0.3, 7, "position")
    q2 = inject_noise(mk_pair("p0"), 0.3, 7, "position")
    assert np.allclose(q1.v_follower, q2.v_follower), "noise not reproducible"
    assert not np.allclose(q1.v_follower, mk_pair("p0").v_follower)

    # ---- orchestration with light fakes injected into module globals ----
    g = globals()
    pnames = ["v0", "T", "a_max", "b", "s0"]
    bounds = [(15, 35), (0.8, 3.0), (0.5, 2.5), (1.0, 3.0), (1.0, 5.0)]

    class FakeModel:
        name = "idm"
        param_names = pnames
        def bounds(self): return bounds
        def param_dict(self, th): return dict(zip(pnames, [float(x) for x in th]))

    def fake_get_model(_): return FakeModel()

    def fake_load_pair(path, units="feet"):
        return mk_pair(os.path.splitext(os.path.basename(path))[0])

    rng = np.random.default_rng(0)

    def fake_calibrate_pairs(model, pairs, objective, segmentations=None,
                             obj_kwargs=None, **kw):
        th = [float(rng.uniform(lo, hi)) for lo, hi in bounds]
        th[3] = 1.0  # rail 'b' to lower bound to exercise the mask
        return types.SimpleNamespace(theta=th, param_dict=model.param_dict(th),
                                     param_names=pnames, fun=1.0, success=True,
                                     pair_names=[p.name for p in pairs])

    def fake_calibrate_pair(model, pair, objective, segmentation=None,
                            obj_kwargs=None, **kw):
        return fake_calibrate_pairs(model, [pair], objective, **kw)

    class FakeEval:
        def __init__(self, pairs, base):
            self.per_pair = [{"pair": p.name, "spacing_rmse": base + 0.1 * i}
                             for i, p in enumerate(pairs)]
            self.aggregate = {"spacing_rmse_mean": base,
                              "spacing_rmse_median": base}
        def rmse_vector(self, _): return np.array([r["spacing_rmse"]
                                                   for r in self.per_pair])

    def fake_evaluate_theta(model, theta, pairs, label=""):
        return FakeEval(pairs, base=4.0 + 0.2 * ("_N" in label))

    def fake_paired_compare(a, b, metric="spacing_rmse", **kw):
        return {"diff_mean": -0.1, "wilcoxon_p": 0.4, "A_better_frac": 0.6}

    def fake_free_simulate(model, theta, pair):
        return types.SimpleNamespace(s=np.asarray(pair.spacing, float)
                                     + 0.5 * np.sin(pair.t))

    def fake_segment_pair(pair):
        return types.SimpleNamespace(critical_points=[30, 70],
                                     decel_points=[30], accel_points=[70],
                                     phases=[])

    g.update(get_model=fake_get_model, load_pair=fake_load_pair,
             calibrate_pairs=fake_calibrate_pairs,
             calibrate_pair=fake_calibrate_pair,
             evaluate_theta=fake_evaluate_theta,
             paired_compare=fake_paired_compare,
             free_simulate=fake_free_simulate, segment_pair=fake_segment_pair)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        names = [f"pair_{i}.csv" for i in range(8)]
        man = pd.DataFrame({"name": names,
                            "raw_path": [os.path.join(d, "raw", n) for n in names],
                            "precise_path": [os.path.join(d, "prc", n) for n in names],
                            "n_frames": 120, "t_start": 0.0, "t_end": 11.9,
                            "aligned": True})
        mpath = os.path.join(d, "manifest.csv"); man.to_csv(mpath, index=False)
        out = os.path.join(d, "out")
        args = build_parser().parse_args(
            ["--manifest", mpath, "--out", out, "--test-frac", "0.25",
             "--sigma-sweep", "0,0.25", "--match-tol-s", "0.5"])
        # run() calls _import_pipeline which would overwrite our fakes; skip it
        # by monkeypatching to a no-op for the test.
        g["_import_pipeline"] = lambda _pd: None
        rc = run(args)
        assert rc == 0
        for fn in ("robustness_summary.json", "degradation_per_pair.csv",
                   "delta_theta_per_pair.csv", "seg_stability.csv",
                   "sigma_sweep.csv", "fig_degradation.png",
                   "fig_delta_theta.png", "fig_example_pair.png",
                   "fig_sigma_curve.png"):
            assert os.path.exists(os.path.join(out, fn)), fn
        summ = json.load(open(os.path.join(out, "robustness_summary.json")))
        # 'b' was railed in every fit -> excluded from identifiable aggregate
        assert summ["delta_theta"]["phase"]["per_param_median"]["b"] is None
    print("[self-test] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
