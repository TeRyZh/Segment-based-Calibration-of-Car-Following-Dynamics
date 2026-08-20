#!/usr/bin/env python3
"""
oa_pelt_sensitivity.py
======================
Sensitivity of the PELT+ CUSUM knobs to unit basis, evaluated in SI on OpenACC.

Motivation
----------
The PELT+ CUSUM knobs shipped in `phase_segmentation.segment_trajectory`
(`cusum_threshold=7.0`, `cusum_drift=1.0`) were tuned on NGSIM *velocity in
ft/s*. This script quantifies how the segmentation responds to those knobs when
the pipeline is run in SI on OpenACC/AstaZero platoon data, across all five
platoon makes for one run and a 120 s window.

Analytical backbone (verified empirically here)
-----------------------------------------------
The CUSUM stage is *exactly scale-covariant*: with v_m = k*v_ft (k = 0.3048),
scaling both knobs by k gives S_m[i] = k*S_ft[i], so `S_m > theta_m` iff
`S_ft > theta_ft` -> identical candidate points. The PELT stage
(cost = len*log(var(position)+eps)) is scale-invariant up to the 1e-10 floor,
since scaling position adds a constant 2*log(k)*n to every segmentation. Hence
the naive conversion (7.0, 1.0) -> (2.1336, 0.3048) is essentially exact, and
any *appropriate* re-tuning for OpenACC would be driven by its different
noise/dynamics, not by units. `--covariance-check` confirms this by segmenting
the same trajectory under ft-knobs-on-ft-velocity vs SI-knobs-on-SI-velocity.

What it produces
----------------
  <outdir>/
    <stem>_sensitivity.csv        long-format: one row per (make, cell)
    <stem>_summary.json           converted-point metrics, elasticities, verdict,
                                  covariance-check result
    <stem>_grid_nphases.png       per-make heatmap of n_phases over theta x drift
    <stem>_slices.png             n_phases vs threshold / vs drift through convert pt
    <stem>_stability_f1.png       boundary F1 vs converted-point reference

Data source (decision D1)
-------------------------
Loads the five makes directly from the raw AstaZero platoon CSV (natively
make-labelled by the Vehicle_order metadata row; one run by construction).
Follower position uses cumulative integral of speed (D4-A), whose segment slope
is exactly the mean velocity that the accel/decel classification compares.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# --- NumPy 2.x shim: phase_segmentation calls np.trapz internally -------------
if not hasattr(np, "trapz"):                       # numpy>=2 removed np.trapz
    np.trapz = np.trapezoid                        # type: ignore[attr-defined]

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import project modules: script sits alongside them locally (calib/); the
# /mnt/project fallback lets it run in this container.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, "/mnt/project"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from phase_segmentation import segment_trajectory   # noqa: E402

FT_TO_M = 0.3048
THR_CONV = round(7.0 * FT_TO_M, 4)     # 2.1336  -- converted cusum_threshold
DRIFT_CONV = round(1.0 * FT_TO_M, 4)   # 0.3048  -- converted cusum_drift


# --------------------------------------------------------------------------- #
# Data loading (raw AstaZero platoon CSV -> five per-vehicle SI series)
# --------------------------------------------------------------------------- #
def _clean_make(s: str) -> str:
    """'Audi(A8)' -> 'Audi A8'."""
    return s.strip().replace("(", " ").replace(")", "").strip()


def _fill_nan(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).copy()
    m = np.isnan(a)
    if m.any() and (~m).any():
        idx = np.arange(len(a))
        a[m] = np.interp(idx[m], idx[~m], a[~m])
    return a


def load_platoon(path: str):
    """Return (t, {make: speed_series}, [makes in platoon order], fs).

    Time is column 0; Speed{i} sits at column 1 + 7*(i-1) in the OpenACC
    schema (Time, Speed/Lat/Lon/Alt/E/N/U per vehicle, Driver*, IVS*). Column
    positions are validated against the header line; a positional fallback is
    used if the header names drift.
    """
    with open(path) as f:
        head = [f.readline().rstrip("\r\n") for _ in range(6)]
    makes = [_clean_make(x) for x in head[1].split(",")[1:6] if x.strip()]
    header = head[5].split(",")

    def find(name: str):
        for i, h in enumerate(header):
            if h.strip().lower() == name.lower():
                return i
        return None

    t_col = find("Time")
    speed_cols = [find(f"Speed{i}") for i in range(1, 6)]
    if t_col is None or any(c is None for c in speed_cols):
        t_col = 0
        speed_cols = [1 + 7 * i for i in range(5)]   # 1, 8, 15, 22, 29

    raw = pd.read_csv(path, skiprows=6, header=None)
    t = np.asarray(raw.iloc[:, t_col], float)
    valid = ~np.isnan(t)
    raw, t = raw.loc[valid].reset_index(drop=True), t[valid]

    speeds = {makes[i]: _fill_nan(np.asarray(raw.iloc[:, speed_cols[i]], float))
              for i in range(len(makes))}
    dt = np.median(np.diff(t))
    fs = float(round(1.0 / dt)) if dt > 0 else 10.0
    return t, speeds, makes, fs


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral, y[0] anchored at 0 (monotone for v>=0)."""
    area = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(area)])


def pick_window(t, speeds, makes, t_start, duration, speed_floor):
    """Choose the [start, end) index window of length `duration` seconds."""
    fs = 1.0 / np.median(np.diff(t))
    n_win = int(round(duration * fs))
    V = np.vstack([speeds[m] for m in makes])           # (n_makes, n)
    if t_start is None:                                  # first all-moving frame
        moving = np.all(V > speed_floor, axis=0)
        start = int(np.argmax(moving)) if moving.any() else 0
    else:
        start = int(np.searchsorted(t, t_start))
    start = max(0, min(start, len(t) - n_win))
    end = min(start + n_win, len(t))
    return start, end


# --------------------------------------------------------------------------- #
# Segmentation metrics + boundary agreement
# --------------------------------------------------------------------------- #
def seg_metrics(t, x, v, penalty, min_seg, thr, drift):
    """Run one segmentation; return metric dict (+ critical-point list)."""
    s = np.zeros_like(v)          # placeholder: s only feeds the (unused) s_end feature
    res = segment_trajectory(t, x, v, s, penalty=penalty,
                             min_segment_length=min_seg,
                             cusum_threshold=thr, cusum_drift=drift)
    durs = np.array([ph.duration for ph in res.phases], float)
    cand = res.diagnostics.get("candidates", {}).get("total_candidates", np.nan)
    return {
        "n_candidates": float(cand),
        "n_cp": len(res.critical_points),
        "n_phases": res.n_phases,
        "n_decel": len(res.decel_points),
        "n_accel": len(res.accel_points),
        "mean_dur": float(np.mean(durs)) if durs.size else np.nan,
        "median_dur": float(np.median(durs)) if durs.size else np.nan,
        "cp": list(res.critical_points),
    }


def match_f1(cp, cp_ref, tol):
    """Matched-boundary F1 between two change-point sets within +/- tol samples."""
    cp, cp_ref = sorted(cp), sorted(cp_ref)
    if not cp and not cp_ref:
        return 1.0
    if not cp or not cp_ref:
        return 0.0
    used = [False] * len(cp_ref)
    tp = 0
    for c in cp:
        best, bestd = -1, tol + 1
        for j, r in enumerate(cp_ref):
            if used[j]:
                continue
            d = abs(c - r)
            if d <= tol and d < bestd:
                best, bestd = j, d
        if best >= 0:
            used[best] = True
            tp += 1
    prec, rec = tp / len(cp), tp / len(cp_ref)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def covariance_check(t, x, v, penalty, min_seg):
    """ft-knobs on ft-velocity vs SI-knobs on SI-velocity -> should match."""
    m_ft = seg_metrics(t, x / FT_TO_M, v / FT_TO_M, penalty, min_seg, 7.0, 1.0)
    m_si = seg_metrics(t, x, v, penalty, min_seg, THR_CONV, DRIFT_CONV)
    cf, cs = set(m_ft["cp"]), set(m_si["cp"])
    return {"match": cf == cs, "n_cp_ft": len(cf), "n_cp_si": len(cs),
            "symdiff": sorted(cf ^ cs)}


def _elasticity(axis_vals, nphases_along, conv_val):
    """d ln(n_phases) / d ln(param) at the converted value (central diff)."""
    a = np.asarray(axis_vals, float)
    idx = int(np.argmin(np.abs(a - conv_val)))
    lo, hi = idx - 1, idx + 1
    if lo < 0:
        lo = idx
    if hi >= len(a):
        hi = idx
    if lo == hi:
        return float("nan")
    n_lo, n_hi, x_lo, x_hi = nphases_along[lo], nphases_along[hi], a[lo], a[hi]
    if min(n_lo, n_hi, x_lo, x_hi) <= 0 or x_hi == x_lo:
        return float("nan")
    return (np.log(n_hi) - np.log(n_lo)) / (np.log(x_hi) - np.log(x_lo))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _heat_axes(ax, Z, thresholds, drifts, jx, iy, title, cbar_label, annotate):
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(drifts)))
    ax.set_xticklabels([f"{d:.2f}" for d in drifts], rotation=45, fontsize=7)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{th:.2f}" for th in thresholds], fontsize=7)
    ax.set_xlabel("cusum_drift  [m/s]", fontsize=8)
    ax.set_ylabel("cusum_threshold  [m/s]", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.scatter([jx], [iy], marker="*", s=140, c="white",
               edgecolor="black", linewidths=0.8, zorder=5)
    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center",
                            fontsize=6, color="w")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        cbar_label, fontsize=8)


def fig_grid(Zmap, cv, thresholds, drifts, makes, jx, iy, path, annotate):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for k, m in enumerate(makes):
        _heat_axes(axes[k], Zmap[m], thresholds, drifts, jx, iy,
                   m, "n_phases", annotate)
    _heat_axes(axes[5], cv, thresholds, drifts, jx, iy,
               "cross-make CV of n_phases", "CV", annotate)
    fig.suptitle("PELT+ CUSUM knob sensitivity (SI, OpenACC)  -  star = "
                 f"converted NGSIM point ({THR_CONV:.2f}, {DRIFT_CONV:.2f})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_slices(Zmap, thresholds, drifts, makes, jx, iy, path):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 5.2))
    for m in makes:
        a0.plot(thresholds, Zmap[m][:, jx], marker="o", ms=4, label=m)
    a0.axvline(THR_CONV, color="k", ls="--", lw=1)
    a0.set_xlabel("cusum_threshold  [m/s]")
    a0.set_ylabel("n_phases")
    a0.set_title(f"n_phases vs threshold  (drift fixed @ {DRIFT_CONV:.2f})")
    a0.grid(alpha=0.3)
    a0.legend(fontsize=8)
    for m in makes:
        a1.plot(drifts, Zmap[m][iy, :], marker="o", ms=4, label=m)
    a1.axvline(DRIFT_CONV, color="k", ls="--", lw=1)
    a1.set_xlabel("cusum_drift  [m/s]")
    a1.set_ylabel("n_phases")
    a1.set_title(f"n_phases vs drift  (threshold fixed @ {THR_CONV:.2f})")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    fig.suptitle("Sensitivity slices through the converted NGSIM operating point",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_stability(Fmap, thresholds, drifts, makes, jx, iy, path, annotate):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for k, m in enumerate(makes):
        im = axes[k].imshow(Fmap[m], origin="lower", aspect="auto",
                            cmap="magma", vmin=0.0, vmax=1.0)
        axes[k].set_xticks(range(len(drifts)))
        axes[k].set_xticklabels([f"{d:.2f}" for d in drifts], rotation=45, fontsize=7)
        axes[k].set_yticks(range(len(thresholds)))
        axes[k].set_yticklabels([f"{th:.2f}" for th in thresholds], fontsize=7)
        axes[k].set_xlabel("cusum_drift  [m/s]", fontsize=8)
        axes[k].set_ylabel("cusum_threshold  [m/s]", fontsize=8)
        axes[k].set_title(m, fontsize=9)
        axes[k].scatter([jx], [iy], marker="*", s=140, c="white",
                        edgecolor="black", linewidths=0.8, zorder=5)
        if annotate:
            for i in range(Fmap[m].shape[0]):
                for j in range(Fmap[m].shape[1]):
                    axes[k].text(j, i, f"{Fmap[m][i, j]:.2f}", ha="center",
                                 va="center", fontsize=6, color="w")
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04).set_label(
            "F1 vs converted", fontsize=8)
    mean_F = np.nanmean(np.stack([Fmap[m] for m in makes]), axis=0)
    im = axes[5].imshow(mean_F, origin="lower", aspect="auto",
                        cmap="magma", vmin=0.0, vmax=1.0)
    axes[5].set_xticks(range(len(drifts)))
    axes[5].set_xticklabels([f"{d:.2f}" for d in drifts], rotation=45, fontsize=7)
    axes[5].set_yticks(range(len(thresholds)))
    axes[5].set_yticklabels([f"{th:.2f}" for th in thresholds], fontsize=7)
    axes[5].set_title("mean F1 across makes", fontsize=9)
    axes[5].scatter([jx], [iy], marker="*", s=140, c="white",
                    edgecolor="black", linewidths=0.8, zorder=5)
    plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)
    fig.suptitle("Boundary-set stability: matched-boundary F1 vs the converted "
                 "point (star, F1=1 by construction)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_grids(args):
    if args.smoke:
        thr = sorted({1.0, THR_CONV, 3.5, 5.0})
        drf = sorted({0.15, DRIFT_CONV, 0.6})
    else:
        thr = args.thresholds or list(np.round(np.linspace(0.5, 6.0, 12), 3))
        drf = args.drifts or list(np.round(np.linspace(0.05, 1.0, 10), 3))
        thr = sorted(set(thr) | {THR_CONV})
        drf = sorted(set(drf) | {DRIFT_CONV})
    return thr, drf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="/mnt/project/ASta_040719_platoon7.csv")
    p.add_argument("--outdir", default="/mnt/user-data/outputs")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--t-start", type=float, default=None,
                   help="Window start (s). Default: first all-moving frame.")
    p.add_argument("--speed-floor", type=float, default=1.0,
                   help="Auto-start when every make exceeds this speed [m/s].")
    p.add_argument("--thresholds", type=float, nargs="+", default=None)
    p.add_argument("--drifts", type=float, nargs="+", default=None)
    p.add_argument("--penalty", type=float, default=75.0)
    p.add_argument("--min-seg", type=int, default=20)
    p.add_argument("--penalty-spot", type=float, nargs="+", default=[50, 75, 100])
    p.add_argument("--tol-sec", type=float, default=0.5,
                   help="Boundary-match tolerance for F1 [s].")
    p.add_argument("--annotate", action="store_true",
                   help="Force per-cell value annotation on heatmaps.")
    p.add_argument("--smoke", action="store_true",
                   help="Coarse 4x3 grid for a quick end-to-end check.")
    args = p.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]

    # ---- load + window
    t_all, speeds, makes, fs = load_platoon(args.csv)
    i0, i1 = pick_window(t_all, speeds, makes, args.t_start,
                         args.duration, args.speed_floor)
    t = t_all[i0:i1]
    tol = int(round(args.tol_sec * fs))
    traj = {m: (t, _cumtrapz(speeds[m][i0:i1], t), speeds[m][i0:i1]) for m in makes}

    print("=" * 74)
    print(f"OpenACC PELT+ CUSUM sensitivity  |  {os.path.basename(args.csv)}")
    print(f"window: t=[{t[0]:.1f}, {t[-1]:.1f}] s  ({len(t)} samples @ {fs:.0f} Hz)"
          f"  |  makes: {', '.join(makes)}")
    print(f"converted NGSIM point: threshold={THR_CONV:.4f}, drift={DRIFT_CONV:.4f} "
          f"m/s  |  F1 tol = +/-{tol} samples")

    thresholds, drifts = build_grids(args)
    jx = int(np.argmin(np.abs(np.array(drifts) - DRIFT_CONV)))
    iy = int(np.argmin(np.abs(np.array(thresholds) - THR_CONV)))
    annotate = args.annotate or (len(thresholds) * len(drifts) <= 20)

    # ---- covariance sanity check
    print("\n[scale-covariance]  ft-knobs-on-ft-velocity  vs  SI-knobs-on-SI-velocity")
    cov = {}
    for m in makes:
        c = covariance_check(*traj[m], args.penalty, args.min_seg)
        cov[m] = c
        tag = "MATCH" if c["match"] else f"DIFF symdiff={c['symdiff']}"
        print(f"    {m:16s} cp_ft={c['n_cp_ft']:2d}  cp_si={c['n_cp_si']:2d}   {tag}")

    # ---- reference (converted) segmentation per make
    ref = {m: seg_metrics(*traj[m], args.penalty, args.min_seg,
                          THR_CONV, DRIFT_CONV) for m in makes}

    # ---- grid sweep
    Zmap = {m: np.full((len(thresholds), len(drifts)), np.nan) for m in makes}
    Fmap = {m: np.full((len(thresholds), len(drifts)), np.nan) for m in makes}
    rows = []
    for m in makes:
        tt, xx, vv = traj[m]
        for i, th in enumerate(thresholds):
            for j, dr in enumerate(drifts):
                mm = seg_metrics(tt, xx, vv, args.penalty, args.min_seg, th, dr)
                f1 = match_f1(mm["cp"], ref[m]["cp"], tol)
                Zmap[m][i, j] = mm["n_phases"]
                Fmap[m][i, j] = f1
                rows.append({
                    "make": m, "sweep_kind": "grid",
                    "cusum_threshold": th, "cusum_drift": dr,
                    "penalty": args.penalty, "min_segment_length": args.min_seg,
                    "n_candidates": mm["n_candidates"], "n_critical_points": mm["n_cp"],
                    "n_phases": mm["n_phases"], "n_decel": mm["n_decel"],
                    "n_accel": mm["n_accel"], "mean_duration_s": mm["mean_dur"],
                    "median_duration_s": mm["median_dur"], "f1_vs_converted": f1,
                })

    cv = np.nanstd(np.stack([Zmap[m] for m in makes]), axis=0) / \
        np.clip(np.nanmean(np.stack([Zmap[m] for m in makes]), axis=0), 1e-9, None)

    # ---- penalty spot-check at converted CUSUM point
    print(f"\n[penalty spot-check]  at converted CUSUM point, penalties="
          f"{[int(x) for x in args.penalty_spot]}  (n_phases per make)")
    hdr = "    " + "make".ljust(16) + "".join(f"p={int(x):<7d}" for x in args.penalty_spot)
    print(hdr)
    for m in makes:
        tt, xx, vv = traj[m]
        cells = []
        for pen in args.penalty_spot:
            mm = seg_metrics(tt, xx, vv, pen, args.min_seg, THR_CONV, DRIFT_CONV)
            f1 = match_f1(mm["cp"], ref[m]["cp"], tol)
            cells.append(mm["n_phases"])
            rows.append({
                "make": m, "sweep_kind": "penalty",
                "cusum_threshold": THR_CONV, "cusum_drift": DRIFT_CONV,
                "penalty": pen, "min_segment_length": args.min_seg,
                "n_candidates": mm["n_candidates"], "n_critical_points": mm["n_cp"],
                "n_phases": mm["n_phases"], "n_decel": mm["n_decel"],
                "n_accel": mm["n_accel"], "mean_duration_s": mm["mean_dur"],
                "median_duration_s": mm["median_dur"], "f1_vs_converted": f1,
            })
        print("    " + m.ljust(16) + "".join(f"{c:<9d}" for c in cells))

    # ---- elasticities + verdict at converted point
    print("\n[converted-point readout]  n_phases, local elasticity, verdict")
    print("    " + "make".ljust(16) + "n_ph  n_cp  mean_dur  el_thr   el_drift  verdict")
    summary_makes = {}
    for m in makes:
        el_thr = _elasticity(thresholds, Zmap[m][:, jx], THR_CONV)
        el_drf = _elasticity(drifts, Zmap[m][iy, :], DRIFT_CONV)
        worst = np.nanmax([abs(el_thr), abs(el_drf)])
        verdict = ("plateau" if worst < 0.5 else
                   "moderate" if worst < 1.0 else "sensitive")
        summary_makes[m] = {
            "n_phases": ref[m]["n_phases"], "n_critical_points": ref[m]["n_cp"],
            "mean_duration_s": ref[m]["mean_dur"], "n_candidates": ref[m]["n_candidates"],
            "elasticity_threshold": None if np.isnan(el_thr) else round(el_thr, 3),
            "elasticity_drift": None if np.isnan(el_drf) else round(el_drf, 3),
            "verdict": verdict,
        }
        md = "  nan  " if np.isnan(ref[m]["mean_dur"]) else f"{ref[m]['mean_dur']:6.2f}"
        et = " nan " if np.isnan(el_thr) else f"{el_thr:6.2f}"
        ed = " nan " if np.isnan(el_drf) else f"{el_drf:6.2f}"
        print(f"    {m.ljust(16)}{ref[m]['n_phases']:<6d}{ref[m]['n_cp']:<6d}"
              f"{md}   {et}   {ed}   {verdict}")

    # ---- write outputs
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, f"{stem}_sensitivity.csv")
    df.to_csv(csv_path, index=False)

    grid_png = os.path.join(args.outdir, f"{stem}_grid_nphases.png")
    slice_png = os.path.join(args.outdir, f"{stem}_slices.png")
    stab_png = os.path.join(args.outdir, f"{stem}_stability_f1.png")
    fig_grid(Zmap, cv, thresholds, drifts, makes, jx, iy, grid_png, annotate)
    fig_slices(Zmap, thresholds, drifts, makes, jx, iy, slice_png)
    fig_stability(Fmap, thresholds, drifts, makes, jx, iy, stab_png, annotate)

    summary = {
        "csv_file": os.path.basename(args.csv),
        "window": {"t_start_s": float(t[0]), "t_end_s": float(t[-1]),
                   "n_samples": int(len(t)), "fs_hz": fs},
        "makes": makes,
        "converted_point": {"cusum_threshold": THR_CONV, "cusum_drift": DRIFT_CONV,
                            "penalty": args.penalty, "min_segment_length": args.min_seg},
        "grid": {"thresholds": [float(x) for x in thresholds],
                 "drifts": [float(x) for x in drifts],
                 "f1_tolerance_samples": tol},
        "covariance_check": {m: {"match": bool(cov[m]["match"]),
                                 "n_cp_ft": cov[m]["n_cp_ft"],
                                 "n_cp_si": cov[m]["n_cp_si"],
                                 "symdiff": cov[m]["symdiff"]} for m in makes},
        "per_make": summary_makes,
        "covariance_all_match": bool(all(cov[m]["match"] for m in makes)),
    }
    json_path = os.path.join(args.outdir, f"{stem}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nwrote:")
    for pth in (csv_path, grid_png, slice_png, stab_png, json_path):
        print(f"    {pth}")
    print("=" * 74)


if __name__ == "__main__":
    main()
