#!/usr/bin/env python3
"""
run_demo.py
===========
End-to-end demonstration of the phase-transition calibration framework on a
single car-following pair.

Pipeline (mirrors the four methodological steps):
    1. load a pair  ->  segment the observed follower into behavioural phases
       (PELT+ critical points = accel<->decel switches)
    2. simulate the follower forward continuously under a global parameter set
    3. score fit only at the phase boundaries (phase-anchored objective)
    4. globally optimise theta with differential evolution
and, for the head-to-head the manuscript rests on, repeats step 4 with a
conventional sample-based RMSE baseline (decision D9).

Pair resolution
---------------
Uses the real extracted pair (follower 1749 / leader 1747) if a matching CSV is
found under the uploads folder, the CWD, or the project directory.  Otherwise it
synthesises an IDM-generated stop-and-go pair with a *known* theta* so the demo
doubles as the bias check (ground-truth recovery) described in the research plan.

Outputs (written to /mnt/user-data/outputs)
    - calibration_demo.png    3-panel figure: speed / spacing / position vs time,
                              PELT+ critical points marked, both calibrated
                              simulations overlaid on the observed trajectory
    - calibration_demo.json   theta* for both objectives + per-phase feature table

A note that this session surfaced (see the feature-sensitivity block below):
under NGSIM-style position noise the phase objective is robust *only* when its
boundary features are position-level (terminal spacing, cumulative distance).
Reading terminal *velocity* at the boundaries re-imports the very
differentiation-noise pathology the framework is meant to avoid, so the demo
defaults the phase features to {s_end, dist}.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cf_models import get_model
from cf_data import PairData, load_pair, REQUIRED_COLS
from phase_segmentation import segment_trajectory, SegmentationResult
from simulate import simulate, SimResult
from objectives import PhaseAnchoredObjective
from calibrate import calibrate_pair

OUTPUT_DIR = "outputs"
THETA_TRUE = np.array([16.5, 1.2, 1.4, 2.2, 2.0])   # IDM v0,T,a_max,b,s0 (synthetic)
PHASE_FEATURES = ("s_end", "dist")                  # position-robust default
SEARCH_DIRS = ("/mnt/user-data/uploads", ".", "/mnt/project", "/home/claude")


# --------------------------------------------------------------------------- #
# Pair resolution
# --------------------------------------------------------------------------- #
def _looks_like_pair(path: str) -> bool:
    try:
        import pandas as pd
        head = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    return set(REQUIRED_COLS).issubset(head.columns)


def find_real_pair() -> Optional[str]:
    """Look for the 1749/1747 pair (or any valid pair CSV) in the usual places."""
    # prefer a filename that names the follower/leader ids
    named: List[str] = []
    generic: List[str] = []
    for d in SEARCH_DIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
            base = os.path.basename(p).lower()
            if "pair_clean" in base or "pair_noisy" in base or base.startswith("_probe"):
                continue                              # skip our own fixtures/probes
            if _looks_like_pair(p):
                if "1749" in base or "1747" in base:
                    named.append(p)
                else:
                    generic.append(p)
    if named:
        return named[0]
    return generic[0] if generic else None


def synthesize(noise_x: float, seed: int = 7) -> "pd.DataFrame":   # noqa: F821
    """IDM stop-and-go pair with known theta* (feeds the bias check)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    dt, T = 0.1, 70.0
    t = np.arange(0.0, T + dt / 2, dt)
    n = len(t)
    vL = np.interp(
        t,
        [0, 12, 20, 30, 40, 48, 56, 64, T],
        [16, 16, 5, 5, 16, 16, 9, 15, 15],
    ).astype(float)
    vL = np.clip(vL, 0, None)
    xL = np.concatenate([[300.0], 300.0 + np.cumsum(0.5 * (vL[1:] + vL[:-1]) * dt)])
    idm = get_model("idm")
    r = simulate(idm, THETA_TRUE, t, xL, vL, leader_length=4.5,
                 x0=xL[0] - 22.0, v0=16.0)
    xf = r.x.copy()
    if noise_x > 0:
        xf = xf + rng.normal(0, noise_x, n)
    vf = np.gradient(xf, t)                            # NGSIM-like differentiation
    af = np.gradient(vf, t)
    L = 4.5
    return pd.DataFrame({
        "t": t, "x_follower": xf, "v_follower": vf, "a_follower": af,
        "x_leader": xL, "v_leader": vL, "a_leader": np.gradient(vL, t),
        "leader_length": L, "spacing": xL - xf - L, "dv": vf - vL})


def resolve_pair() -> Tuple[PairData, bool, Optional[str]]:
    """Return (primary_pair, is_synthetic, noisy_csv_path_if_synthetic)."""
    real = find_real_pair()
    if real is not None:
        print(f"[pair] using real extracted pair: {real}")
        return load_pair(real, units="auto"), False, None

    print("[pair] no real pair found -- synthesising IDM stop-and-go "
          "pair with known theta*")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_p = os.path.join(OUTPUT_DIR, "_synth_pair_clean.csv")
    noisy_p = os.path.join(OUTPUT_DIR, "_synth_pair_noisy.csv")
    synthesize(0.0).to_csv(clean_p, index=False)
    synthesize(0.20).to_csv(noisy_p, index=False)
    # primary = noisy (the realistic, interesting case for the head-to-head)
    return load_pair(noisy_p, units="si"), True, noisy_p


# --------------------------------------------------------------------------- #
# Calibration + reporting
# --------------------------------------------------------------------------- #
def _recovery_line(names, theta, truth) -> str:
    errs = np.abs(np.asarray(theta[:len(truth)]) - truth)
    parts = [f"{nm}:{v:.3f}(|e|{e:.3f})"
             for nm, v, e in zip(names, theta, errs)]
    return "  ".join(parts) + f"   L1={errs.sum():.3f}"


def calibrate_both(model, pair, seg, *, maxiter: int):
    """Phase-anchored (position-robust features) and sample-RMSE baseline."""
    print("\n[calibrate] phase-anchored objective "
          f"(features={'+'.join(PHASE_FEATURES)}) ...")
    t0 = time.time()
    r_phase = calibrate_pair(model, pair, "phase", segmentation=seg,
                             obj_kwargs=dict(feature_keys=PHASE_FEATURES),
                             maxiter=maxiter, seed=42)
    print(f"    done in {time.time()-t0:.1f}s   J*={r_phase.fun:.6g}")

    print("[calibrate] sample-based RMSE baseline (target=spacing) ...")
    t0 = time.time()
    r_samp = calibrate_pair(model, pair, "sample",
                            obj_kwargs=dict(target="spacing", metric="rmse"),
                            maxiter=maxiter, seed=42)
    print(f"    done in {time.time()-t0:.1f}s   RMSE*={r_samp.fun:.6g}")
    return r_phase, r_samp


def feature_sensitivity(model, pair, seg, *, maxiter: int) -> List[dict]:
    """Show that phase robustness hinges on position-level features.

    Only meaningful when theta* is known (synthetic pair)."""
    trials = [("phase / s_end+dist", "phase", dict(feature_keys=("s_end", "dist"))),
              ("phase / s_end",      "phase", dict(feature_keys=("s_end",))),
              ("phase / v_end+s_end", "phase", dict(feature_keys=("v_end", "s_end"))),
              ("phase / v_end",      "phase", dict(feature_keys=("v_end",))),
              ("sample / spacing-RMSE", "sample", dict(target="spacing", metric="rmse")),
              ("sample / speed-RMSE",   "sample", dict(target="speed", metric="rmse"))]
    rows = []
    for label, kind, kw in trials:
        seg_arg = seg if kind == "phase" else None
        r = calibrate_pair(model, pair, kind, segmentation=seg_arg,
                           obj_kwargs=kw, maxiter=maxiter, seed=42)
        errs = np.abs(np.asarray(r.theta[:5]) - THETA_TRUE)
        rows.append({"objective": label, "L1": float(errs.sum()),
                     "theta": [float(x) for x in r.theta[:5]]})
    return rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _sim_series(model, theta, pair: PairData) -> SimResult:
    return simulate(model, theta, pair.t, pair.x_leader, pair.v_leader,
                    pair.leader_length,
                    x0=float(pair.x_follower[0]), v0=float(pair.v_follower[0]))


def make_figure(model, pair, seg, r_phase, r_samp, path: str,
                is_synth: bool) -> None:
    t = pair.t
    sim_p = _sim_series(model, r_phase.theta, pair)
    sim_s = _sim_series(model, r_samp.theta, pair)

    cp_t = np.asarray([pair.t[i] for i in seg.critical_points]) \
        if len(seg.critical_points) else np.array([])
    dec_t = np.asarray([pair.t[i] for i in seg.decel_points]) \
        if len(seg.decel_points) else np.array([])
    acc_t = np.asarray([pair.t[i] for i in seg.accel_points]) \
        if len(seg.accel_points) else np.array([])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True)
    C_OBS, C_LEAD = "#222222", "#8a8a8a"
    C_PH, C_SA = "#c0392b", "#2471a3"

    def mark_cps(ax, y_for_marker=None):
        for x in cp_t:
            ax.axvline(x, color="#d9c200", lw=0.9, ls="--", alpha=0.55, zorder=0)

    # (a) speed -- the axis on which phases are defined
    ax = axes[0]
    ax.plot(t, pair.v_leader, color=C_LEAD, lw=1.2, label="leader (input)")
    ax.plot(t, pair.v_follower, color=C_OBS, lw=1.4, label="follower (observed)")
    ax.plot(t, sim_p.v, color=C_PH, lw=1.6, ls="-",
            label="sim @ phase-θ*")
    ax.plot(t, sim_s.v, color=C_SA, lw=1.4, ls="--",
            label="sim @ sample-θ*")
    mark_cps(ax)
    if len(dec_t):
        ax.scatter(dec_t, np.interp(dec_t, t, pair.v_follower),
                   marker="v", s=55, color="#d9c200", edgecolor="k",
                   zorder=5, label="decel switch")
    if len(acc_t):
        ax.scatter(acc_t, np.interp(acc_t, t, pair.v_follower),
                   marker="^", s=55, color="#27ae60", edgecolor="k",
                   zorder=5, label="accel switch")
    ax.set_ylabel("speed  (m/s)")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ttl = "Phase-transition calibration"
    ttl += "  —  synthetic pair, known θ*" if is_synth else f"  —  pair {pair.name}"
    ax.set_title(ttl, fontsize=12, fontweight="bold")

    # (b) spacing -- the CF gap
    ax = axes[1]
    ax.plot(t, pair.spacing, color=C_OBS, lw=1.4, label="observed gap")
    ax.plot(t, sim_p.s, color=C_PH, lw=1.6, label="sim @ phase-θ*")
    ax.plot(t, sim_s.s, color=C_SA, lw=1.4, ls="--", label="sim @ sample-θ*")
    mark_cps(ax)
    ax.set_ylabel("net spacing  (m)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # (c) position
    ax = axes[2]
    ax.plot(t, pair.x_leader, color=C_LEAD, lw=1.2, label="leader")
    ax.plot(t, pair.x_follower, color=C_OBS, lw=1.4, label="follower (observed)")
    ax.plot(t, sim_p.x, color=C_PH, lw=1.6, label="sim @ phase-θ*")
    ax.plot(t, sim_s.x, color=C_SA, lw=1.4, ls="--", label="sim @ sample-θ*")
    mark_cps(ax)
    ax.set_ylabel("position  (m)")
    ax.set_xlabel("time  (s)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Phase-transition calibration demo.")
    ap.add_argument("--model", default="idm", help="idm | gipps | ovm")
    ap.add_argument("--maxiter", type=int, default=40,
                    help="differential-evolution generations (default 40)")
    ap.add_argument("--pair", default=None,
                    help="explicit pair CSV (overrides auto-resolution)")
    args = ap.parse_args(argv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = get_model(args.model)

    if args.pair:
        pair, is_synth, _ = load_pair(args.pair, units="auto"), False, None
        print(f"[pair] using explicit pair: {args.pair}")
    else:
        pair, is_synth, _ = resolve_pair()

    print(f"[pair] name={pair.name}  n={pair.n}  dt={pair.dt:.3f}s  "
          f"units={pair.units_source}")
    print(f"       speed=[{pair.v_follower.min():.1f},{pair.v_follower.max():.1f}] m/s"
          f"   spacing=[{pair.spacing.min():.1f},{pair.spacing.max():.1f}] m")

    seg = segment_trajectory(pair.t, pair.x_follower, pair.v_follower, pair.spacing)
    print(f"[segment] {seg.n_phases} phases, {len(seg.critical_points)} critical "
          f"points ({len(seg.decel_points)} decel, {len(seg.accel_points)} accel)")

    r_phase, r_samp = calibrate_both(model, pair, seg, maxiter=args.maxiter)

    # ---- theta* report -----------------------------------------------------
    names = list(model.param_names)
    print("\n" + "=" * 70)
    print(f"CALIBRATED θ*   (model={model.name})")
    print("=" * 70)
    print("phase-anchored :")
    for nm, v in r_phase.param_dict.items():
        print(f"    {nm:8s} = {v:.4f}")
    print("sample-RMSE    :")
    for nm, v in r_samp.param_dict.items():
        print(f"    {nm:8s} = {v:.4f}")
    if is_synth:
        print("\nθ* recovery (|e| vs known truth):")
        print("  phase : " + _recovery_line(names, r_phase.theta, THETA_TRUE))
        print("  sample: " + _recovery_line(names, r_samp.theta, THETA_TRUE))

    # ---- per-phase observed vs simulated table -----------------------------
    obj = PhaseAnchoredObjective(model, pair, seg, feature_keys=PHASE_FEATURES)
    print("\nper-phase (observed vs simulated @ phase-θ*):")
    hdr = f"  {'k':>2} {'kind':>6} {'t0':>6} {'t1':>6}"
    for key in PHASE_FEATURES:
        hdr += f" {key+'_obs':>10} {key+'_sim':>10} {key+'_res':>9}"
    print(hdr)
    for row in obj.per_phase_table(r_phase.theta):
        line = f"  {row['k']:>2} {row['kind']:>6} {row['t0']:>6.1f} {row['t1']:>6.1f}"
        for key in PHASE_FEATURES:
            line += (f" {row[key+'_obs']:>10.3f} {row[key+'_sim']:>10.3f}"
                     f" {row[key+'_res']:>9.3f}")
        print(line)

    # ---- feature-sensitivity finding (synthetic only) ----------------------
    sens_rows = None
    if is_synth:
        print("\n" + "=" * 70)
        print("FEATURE SENSITIVITY  (θ* recovery under NGSIM-style position noise)")
        print("  → phase robustness depends on POSITION-level boundary features;")
        print("    terminal-velocity features re-import differentiation noise.")
        print("=" * 70)
        sens_rows = feature_sensitivity(model, pair, seg, maxiter=args.maxiter)
        print(f"  {'objective':24s}{'L1 err':>9s}   θ=(v0,T,a,b,s0)")
        for r in sens_rows:
            th = ", ".join(f"{x:.2f}" for x in r["theta"])
            print(f"  {r['objective']:24s}{r['L1']:9.3f}   [{th}]")

    # ---- figure + json -----------------------------------------------------
    fig_path = os.path.join(OUTPUT_DIR, "calibration_demo.png")
    make_figure(model, pair, seg, r_phase, r_samp, fig_path, is_synth)
    print(f"\n[figure] wrote {fig_path}")

    out = {
        "model": model.name,
        "pair": pair.name,
        "is_synthetic": is_synth,
        "n_samples": pair.n,
        "dt": pair.dt,
        "n_phases": seg.n_phases,
        "critical_points_t": [float(pair.t[i]) for i in seg.critical_points],
        "phase_features": list(PHASE_FEATURES),
        "theta_phase": r_phase.param_dict,
        "theta_sample": r_samp.param_dict,
        "J_phase": r_phase.fun,
        "RMSE_sample": r_samp.fun,
        "per_phase_table": obj.per_phase_table(r_phase.theta),
    }
    if is_synth:
        out["theta_true"] = {n: float(v) for n, v in zip(names, THETA_TRUE)}
        out["feature_sensitivity"] = sens_rows
    json_path = os.path.join(OUTPUT_DIR, "calibration_demo.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json]   wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
