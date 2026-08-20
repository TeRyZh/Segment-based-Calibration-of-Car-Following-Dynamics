#!/usr/bin/env python3
"""
acc_controller_heterogeneity.py
===============================
Three advanced Section 5.3 figures from the SEGMENT-ANCHORED OVM calibration on
the OpenACC AstaZero platoon (`ASta_040719_platoon7.csv`).

  Concept 1  Speed-spacing (state-space) overlay   [fig5_6_speed_spacing_overlay.png]
             2x2, one panel per follower make. Observed (v_follower, s) scatter
             coloured by PELT+ segment regime (accel / decel / equilibrium),
             with the learned optimal-velocity locus {(V_opt(s), s)} overlaid as
             a thick black line and the Critical Spacing S_c marked. Shows the
             learned equilibrium bisecting the segmented human-driving cloud.

  Concept 2  Controller heterogeneity radar         [fig5_7_controller_radar.png]
             Single spider chart, four overlapping polygons (one per make) over
             five axes: response time 1/kappa, Critical Spacing S_c, transition
             width w, median operating gap, peak deceleration. Each axis is
             normalised to its across-make maximum.

  Concept 3  Stimulus-response temporal alignment   [fig5_8_stimulus_response_<make>.png]
             Two stacked panels on the platoon-wide busiest ~120 s window. Top:
             observed relative speed dv = v_leader - v_follower and net spacing
             s(t), with PELT+ accel/decel segments shaded. Bottom: observed
             follower acceleration a_f(t) vs the segment-anchored OVM
             acceleration DELAYED by the empirical lag tau* (Fig 5-4 aggregate).

Reuses acc_controller_behavior (segmentation + Savitzky-Golay accel + regimes +
peak extraction) and cf_models.OVM verbatim; no project module is modified.

Terminology: the behavioural chunk between two critical points is a *segment*
(formerly "phase"); calibration is *segment-anchored*. calib_summary.csv still
stores the learned parameters under the legacy *_phase suffix, so the loader
accepts either *_segment or *_phase columns.

Single run, one vehicle per make, platoon position confounded -> every
cross-make number is a DESCRIPTIVE per-position characterisation, NOT
manufacturer inference. a_f and peak deceleration are Savitzky-Golay
derivatives (kept descriptive, consistent with the manuscript's
differentiation-noise critique). In Concept 3, tau* is a MODEL-FREE lag overlaid
post-hoc on the model-based OVM acceleration; the calibration does not jointly
learn it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import acc_controller_behavior as acb
from cf_models import get_model

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CSV_DEFAULT = "ASta_040719_platoon7.csv"
CALIB_DEFAULT = "acc_calib_ovm_run7/calib_summary.csv"

OVM = get_model("ovm")
OVM_A_MAX, OVM_D_MAX = 2.5, -4.0        # cf_models.OVM.accel kinematic clip

ORDER = [2, 3, 4, 5]                     # follower positions (== follower_veh)
POS_MAKE = {2: "Tesla Model 3", 3: "BMW X5", 4: "Audi A6", 5: "Mercedes A-Class"}
MAKE_COL = {
    "Tesla Model 3": "#0072B2", "BMW X5": "#009E73",
    "Audi A6": "#E69F00", "Mercedes A-Class": "#CC3311",
}
# aggregate response lag tau* from Figure 5-4 (n=10 windowed means, published)
TAU_STAR_FIG54 = {"Tesla Model 3": 1.70, "BMW X5": 2.60,
                  "Audi A6": 2.30, "Mercedes A-Class": 2.10}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def v_opt(s, v_max, s_c, w):
    """Normalised OVM optimal-velocity V_opt(s) (vectorised); matches
    cf_models.OVM._v_opt."""
    s = np.asarray(s, float)
    t_sc = np.tanh(s_c / w)
    return v_max * (np.tanh((s - s_c) / w) + t_sc) / (1.0 + t_sc)


def ovm_accel_series(s, v, theta):
    """Instantaneous OVM acceleration on observed states (memoryless in dv),
    with the same clip as cf_models.OVM.accel."""
    kappa, v_max, s_c, w = theta
    a = kappa * (v_opt(s, v_max, s_c, w) - np.asarray(v, float))
    return np.clip(a, OVM_D_MAX, OVM_A_MAX)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _first_existing(cands: List[str]) -> Optional[str]:
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def find_csv(explicit: Optional[str]) -> str:
    csv = explicit or _first_existing(
        [CSV_DEFAULT, f"/mnt/project/{CSV_DEFAULT}"]
        + sorted(glob.glob("/mnt/user-data/uploads/*.csv")))
    if not csv:
        sys.exit("error: no OpenACC AstaZero CSV found (pass --csv).")
    return csv


def find_calib(explicit: Optional[str]) -> str:
    cal = explicit or _first_existing(
        [CALIB_DEFAULT, "calib_summary.csv",
         "/mnt/user-data/uploads/calib_summary.csv"])
    if not cal:
        sys.exit("error: no calib_summary.csv found (pass --calib).")
    return cal


def load_calib(path: str) -> pd.DataFrame:
    """Load calib_summary.csv, map position->make, and alias the learned OVM
    parameters from *_segment (preferred) or legacy *_phase columns."""
    cal = pd.read_csv(path)
    cal["make"] = cal["follower_position"].map(POS_MAKE)
    for base in ("kappa", "v_max", "s_c", "w"):
        seg, ph = f"{base}_segment", f"{base}_phase"
        col = seg if seg in cal.columns else ph
        if col not in cal.columns:
            sys.exit(f"error: calib_summary.csv missing '{seg}'/'{ph}'.")
        cal[base] = cal[col].astype(float)
    return cal


def theta_for(cal: pd.DataFrame, pos: int) -> List[float]:
    r = cal[cal["follower_position"] == pos].iloc[0]
    return [float(r.kappa), float(r.v_max), float(r.s_c), float(r.w)]


def make_analysis_args() -> types.SimpleNamespace:
    """SimpleNamespace mirroring acc_controller_behavior's parser defaults, so
    acb.analyze_pair runs the standard SI segmentation + accel pipeline."""
    return types.SimpleNamespace(
        sg_window=15, sg_poly=2,
        penalty=55.0, min_seg=20,
        cusum_thresh=acb.SI_CUSUM_THRESH, cusum_drift=acb.SI_CUSUM_DRIFT,
        deadband=0.2, tau_max=5.0,
        eps_stable=0.15, tmin=3.0,
    )


# --------------------------------------------------------------------------- #
# Concept 1: speed-spacing (state-space) overlay
# --------------------------------------------------------------------------- #
def _regime_index_groups(regimes, n) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc, dec, eq = [], [], []
    for (i0, i1, reg) in regimes:
        idx = np.arange(i0, min(i1, n - 1) + 1)
        (acc if reg == "accel" else dec if reg == "decel" else eq).append(idx)
    cat = lambda L: (np.concatenate(L) if L else np.array([], int))
    return cat(acc), cat(dec), cat(eq)


def figure_speed_spacing_overlay(cal, analyses, outdir) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.2))
    axes = axes.ravel()
    for ax, pos in zip(axes, ORDER):
        make = POS_MAKE[pos]
        col = MAKE_COL[make]
        pa = analyses[pos]
        v = np.asarray(pa.pair.v_follower, float)
        s = np.asarray(pa.pair.s, float)
        acc, dec, eq = _regime_index_groups(pa.regimes, len(v))

        ax.scatter(v[eq], s[eq], s=5, c="0.6", alpha=0.10, lw=0, zorder=1)
        ax.scatter(v[dec], s[dec], s=5, c=acb.C_DECEL, alpha=0.16, lw=0, zorder=2)
        ax.scatter(v[acc], s[acc], s=5, c=acb.C_ACCEL, alpha=0.16, lw=0, zorder=2)

        kappa, v_max, s_c, w = theta_for(cal, pos)
        s_lo = max(0.5, float(np.nanpercentile(s, 1)))
        s_hi = float(np.nanpercentile(s, 99))
        s_grid = np.linspace(s_lo, s_hi, 400)
        ax.plot(v_opt(s_grid, v_max, s_c, w), s_grid, color="k", lw=2.6,
                zorder=5, label=r"$V_{\mathrm{opt}}(s)$ (segment-anchored)")
        ax.scatter([v_opt(s_c, v_max, s_c, w)], [s_c], s=90, color="k",
                   zorder=6, edgecolor="white", linewidth=1.2,
                   label=r"$S_c$ = %.1f m" % s_c)

        ax.set_title(make, color=col, fontsize=12.5)
        ax.set_xlabel("follower speed (m/s)")
        ax.set_ylabel("net spacing $s$ (m)")
        ax.set_xlim(max(0.0, float(np.nanpercentile(v, 0.5)) - 1.0),
                    float(np.nanpercentile(v, 99.5)) + 1.0)
        ax.set_ylim(max(0.0, s_lo - 2.0), s_hi + 3.0)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    handles = [
        Line2D([], [], marker="o", ls="", color=acb.C_ACCEL, label="accel segment"),
        Line2D([], [], marker="o", ls="", color=acb.C_DECEL, label="decel segment"),
        Line2D([], [], marker="o", ls="", color="0.6", label="equilibrium segment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle("Segment-anchored OVM equilibrium vs observed driving regimes",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    out = os.path.join(outdir, "fig5_6_speed_spacing_overlay.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# Concept 2: controller heterogeneity radar
# --------------------------------------------------------------------------- #
def figure_controller_radar(cal, analyses, outdir) -> str:
    labels = [r"Response time" "\n" r"$1/\kappa$ (s)",
              r"Critical Spacing" "\n" r"$S_c$ (m)",
              r"Transition" "\n" r"width $w$ (m)",
              "Median gap\n(m)",
              r"Peak decel" "\n" r"$|a|$ (m/s$^2$)"]
    raw: Dict[int, np.ndarray] = {}
    for pos in ORDER:
        r = cal[cal["follower_position"] == pos].iloc[0]
        pa = analyses[pos]
        raw[pos] = np.array([
            1.0 / float(r.kappa),
            float(r.s_c),
            float(r.w),
            float(np.nanmedian(pa.pair.s)),
            float(pa.summary["peak_decel_mag"]),
        ], float)
    M = np.vstack([raw[p] for p in ORDER])
    maxv = np.nanmax(M, axis=0)
    norm = M / maxv

    N = len(labels)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])

    fig = plt.figure(figsize=(8.2, 7.6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for i, pos in enumerate(ORDER):
        col = MAKE_COL[POS_MAKE[pos]]
        vv = np.concatenate([norm[i], norm[i, :1]])
        ax.plot(ang_c, vv, color=col, lw=2.1, label=POS_MAKE[pos], zorder=4)
        ax.fill(ang_c, vv, color=col, alpha=0.11, zorder=3)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="0.45", fontsize=8)
    ax.set_rlabel_position(0.0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10),
              frameon=True, fontsize=9.5)
    ax.set_title("Controller heterogeneity (segment-anchored OVM)\n"
                 "per-axis normalised to across-make maximum", fontsize=12, pad=22)
    fig.text(0.5, 0.005,
             r"Mixed-polarity axes (large $1/\kappa$ = slower response, large peak "
             "decel = harsher); polygon area is not a single aggressiveness "
             "scalar. Single run, positions confounded \u2192 descriptive.",
             ha="center", va="bottom", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(outdir, "fig5_7_controller_radar.png")
    fig.savefig(out)
    plt.close(fig)

    print("[radar] raw values (1/kappa, s_c, w, median_gap, peak_decel):")
    for i, pos in enumerate(ORDER):
        vals = "  ".join(f"{x:7.3f}" for x in raw[pos])
        print(f"    {POS_MAKE[pos]:16s}  {vals}")
    print(f"[fig] wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# Concept 3: stimulus-response temporal alignment
# --------------------------------------------------------------------------- #
def busiest_window(analyses, span, step=5.0) -> Tuple[float, float]:
    """Platoon-wide busiest window: max summed leader critical points over all
    pairs within a sliding `span`-second window."""
    t = np.asarray(analyses[ORDER[0]].pair.t, float)
    onsets = []
    for pa in analyses.values():
        cps = np.asarray(pa.seg_leader.critical_points, int)
        if cps.size:
            onsets.append(np.asarray(pa.pair.t, float)[cps])
    onsets = np.concatenate(onsets) if onsets else np.array([])
    if onsets.size == 0 or (t[-1] - t[0]) <= span:
        return float(t[0]), float(min(t[0] + span, t[-1]))
    best, best_c = (float(t[0]), float(t[0] + span)), -1
    for a in np.arange(t[0], t[-1] - span, step):
        c = int(np.sum((onsets >= a) & (onsets <= a + span)))
        if c > best_c:
            best_c, best = c, (float(a), float(a + span))
    return best


def busiest_responder(analyses, t0, t1) -> int:
    """Follower with the most segment boundaries (critical points) in the window."""
    best, best_c = ORDER[0], -1
    for pos in ORDER:
        t = np.asarray(analyses[pos].pair.t, float)
        cps = np.asarray(analyses[pos].seg_follower.critical_points, int)
        ct = t[cps] if cps.size else np.array([])
        c = int(np.sum((ct >= t0) & (ct <= t1)))
        if c > best_c:
            best_c, best = c, pos
    return best


def _shade_regimes(ax, t, regimes, t0, t1, alpha=0.08):
    for (i0, i1, reg) in regimes:
        ta, tb = float(t[i0]), float(t[i1])
        if tb < t0 or ta > t1:
            continue
        c = acb.C_ACCEL if reg == "accel" else acb.C_DECEL if reg == "decel" else None
        if c is None:
            continue
        ax.axvspan(max(ta, t0), min(tb, t1), color=c, alpha=alpha, lw=0, zorder=0)


def figure_stimulus_response(cal, analyses, pos, t0, t1, outdir) -> str:
    make = POS_MAKE[pos]
    col = MAKE_COL[make]
    pa = analyses[pos]
    t = np.asarray(pa.pair.t, float)
    vL = np.asarray(pa.pair.v_leader, float)
    vF = np.asarray(pa.pair.v_follower, float)
    s = np.asarray(pa.pair.s, float)
    a_f = np.asarray(pa.a_follower, float)
    th = theta_for(cal, pos)
    tau = TAU_STAR_FIG54[make]

    a_model = ovm_accel_series(s, vF, th)
    m = (t >= t0) & (t <= t1)
    tw = t[m]
    dv = vL - vF                                    # +ve = leader faster (opening)
    a_model_del = np.interp(tw - tau, t, a_model)   # OVM delayed by tau*

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(11.6, 7.6), sharex=True)

    # ---- top: stimulus ---------------------------------------------------- #
    _shade_regimes(axT, t, pa.regimes, t0, t1)
    l1, = axT.plot(tw, dv[m], color="#7030A0", lw=1.8,
                   label=r"$\Delta v = v_{\mathrm{lead}}-v_{\mathrm{foll}}$")
    axT.axhline(0, color="0.6", lw=0.8, ls=":")
    axT.set_ylabel(r"$\Delta v$ (m/s)", color="#7030A0")
    axT.tick_params(axis="y", labelcolor="#7030A0")
    axTr = axT.twinx()
    axTr.grid(False)
    l2, = axTr.plot(tw, s[m], color="#1f4e79", lw=1.8, label="net spacing $s$")
    axTr.set_ylabel("net spacing $s$ (m)", color="#1f4e79")
    axTr.tick_params(axis="y", labelcolor="#1f4e79")
    axT.set_title("(a) stimulus \u2014 relative speed and net spacing", loc="left")
    axT.legend(handles=[l1, l2], loc="upper right", framealpha=0.9, fontsize=9)

    # ---- bottom: response ------------------------------------------------- #
    _shade_regimes(axB, t, pa.regimes, t0, t1)
    axB.plot(tw, a_f[m], color="k", lw=1.8, label="observed $a_f(t)$")
    axB.plot(tw, a_model_del, color=col, lw=1.9, ls="--",
             label=rf"segment-OVM, delayed by $\tau^*={tau:.2f}$ s")
    axB.axhline(0, color="0.6", lw=0.8, ls=":")
    axB.set_ylabel(r"follower accel (m/s$^2$)")
    axB.set_xlabel("time (s)")
    axB.set_title("(b) response \u2014 observed vs segment-anchored OVM (delayed)",
                  loc="left")
    seg_handles = [Patch(color=acb.C_ACCEL, alpha=0.30, label="accel segment"),
                   Patch(color=acb.C_DECEL, alpha=0.30, label="decel segment")]
    h, _ = axB.get_legend_handles_labels()
    axB.legend(handles=h + seg_handles, loc="upper right",
               framealpha=0.9, fontsize=8.5)
    axB.set_xlim(t0, t1)

    fig.suptitle(f"Stimulus\u2013response alignment \u2014 {make}   "
                 f"[{t0:.0f}\u2013{t1:.0f} s window]", y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    tag = make.replace(" ", "").replace("-", "")
    out = os.path.join(outdir, f"fig5_8_stimulus_response_{tag}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {out}  (tau*={tau:.2f}s)")
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="OpenACC AstaZero platoon CSV.")
    p.add_argument("--calib", default=None, help="calib_summary.csv path.")
    p.add_argument("--outdir", default=".", help="Output directory.")
    p.add_argument("--span", type=float, default=120.0,
                   help="Concept-3 window width (s).")
    p.add_argument("--t0", type=float, default=None, help="Manual window start (s).")
    p.add_argument("--t1", type=float, default=None, help="Manual window end (s).")
    p.add_argument("--make", default=None,
                   help="Concept-3 make (e.g. 'BMW X5'); default = busiest responder.")
    p.add_argument("--all-makes", action="store_true",
                   help="Emit a Concept-3 figure for every make.")
    p.add_argument("--smoke", action="store_true",
                   help="Render Concept 1 only; print Concept 2/3 setup numbers.")
    return p


def _resolve_make(arg_make) -> Optional[int]:
    if arg_make is None:
        return None
    for pos, mk in POS_MAKE.items():
        if mk.lower() == arg_make.strip().lower():
            return pos
    sys.exit(f"error: --make '{arg_make}' not in {list(POS_MAKE.values())}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    csv = find_csv(args.csv)
    calib = find_calib(args.calib)

    cal = load_calib(calib)
    pairs, info = acb.load_openacc_asta(csv)
    by_pos = {p.follower_veh: p for p in pairs}
    missing = [p for p in ORDER if p not in by_pos]
    if missing:
        sys.exit(f"error: platoon missing follower positions {missing}")

    aargs = make_analysis_args()
    analyses = {pos: acb.analyze_pair(by_pos[pos], aargs) for pos in ORDER}

    print(f"[load] {os.path.basename(csv)}  dt={info['dt']:.3f}s  "
          f"samples={info['n_samples']}  gap-setting={info['distance_setting']}")

    manual = args.t0 is not None and args.t1 is not None
    win = (float(args.t0), float(args.t1)) if manual else busiest_window(analyses, args.span)
    resp = busiest_responder(analyses, win[0], win[1])
    print(f"[window] Concept-3 window = {win[0]:.0f}-{win[1]:.0f} s"
          f"{'' if manual else ' (platoon-wide busiest)'}; "
          f"busiest responder = {POS_MAKE[resp]}")

    if args.smoke:
        f1 = figure_speed_spacing_overlay(cal, analyses, args.outdir)
        print("[smoke] per-make S_c / median-gap / peak-decel / tau*:")
        for pos in ORDER:
            th = theta_for(cal, pos)
            pa = analyses[pos]
            print(f"    {POS_MAKE[pos]:16s}  S_c={th[2]:5.2f} m  "
                  f"median_gap={np.nanmedian(pa.pair.s):5.2f} m  "
                  f"peak_decel={pa.summary['peak_decel_mag']:.2f} m/s^2  "
                  f"tau*={TAU_STAR_FIG54[POS_MAKE[pos]]:.2f} s")
        print("[smoke] rendered Concept 1 only; Concept 2/3 skipped.")
        print("[done]", f1)
        return

    outs = [figure_speed_spacing_overlay(cal, analyses, args.outdir),
            figure_controller_radar(cal, analyses, args.outdir)]
    if args.all_makes:
        for pos in ORDER:
            outs.append(figure_stimulus_response(cal, analyses, pos, win[0], win[1], args.outdir))
    else:
        pos = _resolve_make(args.make) or resp
        outs.append(figure_stimulus_response(cal, analyses, pos, win[0], win[1], args.outdir))

    print(f"\n[done] wrote {len(outs)} figures to {args.outdir}")
    for o in outs:
        print("   ", o)


if __name__ == "__main__":
    main()
