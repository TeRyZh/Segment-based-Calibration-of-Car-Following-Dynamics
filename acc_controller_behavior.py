#!/usr/bin/env python3
"""
acc_controller_behavior.py
==========================
High-level ACC controller behaviour characterisation for the OpenACC AstaZero
platoon (`ASta_040719_platoon7.csv`), built on the project's critical-point /
phase framework.

It reuses `phase_segmentation.segment_trajectory` VERBATIM to place critical
points (accel<->decel switches) on both the leader and each follower, then adds
a thin *controller-analysis* layer that:

  1. Stable phase       -- baseline follower acceleration when the leader holds
                           near-constant speed (smooth-following signature).
  2. Perturbation + lag -- leader critical points mark perturbation onsets;
                           response lag is estimated two ways:
                             (a) event-based: matched leader->follower critical
                                 points (order-preserving, same-kind, nearest-after);
                             (b) signal-based: argmax_tau corr(dv, a_f(t+tau)).
  3. Peak extraction    -- max acceleration / max deceleration over the accel /
                           decel regimes, contextualised against literature
                           human comfort bands.

Centre-piece figure: the response-lag visualisation, which draws connectors
between each matched leader decel/accel critical point and the follower's
responding critical point; the horizontal span of a connector *is* the lag.

IMPORTANT
---------
* All AstaZero channels are SI (speed m/s, spacing IVS m, ENU m), verified.
* Segmentation therefore runs with the *SI* CUSUM thresholds (2.1 / 0.3),
  NOT the ft/s NGSIM defaults (7.0 / 1.0).
* This file has no acceleration channel, so a_f is DERIVED from speed via a
  Savitzky-Golay derivative. That sits against the manuscript's
  differentiation-noise critique, so it is kept strictly descriptive and its
  peak sensitivity is reported across smoothing windows (see --sg-windows).
* Single run, one vehicle per make, platoon position confounded -> every
  cross-vehicle number is a per-follower-position DESCRIPTIVE characterisation,
  NOT manufacturer inference; windowed sub-segments are not independent samples.

CF pairs are consecutive predecessor->follower: 1->2, 2->3, 3->4, 4->5.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# NumPy 2.x removed np.trapz (renamed np.trapezoid). phase_segmentation.py still
# calls np.trapz; alias it here so that read-only module runs unmodified.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid            # type: ignore[attr-defined]

try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:                                   # pragma: no cover
    _HAVE_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

# Font sizing. Defaults are already enlarged vs. matplotlib; scale everything
# (titles, labels, ticks, legends, in-figure lag tags) with --font-scale when
# stitching many panels together. FONT is referenced by the figure functions.
FONT: Dict[str, float] = {}


def set_font_scale(scale: float = 1.0) -> None:
    base = dict(title=14.0, label=13.0, tick=12.0, legend=11.0,
                legend_title=11.0, tag=9.0)
    for k, v in base.items():
        FONT[k] = v * scale
    plt.rcParams.update({
        "font.size": FONT["label"],
        "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"],
        "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"],
        "legend.fontsize": FONT["legend"],
        "legend.title_fontsize": FONT["legend_title"],
    })


set_font_scale(1.0)

from phase_segmentation import segment_trajectory, SegmentationResult  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SI_CUSUM_THRESH = 2.1          # m/s  (SI counterpart of NGSIM 7.0 ft/s)
SI_CUSUM_DRIFT = 0.3           # m/s  (SI counterpart of NGSIM 1.0 ft/s)

# Literature human comfort reference bands (m/s^2). These are REFERENCE values
# for context only -- widely cited comfort/harsh thresholds, not data from this
# run. Comfortable |a| ~ 2 m/s^2; harsh / uncomfortable > ~3.5 m/s^2.
COMFORT_ACCEL = 2.0
COMFORT_DECEL = 2.5
HARSH_LEVEL = 3.5

# Plot colours
C_LEAD = "#1f4e79"
C_FOLL = "#c55a11"
C_DECEL = "#c0392b"
C_ACCEL = "#1e8449"
C_EQUIL = "#7f8c8d"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class PairData:
    """One predecessor->follower pair extracted from the platoon."""
    pair_idx: int
    leader_veh: int
    follower_veh: int
    leader_name: str
    follower_name: str
    t: np.ndarray
    v_leader: np.ndarray
    v_follower: np.ndarray
    s: np.ndarray                  # net spacing IVS_{leader} (m)
    dt: float


def _clean_name(raw: str) -> str:
    """'Tesla(Model3)'->'Tesla Model 3'; 'Audi(A8)'->'Audi A8';
    'Mercedes(AClass)'->'Mercedes A-Class'; 'BMW(X5)'->'BMW X5'."""
    import re
    raw = raw.strip()
    if "(" in raw and raw.endswith(")"):
        make, model = raw[:-1].split("(", 1)
        model = re.sub(r"(?<=[a-z])(?=\d)", " ", model)        # Model3 -> Model 3
        model = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", model)  # AClass -> A-Class
        return f"{make.strip()} {model.strip()}"
    return raw


def load_openacc_asta(path: str) -> Tuple[List[PairData], Dict[str, object]]:
    """Parse an OpenACC AstaZero platoon CSV into consecutive CF pairs.

    Preamble (first 5 rows) carries Vehicle_order / Number_of_vehicles / ACC /
    Distance_setting; row 6 is the wide header (per-vehicle Speed/Lat/Lon/Alt/
    E/N/U, then Driver1..N, then IVS1..N-1).
    """
    # --- preamble ---------------------------------------------------------- #
    with open(path, "r") as fh:
        head = [next(fh) for _ in range(5)]
    meta: Dict[str, object] = {}
    order: List[str] = []
    for line in head:
        cells = [c.strip() for c in line.rstrip("\n").rstrip("\r").split(",")]
        key = cells[0]
        vals = [c for c in cells[1:] if c != ""]
        if key == "Vehicle_order":
            order = [_clean_name(v) for v in vals]
        meta[key] = vals

    df = pd.read_csv(path, skiprows=5)
    df.columns = [c.strip() for c in df.columns]

    t = df["Time"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    n_veh = len(order) if order else int(meta.get("Number_of_vehicles", ["0"])[0])

    # Speeds per vehicle. Guard: confirm SI (m/s) against ENU-implied speed.
    speeds = {i: df[f"Speed{i}"].to_numpy(float) for i in range(1, n_veh + 1)}
    try:
        E = df["E1"].to_numpy(float); N = df["N1"].to_numpy(float)
        lo, hi = len(t) // 3, min(len(t) // 3 + 300, len(t))
        v_enu = np.hypot(np.diff(E[lo:hi]), np.diff(N[lo:hi])) / dt
        ratio = np.median(speeds[1][lo:hi]) / max(np.median(v_enu), 1e-6)
        if not (0.5 < ratio < 2.0):
            print(f"[warn] Speed/ENU ratio={ratio:.2f}; speed units may not be m/s.")
    except Exception:
        pass

    # IVS_i is spacing between veh i and veh i+1.
    ivs = {i: df[f"IVS{i}"].to_numpy(float) for i in range(1, n_veh)}

    pairs: List[PairData] = []
    for i in range(1, n_veh):                        # leader=i, follower=i+1
        pairs.append(PairData(
            pair_idx=i - 1, leader_veh=i, follower_veh=i + 1,
            leader_name=order[i - 1] if order else f"veh{i}",
            follower_name=order[i] if order else f"veh{i+1}",
            t=t, v_leader=speeds[i], v_follower=speeds[i + 1],
            s=ivs[i], dt=dt))

    info = {"n_veh": n_veh, "order": order, "dt": dt, "n_samples": len(t),
            "distance_setting": meta.get("Distance_setting", [""])[0]}
    return pairs, info


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #
def _odd(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def derive_accel(v: np.ndarray, dt: float, window: int = 15, poly: int = 2) -> np.ndarray:
    """Follower acceleration via Savitzky-Golay derivative (falls back to
    central differences if SciPy is unavailable or the window is too small)."""
    n = len(v)
    w = _odd(min(window, n if n % 2 == 1 else n - 1))
    if (not _HAVE_SCIPY) or w <= poly or w < 3:
        return np.gradient(v, dt)
    return savgol_filter(v, w, poly, deriv=1, delta=dt, mode="interp")


def cumdist(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Distance-travelled channel for PELT+ (trapezoid integral of speed)."""
    seg = 0.5 * (v[1:] + v[:-1]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(seg)])


def segment(t: np.ndarray, v: np.ndarray, s: np.ndarray, args) -> SegmentationResult:
    x = cumdist(t, v)
    return segment_trajectory(
        t, x, v, s,
        penalty=args.penalty, min_segment_length=args.min_seg,
        cusum_threshold=args.cusum_thresh, cusum_drift=args.cusum_drift)


# --------------------------------------------------------------------------- #
# Controller-analysis primitives
# --------------------------------------------------------------------------- #
def classify_regimes(seg: SegmentationResult, v: np.ndarray,
                     deadband: float) -> List[Tuple[int, int, str]]:
    """Label each phase accel/decel/equil by net dv over the phase (deadband)."""
    out = []
    for ph in seg.phases:
        dv = v[ph.i_end] - v[ph.i_start]
        reg = "accel" if dv > deadband else "decel" if dv < -deadband else "equil"
        out.append((ph.i_start, ph.i_end, reg))
    return out


def stable_windows(a_leader: np.ndarray, dt: float,
                   eps: float, tmin: float) -> List[Tuple[int, int]]:
    """Contiguous runs where |leader accel| < eps for at least tmin seconds."""
    mask = np.abs(a_leader) < eps
    minlen = int(round(tmin / dt))
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= minlen:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def stable_stats(a_follower: np.ndarray,
                 windows: List[Tuple[int, int]], n_total: int) -> Dict[str, float]:
    if not windows:
        return dict(stable_a_mean=np.nan, stable_a_std=np.nan, stable_a_rms=np.nan,
                    stable_a_absmax=np.nan, stable_frac=0.0, stable_n_win=0)
    vals = np.concatenate([a_follower[i0:i1 + 1] for i0, i1 in windows])
    cov = sum(i1 - i0 + 1 for i0, i1 in windows) / max(n_total, 1)
    return dict(
        stable_a_mean=float(vals.mean()),
        stable_a_std=float(vals.std()),
        stable_a_rms=float(np.sqrt(np.mean(vals ** 2))),
        stable_a_absmax=float(np.max(np.abs(vals))),
        stable_frac=float(cov),
        stable_n_win=len(windows))


def match_same_kind(L: Sequence[int], F: Sequence[int], t: np.ndarray,
                    tau_max: float) -> List[Tuple[int, int, float]]:
    """Order-preserving, one-to-one, nearest-after match of leader points L to
    follower points F of the SAME kind, within tau_max seconds. Returns
    (i_leader, i_follower, lag_seconds)."""
    L = sorted(L); F = sorted(F)
    i = j = 0
    out: List[Tuple[int, int, float]] = []
    while i < len(L) and j < len(F):
        if F[j] <= L[i]:                    # follower not strictly after leader
            j += 1
        elif (t[F[j]] - t[L[i]]) > tau_max:  # nearest follower too far -> skip leader
            i += 1
        else:
            out.append((L[i], F[j], float(t[F[j]] - t[L[i]])))
            i += 1
            j += 1
    return out


def xcorr_lag(dv: np.ndarray, a: np.ndarray, dt: float,
              tau_max: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """argmax over tau in [0, tau_max] of corr(dv[t], a[t+tau]). Returns
    (tau*, r*, lags, r_curve)."""
    kmax = int(round(tau_max / dt))
    lags = np.arange(0, kmax + 1) * dt
    rs = np.full(kmax + 1, np.nan)
    for k in range(kmax + 1):
        x = dv if k == 0 else dv[:-k]
        y = a if k == 0 else a[k:]
        if len(x) >= 10 and np.std(x) > 1e-9 and np.std(y) > 1e-9:
            rs[k] = np.corrcoef(x, y)[0, 1]
    if np.all(~np.isfinite(rs)):
        return 0.0, np.nan, lags, rs
    kbest = int(np.nanargmax(rs))
    return float(lags[kbest]), float(rs[kbest]), lags, rs


def extract_peaks(a: np.ndarray,
                  regimes: List[Tuple[int, int, str]]) -> Dict[str, float]:
    """Peak accel over accel regimes, peak decel over decel regimes, plus the
    per-phase peak distributions and unconditional global extremes."""
    accel_peaks = [float(np.max(a[i0:i1 + 1])) for i0, i1, r in regimes if r == "accel"]
    decel_peaks = [float(np.min(a[i0:i1 + 1])) for i0, i1, r in regimes if r == "decel"]
    return dict(
        peak_accel=max(accel_peaks) if accel_peaks else np.nan,
        peak_decel=min(decel_peaks) if decel_peaks else np.nan,           # signed (<0)
        median_accel_peak=float(np.median(accel_peaks)) if accel_peaks else np.nan,
        median_decel_peak=float(np.median(decel_peaks)) if decel_peaks else np.nan,
        n_accel_phase=len(accel_peaks), n_decel_phase=len(decel_peaks),
        global_a_max=float(np.max(a)), global_a_min=float(np.min(a)))


# --------------------------------------------------------------------------- #
# Per-pair analysis
# --------------------------------------------------------------------------- #
@dataclass
class PairAnalysis:
    pair: PairData
    seg_leader: SegmentationResult
    seg_follower: SegmentationResult
    a_follower: np.ndarray
    a_leader: np.ndarray
    regimes: List[Tuple[int, int, str]]
    decel_matches: List[Tuple[int, int, float]]
    accel_matches: List[Tuple[int, int, float]]
    xcorr: Tuple[float, float, np.ndarray, np.ndarray]
    summary: Dict[str, object]
    events: List[Dict[str, object]] = field(default_factory=list)


def analyze_pair(pair: PairData, args) -> PairAnalysis:
    t, dt = pair.t, pair.dt
    a_f = derive_accel(pair.v_follower, dt, args.sg_window, args.sg_poly)
    a_l = derive_accel(pair.v_leader, dt, args.sg_window, args.sg_poly)

    seg_l = segment(t, pair.v_leader, pair.s, args)
    seg_f = segment(t, pair.v_follower, pair.s, args)

    regimes = classify_regimes(seg_f, pair.v_follower, args.deadband)

    # Event-based lag: match leader critical points to follower responses.
    decel_m = match_same_kind(seg_l.decel_points, seg_f.decel_points, t, args.tau_max)
    accel_m = match_same_kind(seg_l.accel_points, seg_f.accel_points, t, args.tau_max)

    # Signal-based lag: dv vs follower acceleration.
    dv = pair.v_leader - pair.v_follower
    xc = xcorr_lag(dv, a_f, dt, args.tau_max)

    # Stable-phase follower behaviour (leader near-constant).
    win = stable_windows(a_l, dt, args.eps_stable, args.tmin)
    stab = stable_stats(a_f, win, len(t))

    peaks = extract_peaks(a_f, regimes)

    decel_lags = np.array([m[2] for m in decel_m]) if decel_m else np.array([])
    accel_lags = np.array([m[2] for m in accel_m]) if accel_m else np.array([])
    all_lags = np.concatenate([decel_lags, accel_lags]) if (len(decel_m) + len(accel_m)) else np.array([])

    summary = dict(
        pair_idx=pair.pair_idx,
        follower=pair.follower_name, follower_veh=pair.follower_veh,
        leader=pair.leader_name, leader_veh=pair.leader_veh,
        n_phases=seg_f.n_phases,
        n_crit_leader=len(seg_l.critical_points),
        n_crit_follower=len(seg_f.critical_points),
        n_decel_events=len(decel_m), n_accel_events=len(accel_m),
        lag_decel_mean=float(decel_lags.mean()) if decel_lags.size else np.nan,
        lag_decel_median=float(np.median(decel_lags)) if decel_lags.size else np.nan,
        lag_accel_mean=float(accel_lags.mean()) if accel_lags.size else np.nan,
        lag_accel_median=float(np.median(accel_lags)) if accel_lags.size else np.nan,
        lag_all_mean=float(all_lags.mean()) if all_lags.size else np.nan,
        lag_all_median=float(np.median(all_lags)) if all_lags.size else np.nan,
        xcorr_lag=xc[0], xcorr_r=xc[1],
        peak_accel=peaks["peak_accel"],
        peak_decel_mag=abs(peaks["peak_decel"]) if np.isfinite(peaks["peak_decel"]) else np.nan,
        median_accel_peak=peaks["median_accel_peak"],
        median_decel_peak_mag=abs(peaks["median_decel_peak"]) if np.isfinite(peaks["median_decel_peak"]) else np.nan,
        global_a_max=peaks["global_a_max"], global_a_min=peaks["global_a_min"],
        **stab,
    )

    events: List[Dict[str, object]] = []
    for (iL, iF, lag) in decel_m:
        events.append(dict(follower=pair.follower_name, kind="decel",
                           t_leader_onset=float(t[iL]), t_follower_resp=float(t[iF]),
                           lag_s=lag, follower_a_at_resp=float(a_f[iF])))
    for (iL, iF, lag) in accel_m:
        events.append(dict(follower=pair.follower_name, kind="accel",
                           t_leader_onset=float(t[iL]), t_follower_resp=float(t[iF]),
                           lag_s=lag, follower_a_at_resp=float(a_f[iF])))
    events.sort(key=lambda e: e["t_leader_onset"])

    return PairAnalysis(pair, seg_l, seg_f, a_f, a_l, regimes,
                        decel_m, accel_m, xc, summary, events)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _densest_window(onsets: np.ndarray, t: np.ndarray, span: float) -> Tuple[float, float]:
    if onsets.size == 0:
        return float(t[0]), float(min(t[0] + span, t[-1]))
    best, best_c = (t[0], t[0] + span), -1
    for t0 in np.arange(t[0], max(t[-1] - span, t[0] + 1), 5.0):
        c = int(np.sum((onsets >= t0) & (onsets <= t0 + span)))
        if c > best_c:
            best_c, best = c, (float(t0), float(t0 + span))
    return best


def fig_lag_connectors(pa: PairAnalysis, out_path: str,
                       t0: Optional[float] = None, t1: Optional[float] = None,
                       span: float = 55.0) -> str:
    """CENTRE-PIECE: leader & follower speed with decel/accel critical points,
    and connectors drawn between each matched leader->follower critical point.
    The horizontal extent of every connector is the response lag."""
    p = pa.pair
    t, vL, vF = p.t, p.v_leader, p.v_follower
    onsets = np.array([m[0] for m in (pa.decel_matches + pa.accel_matches)], float)
    onsets_t = t[onsets.astype(int)] if onsets.size else np.array([])
    if t0 is None or t1 is None:
        t0, t1 = _densest_window(onsets_t, t, span)
    m = (t >= t0) & (t <= t1)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(t[m], vL[m], color=C_LEAD, lw=1.8, label=f"leader speed ({p.leader_name})")
    ax.plot(t[m], vF[m], color=C_FOLL, lw=1.8, label=f"follower speed ({p.follower_name})")

    def _in(idx):
        return [i for i in idx if t0 <= t[i] <= t1]

    Ld, La = _in(pa.seg_leader.decel_points), _in(pa.seg_leader.accel_points)
    Fd, Fa = _in(pa.seg_follower.decel_points), _in(pa.seg_follower.accel_points)
    ax.scatter(t[Ld], vL[Ld], marker="v", s=70, color=C_DECEL, zorder=5,
               edgecolor="white", linewidth=0.6, label="leader decel point")
    ax.scatter(t[La], vL[La], marker="^", s=70, color=C_ACCEL, zorder=5,
               edgecolor="white", linewidth=0.6, label="leader accel point")
    ax.scatter(t[Fd], vF[Fd], marker="v", s=55, facecolor="none",
               edgecolor=C_DECEL, linewidth=1.6, zorder=5, label="follower decel point")
    ax.scatter(t[Fa], vF[Fa], marker="^", s=55, facecolor="none",
               edgecolor=C_ACCEL, linewidth=1.6, zorder=5, label="follower accel point")

    # Connectors (the lag). Colour by kind; annotate lag at midpoint.
    def _connect(matches, col):
        for iL, iF, lag in matches:
            if not (t0 <= t[iL] <= t1 and t0 <= t[iF] <= t1):
                continue
            ax.annotate("", xy=(t[iF], vF[iF]), xytext=(t[iL], vL[iL]),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.3, alpha=0.8, shrinkA=3, shrinkB=3))
            ax.text(0.5 * (t[iL] + t[iF]), 0.5 * (vL[iL] + vF[iF]),
                    f"{lag:.1f}s", fontsize=FONT["tag"], color=col, ha="center",
                    va="center", bbox=dict(boxstyle="round,pad=0.15",
                                           fc="white", ec=col, alpha=0.85))
    _connect(pa.decel_matches, C_DECEL)
    _connect(pa.accel_matches, C_ACCEL)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title(f"Response-time lag:  {p.leader_name} \u2192 {p.follower_name}",
                 wrap=True)
    ax.legend(loc="best", fontsize=FONT["legend"], ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _resolve_vref(detrend, v_window: np.ndarray) -> Optional[float]:
    if detrend is None or str(detrend).lower() == "none":
        return None
    if str(detrend).lower() == "auto":
        return float(np.mean(v_window))
    return float(detrend)


def fig_lag_connectors_distance(pa: PairAnalysis, out_path: str,
                                t0: Optional[float] = None, t1: Optional[float] = None,
                                span: float = 55.0, detrend="auto") -> str:
    """Response-lag on the space-time (distance) trajectory -- the channel PELT+
    actually segments. Leader position is the integral of its speed; the
    follower is anchored one spacing behind (x_F = x_L - s), which reproduces
    the follower's own speed by construction so its critical points sit on the
    curve. Plotted in oblique coordinates x - v_ref*t (default v_ref = window
    mean leader speed) so the accel/decel bends, the gap, and the lag are all
    legible; connectors join matched leader->follower critical points and their
    horizontal span is the response lag."""
    p = pa.pair
    t, vL, s = p.t, p.v_leader, p.s
    xL = cumdist(t, vL)
    xF = xL - s                                    # IVS-anchored follower position

    onsets = np.array([m[0] for m in (pa.decel_matches + pa.accel_matches)], float)
    onsets_t = t[onsets.astype(int)] if onsets.size else np.array([])
    if t0 is None or t1 is None:
        t0, t1 = _densest_window(onsets_t, t, span)
    m = (t >= t0) & (t <= t1)
    idx0 = int(np.argmax(m))                        # first index inside window

    v_ref = _resolve_vref(detrend, vL[m])
    base = xL[idx0]
    if v_ref is None:
        yL, yF = xL - base, xF - base
        ylab, vref_txt = "distance travelled (m)", "raw"
    else:
        yL = (xL - base) - v_ref * (t - t[idx0])
        yF = (xF - base) - v_ref * (t - t[idx0])
        ylab = r"$x - v_{\mathrm{ref}}\,t$  (m, oblique)"
        vref_txt = rf"$v_{{\mathrm{{ref}}}}$={v_ref:.1f} m/s"

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(t[m], yL[m], color=C_LEAD, lw=1.8, label=f"leader ({p.leader_name})")
    ax.plot(t[m], yF[m], color=C_FOLL, lw=1.8, label=f"follower ({p.follower_name})")

    def _in(idx):
        return [i for i in idx if t0 <= t[i] <= t1]
    Ld, La = _in(pa.seg_leader.decel_points), _in(pa.seg_leader.accel_points)
    Fd, Fa = _in(pa.seg_follower.decel_points), _in(pa.seg_follower.accel_points)
    ax.scatter(t[Ld], yL[Ld], marker="v", s=70, color=C_DECEL, zorder=5,
               edgecolor="white", linewidth=0.6, label="leader decel point")
    ax.scatter(t[La], yL[La], marker="^", s=70, color=C_ACCEL, zorder=5,
               edgecolor="white", linewidth=0.6, label="leader accel point")
    ax.scatter(t[Fd], yF[Fd], marker="v", s=55, facecolor="none",
               edgecolor=C_DECEL, linewidth=1.6, zorder=5, label="follower decel point")
    ax.scatter(t[Fa], yF[Fa], marker="^", s=55, facecolor="none",
               edgecolor=C_ACCEL, linewidth=1.6, zorder=5, label="follower accel point")

    def _connect(matches, col):
        for iL, iF, lag in matches:
            if not (t0 <= t[iL] <= t1 and t0 <= t[iF] <= t1):
                continue
            ax.annotate("", xy=(t[iF], yF[iF]), xytext=(t[iL], yL[iL]),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.3, alpha=0.8, shrinkA=3, shrinkB=3))
            ax.text(0.5 * (t[iL] + t[iF]), 0.5 * (yL[iL] + yF[iF]),
                    f"{lag:.1f}s", fontsize=FONT["tag"], color=col, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, alpha=0.85))
    _connect(pa.decel_matches, C_DECEL)
    _connect(pa.accel_matches, C_ACCEL)

    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylab)
    ax.set_title(f"Space-time lag:  {p.leader_name} \u2192 {p.follower_name}  "
                 f"({vref_txt})", wrap=True)
    ax.legend(loc="best", fontsize=FONT["legend"], ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def fig_timeseries_phases(pa: PairAnalysis, out_path: str,
                          t0: Optional[float] = None, t1: Optional[float] = None,
                          span: float = 90.0) -> str:
    """Three stacked panels: speeds+critical points; follower accel with regime
    shading; approach rate dv."""
    p = pa.pair
    t = p.t
    onsets = np.array([m[0] for m in (pa.decel_matches + pa.accel_matches)], float)
    onsets_t = t[onsets.astype(int)] if onsets.size else np.array([])
    if t0 is None or t1 is None:
        t0, t1 = _densest_window(onsets_t, t, span)
    m = (t >= t0) & (t <= t1)

    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].plot(t[m], p.v_leader[m], color=C_LEAD, lw=1.6, label="leader")
    ax[0].plot(t[m], p.v_follower[m], color=C_FOLL, lw=1.6, label="follower")
    for i in [i for i in pa.seg_follower.decel_points if t0 <= t[i] <= t1]:
        ax[0].scatter(t[i], p.v_follower[i], marker="v", color=C_DECEL, s=35, zorder=5)
    for i in [i for i in pa.seg_follower.accel_points if t0 <= t[i] <= t1]:
        ax[0].scatter(t[i], p.v_follower[i], marker="^", color=C_ACCEL, s=35, zorder=5)
    ax[0].set_ylabel("speed (m/s)"); ax[0].legend(fontsize=FONT["legend"]); ax[0].grid(alpha=0.25)
    ax[0].set_title(f"{p.leader_name} \u2192 {p.follower_name}:  phase decomposition",
                    wrap=True)

    ax[1].plot(t[m], pa.a_follower[m], color="#34495e", lw=1.3)
    ax[1].axhline(0, color="k", lw=0.6)
    for i0, i1, reg in pa.regimes:
        if t[i1] < t0 or t[i0] > t1 or reg == "equil":
            continue
        col = C_ACCEL if reg == "accel" else C_DECEL
        ax[1].axvspan(max(t[i0], t0), min(t[i1], t1), color=col, alpha=0.12)
    for lvl, ls in [(COMFORT_DECEL, ":"), (-COMFORT_DECEL, ":")]:
        ax[1].axhline(lvl, color="#7f8c8d", ls=ls, lw=0.9)
    ax[1].set_ylabel("follower accel (m/s$^2$)"); ax[1].grid(alpha=0.25)

    ax[2].plot(t[m], (p.v_leader - p.v_follower)[m], color="#8e44ad", lw=1.3)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set_ylabel(r"$\Delta v = v_L - v_F$ (m/s)")
    ax[2].set_xlabel("time (s)"); ax[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def fig_peaks(analyses: List[PairAnalysis], out_path: str) -> str:
    """Peak accel and peak decel magnitude per follower, with human bands."""
    names = [pa.pair.follower_name for pa in analyses]
    p_acc = [pa.summary["peak_accel"] for pa in analyses]
    p_dec = [pa.summary["peak_decel_mag"] for pa in analyses]
    x = np.arange(len(names)); w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axhspan(0, COMFORT_ACCEL, color=C_ACCEL, alpha=0.06)
    ax.axhspan(COMFORT_ACCEL, HARSH_LEVEL, color="#f39c12", alpha=0.06)
    ax.axhspan(HARSH_LEVEL, max(max(p_acc), max(p_dec)) * 1.15 + 0.5,
               color=C_DECEL, alpha=0.06)
    ax.axhline(COMFORT_ACCEL, color="#7f8c8d", ls="--", lw=1,
               label=f"comfortable ~{COMFORT_ACCEL:.1f} m/s$^2$")
    ax.axhline(HARSH_LEVEL, color=C_DECEL, ls="--", lw=1,
               label=f"harsh > {HARSH_LEVEL:.1f} m/s$^2$")
    ax.bar(x - w / 2, p_acc, w, color=C_ACCEL, label="peak acceleration")
    ax.bar(x + w / 2, p_dec, w, color=C_DECEL, label="peak deceleration (|.|)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("peak |acceleration| (m/s$^2$)")
    ax.set_title("Peak accel / decel per follower  "
                 "(bands: human comfort reference)", wrap=True)
    ax.legend(fontsize=FONT["legend"]); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def fig_xcorr(pa: PairAnalysis, out_path: str) -> str:
    """Cross-correlation curve corr(dv, a_f(t+tau)) vs lag, with tau* marked."""
    tau, r, lags, rs = pa.xcorr
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lags, rs, color=C_FOLL, lw=1.8)
    if np.isfinite(r):
        ax.axvline(tau, color=C_DECEL, ls="--", lw=1.2,
                   label=f"$\\tau^*$={tau:.2f}s (r={r:.2f})")
    ax.set_xlabel(r"lag $\tau$ (s)")
    ax.set_ylabel(r"corr$(\Delta v(t),\, a_f(t+\tau))$")
    ax.set_title(f"Signal-based response lag:  {pa.pair.leader_name} \u2192 "
                 f"{pa.pair.follower_name}", wrap=True)
    ax.legend(fontsize=FONT["legend"]); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Common window across pairs + platoon propagation overlay
# --------------------------------------------------------------------------- #
# Leader -> tail sequential palette so platoon position reads off the colour.
PLATOON_COLORS = ["#1f4e79", "#2e86c1", "#17a589", "#ca6f1e", "#a93226",
                  "#6c3483", "#909497"]


def _visible_connectors(pa: PairAnalysis, t: np.ndarray,
                        t0: float, t1: float) -> int:
    """Count matched connectors whose BOTH endpoints fall inside [t0, t1]."""
    return sum(1 for (iL, iF, _) in (pa.decel_matches + pa.accel_matches)
               if t0 <= t[iL] <= t1 and t0 <= t[iF] <= t1)


def _common_window(analyses: List[PairAnalysis], t: np.ndarray,
                   span: float, step: float = 5.0) -> Tuple[float, float]:
    """One [t0, t0+span] maximising summed visible connectors across ALL pairs,
    subject to every pair contributing >=1 (guard against landing where the wave
    has damped before reaching the tail). Falls back to pure max-sum if no
    window covers all pairs."""
    lo, hi = float(t[0]), float(t[-1] - span)
    if hi <= lo:
        return float(t[0]), float(t[-1])
    best_guard: Optional[Tuple[int, float]] = None
    best_any: Optional[Tuple[int, float]] = None
    for t0 in np.arange(lo, hi, step):
        t1 = t0 + span
        counts = [_visible_connectors(pa, t, t0, t1) for pa in analyses]
        total = int(sum(counts))
        if best_any is None or total > best_any[0]:
            best_any = (total, float(t0))
        if all(c >= 1 for c in counts) and (best_guard is None or total > best_guard[0]):
            best_guard = (total, float(t0))
    chosen = best_guard if best_guard is not None else best_any
    t0 = chosen[1]
    return t0, t0 + span


def _platoon_vref(analyses: List[PairAnalysis], t: np.ndarray,
                  t0: float, t1: float) -> float:
    """Platoon-mean speed over the window (all vehicles pooled)."""
    m = (t >= t0) & (t <= t1)
    speeds = [analyses[0].pair.v_leader[m]] + [pa.pair.v_follower[m] for pa in analyses]
    return float(np.mean(np.concatenate(speeds)))


def fig_platoon_spacetime(analyses: List[PairAnalysis], out_path: str,
                          t0: float, t1: float, v_ref: float,
                          tags: bool = True, tag_min_dt: float = 2.5) -> str:
    """All vehicles on one oblique space-time panel over a shared window and a
    single v_ref, so a perturbation is seen entering at the lead vehicle and
    cascading down the platoon (staggered lags + downstream attenuation) in one
    frame. Positions are chained through the IVS spacings; connectors between
    consecutive vehicles (from each pair's matched critical points) trace the
    propagation diagonal and are tagged with the response lag (a per-band
    spacing guard, tag_min_dt, suppresses overlapping labels)."""
    t = analyses[0].pair.t
    m = (t >= t0) & (t <= t1)
    idx0 = int(np.argmax(m))

    # Chained IVS positions: X_1 = int v_1; X_{i+1} = X_i - s_i.
    X = [cumdist(t, analyses[0].pair.v_leader)]
    for pa in analyses:
        X.append(X[-1] - pa.pair.s)
    base = X[0][idx0]
    Ys = [(Xk - base) - v_ref * (t - t[idx0]) for Xk in X]

    names = [analyses[0].pair.leader_name] + [pa.pair.follower_name for pa in analyses]
    segs = [analyses[0].seg_leader] + [pa.seg_follower for pa in analyses]

    fig, ax = plt.subplots(figsize=(13, 7))
    for k, (Y, nm, sg) in enumerate(zip(Ys, names, segs)):
        col = PLATOON_COLORS[k % len(PLATOON_COLORS)]
        ax.plot(t[m], Y[m], color=col, lw=1.8, label=f"{k+1}. {nm}")
        Din = [i for i in sg.decel_points if t0 <= t[i] <= t1]
        Ain = [i for i in sg.accel_points if t0 <= t[i] <= t1]
        ax.scatter(t[Din], Y[Din], marker="v", s=34, color=col,
                   edgecolor="white", linewidth=0.5, zorder=5)
        ax.scatter(t[Ain], Y[Ain], marker="^", s=34, color=col,
                   edgecolor="white", linewidth=0.5, zorder=5)

    # Propagation connectors between consecutive vehicles, tagged with the lag.
    for i, pa in enumerate(analyses):
        YL, YF = Ys[i], Ys[i + 1]
        combined = ([(iL, iF, lag, C_DECEL) for (iL, iF, lag) in pa.decel_matches]
                    + [(iL, iF, lag, C_ACCEL) for (iL, iF, lag) in pa.accel_matches])
        combined.sort(key=lambda e: t[e[0]])
        last_tag_t = -1e9
        for (iL, iF, lag, col) in combined:
            if not (t0 <= t[iL] <= t1 and t0 <= t[iF] <= t1):
                continue
            ax.annotate("", xy=(t[iF], YF[iF]), xytext=(t[iL], YL[iL]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                        alpha=0.55, shrinkA=2, shrinkB=2))
            mid_t = 0.5 * (t[iL] + t[iF])
            if tags and (mid_t - last_tag_t) >= tag_min_dt:
                ax.text(mid_t, 0.5 * (YL[iL] + YF[iF]), f"{lag:.1f}s",
                        fontsize=FONT["tag"], color=col, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec=col, alpha=0.85), zorder=6)
                last_tag_t = mid_t

    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$x - v_{\mathrm{ref}}\,t$  (m, oblique)")
    ax.set_title(f"Platoon propagation  "
                 f"($v_{{\\mathrm{{ref}}}}$={v_ref:.1f} m/s, {t0:.0f}-{t1:.0f}s)",
                 wrap=True)
    ax.legend(loc="best", fontsize=FONT["legend"], ncol=2, framealpha=0.9, title="platoon position")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Output tables
# --------------------------------------------------------------------------- #
def write_tables(analyses: List[PairAnalysis], outdir: str) -> Tuple[str, str]:
    summ = pd.DataFrame([pa.summary for pa in analyses])
    per_summary = os.path.join(outdir, "acc_controller_summary.csv")
    summ.to_csv(per_summary, index=False)

    ev_rows = [e for pa in analyses for e in pa.events]
    ev = pd.DataFrame(ev_rows)
    per_event = os.path.join(outdir, "acc_controller_events.csv")
    ev.to_csv(per_event, index=False)
    return per_summary, per_event


# --------------------------------------------------------------------------- #
# Sensitivity: peaks across SG smoothing windows (differentiation caveat)
# --------------------------------------------------------------------------- #
def peak_sensitivity(pairs: List[PairData], args) -> pd.DataFrame:
    rows = []
    for w in args.sg_windows:
        for p in pairs:
            a = derive_accel(p.v_follower, p.dt, w, args.sg_poly)
            rows.append(dict(sg_window=w, follower=p.follower_name,
                             peak_accel=float(np.max(a)),
                             peak_decel_mag=float(abs(np.min(a)))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _find_default_csv() -> Optional[str]:
    import glob
    cands = (["ASta_040719_platoon7.csv", "/mnt/project/ASta_040719_platoon7.csv"]
             + glob.glob("/mnt/user-data/uploads/*.csv"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="?", default=None, help="OpenACC AstaZero CSV path.")
    p.add_argument("--outdir", default="oa_outputs")
    # D1: acceleration estimator
    p.add_argument("--sg-window", type=int, default=15, help="Savitzky-Golay window (samples).")
    p.add_argument("--sg-poly", type=int, default=2, help="Savitzky-Golay polyorder.")
    p.add_argument("--sg-windows", type=int, nargs="+", default=[11, 15, 21],
                   help="Windows for peak sensitivity report.")
    # D5: regime / stable thresholds
    p.add_argument("--deadband", type=float, default=0.2, help="Net-dv deadband (m/s) for regime label.")
    p.add_argument("--eps-stable", type=float, default=0.15, help="|leader accel| ceiling for stable window.")
    p.add_argument("--tmin", type=float, default=3.0, help="Min stable-window duration (s).")
    # lag
    p.add_argument("--tau-max", type=float, default=5.0, help="Max response lag considered (s).")
    # segmentation knobs (SI defaults)
    p.add_argument("--penalty", type=float, default=55.0)
    p.add_argument("--min-seg", type=int, default=20)
    p.add_argument("--cusum-thresh", type=float, default=SI_CUSUM_THRESH)
    p.add_argument("--cusum-drift", type=float, default=SI_CUSUM_DRIFT)
    # lag figure window
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--t1", type=float, default=None)
    p.add_argument("--detrend", default="auto",
                   help="Oblique space-time detrend speed for the distance-lag "
                        "figure: 'auto' (window mean), 'none' (raw), or a float m/s.")
    p.add_argument("--common-window", action="store_true",
                   help="Render all lag figures on ONE shared window/frame "
                        "(max summed connectors across pairs) + the platoon "
                        "propagation overlay. Opt-in; default is per-pair auto.")
    p.add_argument("--span", type=float, default=60.0,
                   help="Width (s) of the shared lag window.")
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="Multiply all figure font sizes (titles, labels, ticks, "
                        "legends, lag tags). Bump for stitching panels together.")
    p.add_argument("--overlay-tags", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Show response-lag tags on the platoon overlay "
                        "(--no-overlay-tags to hide).")
    p.add_argument("--overlay-tag-spacing", type=float, default=2.5,
                   help="Min seconds between overlay lag tags in a band "
                        "(raise to thin crowded labels).")
    p.add_argument("--smoke", action="store_true",
                   help="Analyse only pair 0, print sanity, render the lag figure.")
    return p


def _resolve_window_and_vref(args, analyses, t):
    """Lag-figure window + optional shared detrend speed.
    Precedence: manual --t0/--t1 > --common-window > per-pair auto (None)."""
    manual = args.t0 is not None and args.t1 is not None
    if manual:
        w = (float(args.t0), float(args.t1))
    elif args.common_window:
        w = _common_window(analyses, t, args.span)
    else:
        return (None, None), None          # per-pair auto; no shared frame
    shared_vref = (_platoon_vref(analyses, t, w[0], w[1])
                   if str(args.detrend).lower() == "auto" else None)
    return w, shared_vref


def _overlay_vref(args, analyses, t, w, shared_vref):
    """Concrete v_ref for the platoon overlay (always skewed for legibility)."""
    if shared_vref is not None:
        return shared_vref
    if str(args.detrend).lower() == "none":
        return _platoon_vref(analyses, t, w[0], w[1])
    return float(args.detrend)


def _print_window_coverage(analyses, t, w) -> None:
    print(f"[window] shared lag window: {w[0]:.1f}-{w[1]:.1f}s")
    for pa in analyses:
        print(f"[window]   {pa.pair.follower_name}: "
              f"{_visible_connectors(pa, t, w[0], w[1])} connectors")


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    set_font_scale(args.font_scale)
    csv = args.csv or _find_default_csv()
    if not csv:
        sys.exit("error: no OpenACC AstaZero CSV found (pass a path).")
    os.makedirs(args.outdir, exist_ok=True)

    pairs, info = load_openacc_asta(csv)
    print(f"[load] {os.path.basename(csv)}  vehicles={info['n_veh']}  "
          f"dt={info['dt']:.3f}s  samples={info['n_samples']}  "
          f"gap-setting={info['distance_setting']}")
    print(f"[load] CF pairs: " +
          ", ".join(f"{p.leader_name}->{p.follower_name}" for p in pairs))

    if args.smoke:
        analyses = [analyze_pair(p, args) for p in pairs]
        t = pairs[0].t
        pa = analyses[0]
        s = pa.summary
        print("\n[smoke] pair 0 sanity ----------------------------------------")
        print(f"  follower={s['follower']}  phases={s['n_phases']}  "
              f"crit(leader/follower)={s['n_crit_leader']}/{s['n_crit_follower']}")
        print(f"  matched events: decel={s['n_decel_events']} accel={s['n_accel_events']}")
        print(f"  event lag (all): mean={s['lag_all_mean']:.2f}s "
              f"median={s['lag_all_median']:.2f}s")
        print(f"  xcorr lag={s['xcorr_lag']:.2f}s (r={s['xcorr_r']:.2f})")
        print(f"  peak accel={s['peak_accel']:.2f}  peak decel(|.|)={s['peak_decel_mag']:.2f} m/s^2")
        print(f"  stable-window follower a: rms={s['stable_a_rms']:.3f} "
              f"|a|max={s['stable_a_absmax']:.3f}  coverage={s['stable_frac']*100:.0f}%")

        # Common-window demo (force it on for the smoke, independent of the flag).
        w = _common_window(analyses, t, args.span)
        _print_window_coverage(analyses, t, w)
        vref = _platoon_vref(analyses, t, w[0], w[1])
        over = os.path.join(args.outdir, "platoon_spacetime_smoke.png")
        fig_platoon_spacetime(analyses, over, w[0], w[1], vref,
                              tags=args.overlay_tags, tag_min_dt=args.overlay_tag_spacing)
        phases = os.path.join(args.outdir,
                              f"phases_common_pair0_{s['follower'].replace(' ','')}.png")
        fig_timeseries_phases(pa, phases, w[0], w[1])
        print(f"[smoke] font_scale={args.font_scale}  overlay_tags={args.overlay_tags}")
        print(f"[smoke] wrote {over}")
        print(f"[smoke] wrote {phases}")
        return

    analyses = [analyze_pair(p, args) for p in pairs]
    t = pairs[0].t
    per_summary, per_event = write_tables(analyses, args.outdir)
    sens = peak_sensitivity(pairs, args)
    sens_path = os.path.join(args.outdir, "acc_peak_sensitivity.csv")
    sens.to_csv(sens_path, index=False)

    w, shared_vref = _resolve_window_and_vref(args, analyses, t)
    if w[0] is not None:
        _print_window_coverage(analyses, t, w)
    dist_detrend = shared_vref if shared_vref is not None else args.detrend

    figs = [per_summary, per_event, sens_path]
    for pa in analyses:
        tag = pa.pair.follower_name.replace(" ", "")
        figs.append(fig_lag_connectors(
            pa, os.path.join(args.outdir, f"lag_connectors_{tag}.png"), w[0], w[1]))
        figs.append(fig_lag_connectors_distance(
            pa, os.path.join(args.outdir, f"lag_distance_{tag}.png"),
            w[0], w[1], detrend=dist_detrend))
        figs.append(fig_timeseries_phases(
            pa, os.path.join(args.outdir, f"phases_{tag}.png"), w[0], w[1]))
        figs.append(fig_xcorr(
            pa, os.path.join(args.outdir, f"xcorr_{tag}.png")))
    figs.append(fig_peaks(analyses, os.path.join(args.outdir, "peaks_by_follower.png")))

    # Platoon propagation overlay: only meaningful on a single shared window.
    if w[0] is not None:
        ov_vref = _overlay_vref(args, analyses, t, w, shared_vref)
        figs.append(fig_platoon_spacetime(
            analyses, os.path.join(args.outdir, "platoon_spacetime.png"),
            w[0], w[1], ov_vref, tags=args.overlay_tags,
            tag_min_dt=args.overlay_tag_spacing))

    print(f"\n[done] wrote {len(figs)} outputs to {args.outdir}")
    print(pd.DataFrame([pa.summary for pa in analyses])[
        ["follower", "n_decel_events", "n_accel_events", "lag_all_median",
         "xcorr_lag", "peak_accel", "peak_decel_mag"]].to_string(index=False))


if __name__ == "__main__":
    main()
