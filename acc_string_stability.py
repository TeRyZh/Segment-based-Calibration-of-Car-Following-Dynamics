#!/usr/bin/env python3
"""
acc_string_stability.py
=======================
Validation for the ACC study, comparing the two calibration objectives produced
by ``calibrate_acc.py`` -- sample-based and segment-based -- on the same data.
The purpose of this section is to replicate ACC string instability, so the
cross-make platoon is the PRIMARY deliverable.

For each requested objective (``--which sample|segment|both``) the single per-make
IDM is free-run against the observed leader and, closed-loop, down the platoon.

  (1) PLATOON string instability  [primary]  -- closed-loop chain; cumulative
      amplitude A_i per vehicle and per-link amplification Gamma_i = A_i/A_{i-1}
      (Gamma > 1 == string unstable). Observed vs sample-sim vs segment-sim, so the
      figure answers: which objective reproduces the observed amplification?
      A_i, its std, and Gamma_i are measured over the busiest --window seconds
      (the SAME slice panel 1 draws), NOT the whole held-out segment.
  (2) PHASE-REGIME comparison  -- re-segment the simulated follower and compare
      its accel<->decel critical-point structure to observed (per-frame agreement,
      CP precision/recall/timing). One row per objective.
  (3) TRAJECTORY trace  -- observed vs each objective's follower.
  (4) HYSTERESIS  -- optional (--hysteresis).

Figures: platoon_{run}.png (obs + both arms), phase_decomp_f{pos}_{make}_{obj}.png,
trace_f{pos}_{make}.png, [hysteresis_f{pos}_{make}_{obj}.png].
Metrics: platoon_metrics.csv, phase_regime_metrics.csv, validation_metrics.csv.


command lines: 

python acc_string_stability.py --pairs-dir oa_cf_pairs --calib-dir acc_calib_ovm_run7 -o acc_string_stability

python acc_string_stability.py --pairs-dir oa_cf_pairs --calib-dir acc_calib_ovm_run7 -o acc_string_stability --run 040719_platoon7

"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import simulate
from cf_models import get_model
from phase_segmentation import segment_trajectory
from calibrate_acc import load_pair_arrays, slice_arrays, _filesafe

OBJECTIVES = ("sample", "segment")  # sample-based vs segment-based
COL = {"observed": "#000000", "leader": "#4A6FA5", "decel": "#CC3311", "accel": "#009988"}
ARM_COL = {"sample": "#0072B2", "segment": "#009E73"}

matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.grid": True,
    "grid.alpha": 0.25, "legend.fontsize": 8,
})


def _vec(named: Dict[str, float], param_names) -> np.ndarray:
    return np.array([named[p] for p in param_names], dtype=float)


def _theta_for(calib: dict, which: str) -> Optional[Dict[str, float]]:
    # 'segment' is the current key; 'theta_phase' is kept as a legacy fallback so
    # calib_*.json written before the phase->segment rename still validate.
    if which == "segment":
        return calib.get("theta_segment") or calib.get("theta_phase")
    return calib.get(f"theta_{which}") or (calib.get("theta") if which == "sample" else None)


def simulate_follower(model, theta_named, A, x_leader=None, v_leader=None):
    if not theta_named:
        return None
    xL = A["x_l"] if x_leader is None else x_leader
    vL = A["v_l"] if v_leader is None else v_leader
    return simulate.simulate(model, _vec(theta_named, model.param_names), A["t"],
                             xL, vL, A["L"], float(A["x_f"][0]), float(A["v_f"][0]))


def load_jobs(args) -> List[Tuple[dict, dict, dict]]:
    calibs = sorted(glob.glob(os.path.join(args.calib_dir, "calib_*.json")))
    if not calibs:
        sys.exit(f"No calib_*.json in {args.calib_dir}.")
    man = pd.read_csv(os.path.join(args.pairs_dir, args.manifest_name))
    if args.run:
        man = man[man["run"].astype(str).str.contains(str(args.run))]
    jobs = []
    for jp in calibs:
        with open(jp) as f:
            r = json.load(f)
        pos = r.get("follower_position")
        row = man[man["follower_position"] == pos]
        if row.empty:
            continue
        row = row.iloc[0].to_dict()
        base = os.path.basename(str(row.get("path", "")))
        csv = os.path.join(args.pairs_dir, base)
        if not os.path.exists(csv):
            csv = str(row.get("path", ""))
        if not os.path.exists(csv):
            continue
        jobs.append((r, load_pair_arrays(csv), row))   # full arrays; sliced at use
    if args.only:
        keep = {p.strip() for p in str(args.only).split(",")}
        jobs = [j for j in jobs if str(j[0].get("follower_position")) in keep]
    return jobs


def _busiest_window(t, v, win_s):
    dt = float(np.median(np.diff(t)))
    w = max(5, int(round(win_s / dt)))
    if w >= len(t):
        return 0, len(t)
    best_i, best = 0, -1.0
    for i in range(0, len(t) - w, max(1, w // 4)):
        sd = float(np.std(v[i:i + w]))
        if sd > best:
            best, best_i = sd, i
    return best_i, w


# --------------------------------------------------------------------------- #
# (1) Platoon string instability
# --------------------------------------------------------------------------- #
def build_platoon(model, run_jobs, which, warmup):
    """Closed-loop chain for one objective on the HELD-OUT segment, WARM-STARTED.

    veh1 = observed leader; each follower follows the simulated predecessor. The
    chain is simulated from `warmup` frames BEFORE the split (drawing only on the
    tail of training to let the closed loop settle) through to the end; the held-out
    part alone (from the split onward) is returned for metrics/plots, so the
    cold-start transient is absorbed by the warm-up and never scored."""
    run_jobs = sorted(run_jobs, key=lambda j: j[0].get("follower_position", 0))
    pn = model.param_names
    A0 = run_jobs[0][1]
    n = len(A0["t"])
    split = int(run_jobs[0][0].get("split", {}).get("split_index", 0))
    w0 = max(0, split - int(warmup))
    ho = split - w0                                    # held-out offset within window
    t = A0["t"]; tw = t[w0:n]

    v_obs_win = [A0["v_l"][w0:n]] + [j[1]["v_f"][w0:n] for j in run_jobs]
    gaps_win = [j[1]["s"][w0:n] for j in run_jobs]
    labels = [run_jobs[0][0].get("leader_make", "veh1")] + \
             [j[0].get("follower_make", f"veh{i+2}") for i, j in enumerate(run_jobs)]

    # observed positions in the window frame (veh1 starts at 0 at the window start)
    x_obs_win = [np.concatenate([[0.0], np.cumsum(0.5 * (v_obs_win[0][1:] + v_obs_win[0][:-1]) * np.diff(tw))])]
    for g in gaps_win:
        x_obs_win.append(x_obs_win[-1] - g)

    x_sim_win, v_sim_win = [x_obs_win[0]], [v_obs_win[0]]
    for i, (calib, _A, _row) in enumerate(run_jobs):
        th = _theta_for(calib, which)
        if not th:
            x_sim_win.append(x_obs_win[i + 1]); v_sim_win.append(v_obs_win[i + 1]); continue
        r = simulate.simulate(model, _vec(th, pn), tw, x_sim_win[-1], v_sim_win[-1],
                              0.0, float(x_obs_win[i + 1][0]), float(v_obs_win[i + 1][0]))
        x_sim_win.append(r.x); v_sim_win.append(r.v)

    # expose the held-out part only (drop the warm-up prefix [0:ho))
    return {"t": tw[ho:], "labels": labels, "warmup_frames": ho, "warmup_start": w0,
            "v_obs": [v[ho:] for v in v_obs_win], "v_sim": [v[ho:] for v in v_sim_win],
            "x_obs": [x[ho:] for x in x_obs_win], "x_sim": [x[ho:] for x in x_sim_win]}


def amplitudes(P, win=None):
    """Cumulative amplitude and per-link amplification per vehicle.

    If ``win=(i0, i1)`` is given, A_i / std / Gamma_i are measured over that slice
    (the busiest analysis window) rather than the whole held-out segment; the SAME
    slice is applied to every vehicle and both arms, so Gamma stays a fair
    observed-vs-simulated comparison. ``win=None`` falls back to the full arrays."""
    sl = slice(*win) if win is not None else slice(None)
    ncar = len(P["labels"])
    A_o = [float(np.ptp(P["v_obs"][i][sl])) for i in range(ncar)]
    A_s = [float(np.ptp(P["v_sim"][i][sl])) for i in range(ncar)]
    sd_o = [float(np.std(P["v_obs"][i][sl])) for i in range(ncar)]
    sd_s = [float(np.std(P["v_sim"][i][sl])) for i in range(ncar)]
    g_o = [np.nan] + [A_o[i] / A_o[i - 1] if A_o[i - 1] > 0 else np.nan for i in range(1, ncar)]
    g_s = [np.nan] + [A_s[i] / A_s[i - 1] if A_s[i - 1] > 0 else np.nan for i in range(1, ncar)]
    return {"A_obs": A_o, "A_sim": A_s, "std_obs": sd_o, "std_sim": sd_s,
            "gamma_obs": g_o, "gamma_sim": g_s}


def plot_platoon(run_id, P_ref, amps, sims, which_list, out_png, win_s, win=None):
    t = P_ref["t"]
    if win is None:
        i0, w = _busiest_window(t, P_ref["v_obs"][-1], win_s); win = (i0, i0 + w)
    i0, i1 = win; sl = slice(i0, i1)
    tt = t[sl] - t[i0]; ncar = len(P_ref["labels"]); pos = np.arange(1, ncar + 1)
    colors = plt.cm.plasma(np.linspace(0.08, 0.9, ncar))
    primary = "segment" if "segment" in which_list else which_list[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.4, 9.8),
                                        gridspec_kw={"height_ratios": [3, 2, 2]})

    # panel 1: observed (solid) vs the primary objective's closed-loop sim (dashed)
    for i in range(ncar):
        ax1.plot(tt, P_ref["v_obs"][i][sl], color=colors[i], lw=1.8,
                 label=f"{i+1} {P_ref['labels'][i]}")
        ax1.plot(tt, sims[primary]["v_sim"][i][sl], color=colors[i], lw=1.1, ls="--")
    ax1.set_ylabel("speed (m/s)")
    ax1.set_xlabel(f"time (s)  [busiest {int(win_s)} s window, start t={t[i0]:.0f} s;  "
                   f"solid=observed, dashed={primary}-sim]")
    ax1.legend(ncol=ncar, loc="upper right", fontsize=7)
    ax1.set_title(f"Cross-make platoon (closed loop, held-out, warm-started) - run {run_id}")

    # panel 2: cumulative amplitude A_i over the busiest window (observed + each objective)
    ax2.plot(pos, amps[primary]["A_obs"], "-o", color=COL["observed"], lw=1.8, label="observed")
    for which in which_list:
        ax2.plot(pos, amps[which]["A_sim"], "--s", color=ARM_COL[which], lw=1.6, label=f"{which}-sim")
    ax2.set_ylabel("speed peak-to-peak A$_i$ (m/s)")
    ax2.set_xticks(pos); ax2.set_xlabel("platoon position (1 = leader)")
    ax2.set_title(f"Cumulative disturbance amplitude down the platoon (busiest {int(win_s)} s)")
    ax2.legend(loc="upper left")

    # panel 3: per-link amplification Gamma_i with the Gamma=1 stability line
    links = pos[1:]
    ax3.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax3.text(links[-1], 1.0, "  Γ=1", color="0.4", va="bottom", ha="right", fontsize=8)
    ax3.plot(links, amps[primary]["gamma_obs"][1:], "-o", color=COL["observed"], lw=1.8, label="observed")
    for which in which_list:
        ax3.plot(links, amps[which]["gamma_sim"][1:], "--s", color=ARM_COL[which], lw=1.6, label=f"{which}-sim")
    ax3.set_ylabel("amplification Γ$_i$ = A$_i$/A$_{i-1}$")
    ax3.set_xticks(links); ax3.set_xticklabels([f"{i-1}→{i}" for i in links])
    ax3.set_xlabel("platoon link   (Γ > 1 = string unstable)")
    ax3.set_title("Per-link string amplification")
    ax3.legend(loc="upper left")

    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)


def platoon_metric_rows(run_id, P_ref, amps, which_list, win=None) -> List[dict]:
    t = P_ref["t"]
    sl = slice(*win) if win is not None else slice(None)
    if win is not None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        win_start_s = float(t[win[0]]); win_len_s = float((win[1] - win[0]) * dt)
    else:
        win_start_s = float(t[0]) if len(t) else np.nan
        win_len_s = float(t[-1] - t[0]) if len(t) > 1 else np.nan
    rows = []
    for i, lab in enumerate(P_ref["labels"]):
        row = {"run": run_id, "position": i + 1, "make": lab,
               "A_ptp_obs_ms": amps[which_list[0]]["A_obs"][i],
               "A_std_obs_ms": amps[which_list[0]]["std_obs"][i],
               "gamma_obs": amps[which_list[0]]["gamma_obs"][i],
               "mean_speed_ms": float(np.mean(P_ref["v_obs"][i][sl])),
               "window_start_s": win_start_s, "window_len_s": win_len_s}
        for which in which_list:
            row[f"A_ptp_{which}_ms"] = amps[which]["A_sim"][i]
            row[f"gamma_{which}"] = amps[which]["gamma_sim"][i]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# (2) Phase-regime comparison
# --------------------------------------------------------------------------- #
def _segment(A, x, v, s, args):
    return segment_trajectory(A["t"], x, v, s, penalty=args.penalty,
                              min_segment_length=args.min_seg,
                              cusum_threshold=args.cusum_threshold,
                              cusum_drift=args.cusum_drift)


def regime_per_frame(seg, v, n) -> np.ndarray:
    reg = np.zeros(n, int)
    for ph in seg.phases:
        i0, i1 = ph.i_start, ph.i_end
        if i1 <= i0:
            continue
        reg[i0:i1 + 1] = 1 if v[i1] >= v[i0] else -1
    return reg


def cp_match(obs_t, sim_t, tol) -> Tuple[float, float, float]:
    obs_t = np.asarray(obs_t, float); sim_t = np.asarray(sim_t, float)
    def _near(a, bs):
        return (bs.size > 0) and (np.min(np.abs(bs - a)) <= tol)
    rec = np.mean([_near(to, sim_t) for to in obs_t]) if obs_t.size else np.nan
    prec = np.mean([_near(ts, obs_t) for ts in sim_t]) if sim_t.size else np.nan
    errs = [float(np.min(np.abs(sim_t - to))) for to in obs_t if _near(to, sim_t)]
    return float(prec), float(rec), (float(np.mean(errs)) if errs else np.nan)


def phase_regime_row(A, seg_obs, seg_sim, r_sim, args, pos, make, pair_id, which) -> dict:
    t = A["t"]; n = len(t)
    reg_obs = regime_per_frame(seg_obs, A["v_f"], n)
    reg_sim = regime_per_frame(seg_sim, r_sim.v, n)
    obs_cp = t[np.asarray(seg_obs.critical_points, int)] if len(seg_obs.critical_points) else np.array([])
    sim_cp = t[np.asarray(seg_sim.critical_points, int)] if len(seg_sim.critical_points) else np.array([])
    prec, rec, mae = cp_match(obs_cp, sim_cp, args.cp_tol)
    return {"pair_id": pair_id, "follower_make": make, "follower_position": pos,
            "objective": which, "regime_agreement": float(np.mean(reg_obs == reg_sim)),
            "cp_precision": prec, "cp_recall": rec, "cp_timing_mae_s": mae,
            "n_cp_obs": int(len(seg_obs.critical_points)),
            "n_cp_sim": int(len(seg_sim.critical_points))}


def plot_phase_decomp(pos, make, which, A, seg_obs, r_sim, seg_sim, out_png, win_s):
    t = A["t"]; i0, w = _busiest_window(t, A["v_f"], win_s)
    sl = slice(i0, i0 + w); t0, t1 = t[i0], t[i0 + w - 1]
    scol = ARM_COL[which]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 1]})
    ax1.plot(t[sl], A["v_l"][sl], color=COL["leader"], lw=1.4, label="leader")
    ax1.plot(t[sl], A["v_f"][sl], color=COL["observed"], lw=1.6, label="observed follower")
    ax1.plot(t[sl], r_sim.v[sl], color=scol, lw=1.5, label=f"{which}-IDM (sim)")

    def _in(idxs):
        idxs = np.asarray(idxs, int)
        return idxs[(idxs >= i0) & (idxs < i0 + w)]
    for idxs, mk, cc, fill in [(seg_obs.accel_points, "^", COL["accel"], True),
                               (seg_obs.decel_points, "v", COL["decel"], True),
                               (seg_sim.accel_points, "^", COL["accel"], False),
                               (seg_sim.decel_points, "v", COL["decel"], False)]:
        ii = _in(idxs)
        if ii.size == 0:
            continue
        yv = (A["v_f"] if fill else r_sim.v)[ii]
        ax1.scatter(t[ii], yv, marker=mk, s=42, zorder=6,
                    facecolors=(cc if fill else "none"), edgecolors=cc, linewidths=1.3)
    ax1.set_ylabel("speed (m/s)"); ax1.legend(ncol=3, loc="upper left", fontsize=7)
    ax1.set_title(f"{make}: phase decomposition, {which} objective "
                  f"(obs filled / sim open markers)")

    ax2.axhline(0, color="k", lw=0.8)
    for ph in seg_obs.phases:
        a0, a1 = ph.i_start, ph.i_end
        if a1 <= a0 or a1 < i0 or a0 > i0 + w:
            continue
        dec = A["v_f"][a1] < A["v_f"][a0]
        ax2.axvspan(t[max(a0, i0)], t[min(a1, i0 + w - 1)],
                    color=COL["decel"] if dec else COL["accel"], alpha=0.10, lw=0)
    ax2.plot(t[sl], A["a_f"][sl], color=COL["observed"], lw=1.4, label="obs accel")
    ax2.plot(t[sl], r_sim.a[sl], color=scol, lw=1.3, label=f"{which} accel")
    ax2.set_ylabel("follower accel (m/s²)"); ax2.legend(loc="upper right", fontsize=7)

    reg_obs = regime_per_frame(seg_obs, A["v_f"], len(t))
    reg_sim = regime_per_frame(seg_sim, r_sim.v, len(t))
    for yc, reg, tag in [(1.0, reg_obs, "obs"), (0.0, reg_sim, "sim")]:
        ax3.fill_between(t[sl], yc, yc + 0.8, where=(reg[sl] > 0),
                         color=COL["accel"], alpha=0.6, lw=0, step="mid")
        ax3.fill_between(t[sl], yc, yc + 0.8, where=(reg[sl] < 0),
                         color=COL["decel"], alpha=0.6, lw=0, step="mid")
        ax3.text(t0, yc + 0.4, tag + " ", ha="right", va="center", fontsize=8)
    agree = float(np.mean(reg_obs[sl] == reg_sim[sl]))
    ax3.set_ylim(-0.2, 2.0); ax3.set_yticks([]); ax3.grid(False); ax3.set_xlim(t0, t1)
    ax3.set_xlabel(f"time (s)   [regime agreement in window = {agree*100:.0f}%  "
                   f"(green=accel, red=decel)]")
    fig.suptitle(f"Phase-regime validation (held-out) - f{pos} {make} ({which})", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out_png); plt.close(fig)


def plot_traces(pos, make, A, sims_by_arm, out_png, win_s):
    t = A["t"]; i0, w = _busiest_window(t, A["v_f"], win_s); sl = slice(i0, i0 + w)
    tt = t[sl] - t[i0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True)
    ax1.plot(tt, A["s"][sl], color=COL["observed"], lw=2.0, label="observed", zorder=5)
    ax2.plot(tt, A["v_f"][sl], color=COL["observed"], lw=2.0, label="observed", zorder=5)
    ax2.plot(tt, A["v_l"][sl], color="0.55", lw=1.0, ls=":", label="leader")
    for which, r in sims_by_arm.items():
        ax1.plot(tt, r.s[sl], color=ARM_COL[which], lw=1.3, label=f"{which}")
        ax2.plot(tt, r.v[sl], color=ARM_COL[which], lw=1.3, label=f"{which}")
    ax1.set_ylabel("net spacing (m)"); ax2.set_ylabel("speed (m/s)")
    ax2.set_xlabel(f"time (s)  [busiest {int(win_s)} s window, start t={t[i0]:.0f} s]")
    ax1.legend(loc="upper right")
    fig.suptitle(f"Trajectory tracking (held-out) - f{pos} {make}", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_png); plt.close(fig)


# --------------------------------------------------------------------------- #
# (4) Hysteresis (optional)
# --------------------------------------------------------------------------- #
def _smooth(y, w):
    if w and w > 1:
        return pd.Series(y).rolling(int(w), center=True, min_periods=1).mean().to_numpy()
    return np.asarray(y, float)


def _binned_mean(x, y, edges, mc):
    out = np.full(len(edges) - 1, np.nan)
    if len(x):
        idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            m = idx == b
            if np.count_nonzero(m) >= mc:
                out[b] = float(np.mean(y[m]))
    return out


def hyst_width(vf, s, vl, edges, mc, sw):
    vf, ss, vl = _smooth(vf, sw), _smooth(s, sw), _smooth(vl, sw)
    closing = (vf - vl) >= 0.0
    sc = _binned_mean(vf[closing], ss[closing], edges, mc)
    so = _binned_mean(vf[~closing], ss[~closing], edges, mc)
    width = so - sc; valid = ~np.isnan(width)
    return (float(np.nanmean(width[valid])) if valid.sum() >= 2 else np.nan), sc, so


def plot_hysteresis(pos, make, which, A, r_sim, edges, out_png, mc, sw):
    c = 0.5 * (edges[:-1] + edges[1:]); scol = ARM_COL[which]
    wo, sco, soo = hyst_width(A["v_f"], A["s"], A["v_l"], edges, mc, sw)
    ws, scs, sos = hyst_width(r_sim.v, r_sim.s, A["v_l"], edges, mc, sw)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    vo = ~(np.isnan(sco) | np.isnan(soo))
    ax.fill_between(c[vo], sco[vo], soo[vo], color="0.6", alpha=0.30, lw=0)
    ax.plot(c, sco, color=COL["observed"], lw=1.4); ax.plot(c, soo, color=COL["observed"], lw=1.4, ls="--")
    vs = ~(np.isnan(scs) | np.isnan(sos))
    ax.fill_between(c[vs], scs[vs], sos[vs], color=scol, alpha=0.22, lw=0)
    ax.plot(c, scs, color=scol, lw=1.6); ax.plot(c, sos, color=scol, lw=1.6, ls="--")
    ax.set_xlabel("follower speed (m/s)"); ax.set_ylabel("net spacing (m)")
    ax.set_title(f"f{pos} {make} ({which})\n|width| obs={abs(wo):.2f}  sim={abs(ws):.2f} m")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    return wo, ws


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _process_pair(model, calib, A, row, args, which_list, prows, mrows):
    make = calib.get("follower_make", row.get("follower_make", "?"))
    pos = calib.get("follower_position", row.get("follower_position", 0))
    pair_id = row.get("pair_id", f"f{pos}")
    seg_obs = _segment(A, A["x_f"], A["v_f"], A["s"], args)

    edges = None
    if args.hysteresis:
        vmin = float(np.floor(np.min(A["v_f"]))); vmax = float(np.ceil(np.max(A["v_f"])))
        edges = np.arange(vmin, vmax + args.speed_bin, args.speed_bin)

    sims_by_arm: Dict[str, object] = {}
    summary_bits = []
    for which in which_list:
        th = _theta_for(calib, which)
        r = simulate_follower(model, th, A)
        if r is None:
            continue
        sims_by_arm[which] = r
        seg_sim = _segment(A, r.x, r.v, r.s, args)
        prow = phase_regime_row(A, seg_obs, seg_sim, r, args, pos, make, pair_id, which)
        prows.append(prow)
        plot_phase_decomp(pos, make, which, A, seg_obs, r, seg_sim,
                          os.path.join(args.output_dir,
                                       f"phase_decomp_f{pos}_{_filesafe(make)}_{which}.png"),
                          args.phase_window)
        rmse = float(np.sqrt(np.mean((r.s - A["s"]) ** 2)))
        wo = ws = np.nan
        if args.hysteresis:
            wo, ws = plot_hysteresis(pos, make, which, A, r, edges,
                                     os.path.join(args.output_dir,
                                                  f"hysteresis_f{pos}_{_filesafe(make)}_{which}.png"),
                                     args.min_count, args.smooth)
        mrows.append({"pair_id": pair_id, "follower_make": make, "follower_position": pos,
                      "objective": which, "rmse_spacing_m": rmse,
                      "peak_decel_ms2": float(np.min(r.a)),
                      "hyst_width_obs_m": wo, "hyst_width_sim_m": ws,
                      "n_barrier": int(getattr(r, "n_barrier", 0))})
        summary_bits.append(f"{which}: RMSE {rmse:.2f}m, regime {prow['regime_agreement']*100:.0f}%, "
                            f"CPrec {prow['cp_recall']:.2f}")

    if sims_by_arm:
        plot_traces(pos, make, A, sims_by_arm,
                    os.path.join(args.output_dir, f"trace_f{pos}_{_filesafe(make)}.png"),
                    args.window)
    print(f"[f{pos}] {make}: " + "  |  ".join(summary_bits))


def run(args) -> None:
    if args.self_test:
        _selftest(args)
        return
    if not (args.pairs_dir and args.calib_dir):
        sys.exit("error: --pairs-dir and --calib-dir are required.")
    which_list = list(OBJECTIVES) if args.which == "both" else [args.which]
    os.makedirs(args.output_dir, exist_ok=True)
    jobs = load_jobs(args)
    if not jobs:
        sys.exit("No (calibration, pair) jobs found.")

    prows: List[dict] = []
    mrows: List[dict] = []
    dt = float(jobs[0][1]["t"][1] - jobs[0][1]["t"][0])
    warmup_frames = int(round(args.platoon_warmup_sec / dt))
    for calib, A_full, row in jobs:
        model = get_model(str(calib.get("model", "idm")))   # <-- model per calib JSON
        split = int(calib.get("split", {}).get("split_index", 0))
        A_ho = slice_arrays(A_full, split, len(A_full["t"])) if split > 0 else A_full
        _process_pair(model, calib, A_ho, row, args, which_list, prows, mrows)

    # (1) platoon per run, both objectives overlaid (warm-started, held-out)
    plat_rows: List[dict] = []
    by_run: "OrderedDict[str, list]" = OrderedDict()
    for calib, A, row in jobs:
        by_run.setdefault(str(row.get("run", "run")), []).append((calib, A, row))
    for run_id, rjobs in by_run.items():
        if len(rjobs) < 2:
            print(f"  (run {run_id}: <2 followers, platoon skipped)")
            continue
        model = get_model(str(rjobs[0][0].get("model", "idm")))
        sims = {which: build_platoon(model, rjobs, which, warmup_frames)
                for which in which_list}
        P_ref = sims[which_list[0]]
        # ONE observed-anchored busiest window; amplitudes/Gamma measured on it.
        i0, w = _busiest_window(P_ref["t"], P_ref["v_obs"][-1], args.window)
        win = (i0, i0 + w)
        amps = {which: amplitudes(sims[which], win) for which in which_list}
        plat_rows.extend(platoon_metric_rows(run_id, P_ref, amps, which_list, win=win))
        plot_platoon(run_id, P_ref, amps, sims, which_list,
                     os.path.join(args.output_dir, f"platoon_{_filesafe(run_id)}.png"),
                     args.window, win=win)

    if prows:
        pd.DataFrame(prows).to_csv(os.path.join(args.output_dir, "phase_regime_metrics.csv"), index=False)
    if mrows:
        pd.DataFrame(mrows).to_csv(os.path.join(args.output_dir, "validation_metrics.csv"), index=False)
    if plat_rows:
        pd.DataFrame(plat_rows).to_csv(os.path.join(args.output_dir, "platoon_metrics.csv"), index=False)
    with open(os.path.join(args.output_dir, "validation_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("\n" + "=" * 74)
    print(f"Validated {len(jobs)} pair(s), objectives={which_list} -> {args.output_dir}")
    if plat_rows:
        pdf = pd.DataFrame(plat_rows)
        keep = ["run", "position", "make", "A_ptp_obs_ms"] + \
               [f"A_ptp_{w}_ms" for w in which_list] + \
               ["gamma_obs"] + [f"gamma_{w}" for w in which_list]
        keep = [c for c in keep if c in pdf.columns]
        print(f"String amplification (observed vs objectives; busiest {int(args.window)} s window):")
        print(pdf[keep].round(3).to_string(index=False))
    print("=" * 74)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest(args) -> None:
    import tempfile
    tmp = tempfile.mkdtemp()
    pairs_dir = os.path.join(tmp, "pairs"); calib_dir = os.path.join(tmp, "calib")
    os.makedirs(pairs_dir); os.makedirs(calib_dir)
    model = get_model("idm"); pn = model.param_names
    dt, n = 0.1, 1500
    t = np.round(np.arange(n) * dt, 4)
    v1 = np.clip(18 + 6 * np.sin(2 * np.pi * t / 45.0), 0, None)
    x1 = np.concatenate([[400.0], 400.0 + np.cumsum(0.5 * (v1[1:] + v1[:-1]) * dt)])
    th = {k: v for k, v in zip(pn, [30, 1.1, 1.6, 2.2, 2.5])}
    th2 = {k: v for k, v in zip(pn, [30, 1.3, 1.4, 2.6, 2.0])}
    r2 = simulate.simulate(model, _vec(th, pn), t, x1, v1, 0.0, x1[0] - 22.0, 18.0)
    r3 = simulate.simulate(model, _vec(th, pn), t, r2.x, r2.v, 0.0, r2.x[0] - 22.0, 18.0)
    specs = [(2, "Car B", "Car A", x1, v1, r2), (3, "Car C", "Car B", r2.x, r2.v, r3)]
    for pos, mk, lead, xl, vl, r in specs:
        pd.DataFrame({
            "t": t, "x_follower": r.x, "v_follower": r.v, "a_follower": r.a,
            "x_leader": xl, "v_leader": vl, "leader_length": 0.0,
            "spacing": r.s, "dv": r.v - vl,
        }).to_csv(os.path.join(pairs_dir, f"oa_pair_syn_f{pos}_{mk.replace(' ', '_')}.csv"), index=False)
    pd.DataFrame([{"pair_id": f"syn_f{p}", "run": "syn", "follower_position": p,
                   "follower_make": mk, "leader_make": lead,
                   "path": os.path.join(pairs_dir, f"oa_pair_syn_f{p}_{mk.replace(' ', '_')}.csv")}
                  for p, mk, lead, *_ in specs]).to_csv(
        os.path.join(pairs_dir, "manifest.csv"), index=False)
    for p, mk, *_ in specs:
        json.dump({"model": "idm", "follower_make": mk, "follower_position": p,
                   "split": {"split_index": int(round(0.7 * n)), "train_frac": 0.7, "n": n},
                   "theta_sample": th, "theta_segment": th2},
                  open(os.path.join(calib_dir, f"calib_f{p}_{mk.replace(' ', '_')}.json"), "w"))

    args.pairs_dir, args.calib_dir = pairs_dir, calib_dir
    args.output_dir = args.output_dir or os.path.join(tmp, "out")
    args.hysteresis = True
    os.makedirs(args.output_dir, exist_ok=True)
    model = get_model("idm"); jobs = load_jobs(args)
    assert len(jobs) == 2, f"expected 2 jobs, got {len(jobs)}"
    which_list = list(OBJECTIVES)
    prows, mrows = [], []
    for calib, A_full, row in jobs:
        split = int(calib.get("split", {}).get("split_index", 0))
        A_ho = slice_arrays(A_full, split, len(A_full["t"])) if split > 0 else A_full
        _process_pair(model, calib, A_ho, row, args, which_list, prows, mrows)
    wf = 120
    sims = {w: build_platoon(model, [(c, A, r) for c, A, r in jobs], w, wf) for w in which_list}
    P_ref = sims[which_list[0]]
    i0, ww = _busiest_window(P_ref["t"], P_ref["v_obs"][-1], args.window)
    win = (i0, i0 + ww)
    amps = {w: amplitudes(sims[w], win) for w in which_list}
    pm = platoon_metric_rows("syn", P_ref, amps, which_list, win=win)
    assert pm and {"window_start_s", "window_len_s"} <= set(pm[0]), "window metric cols"
    plot_platoon("syn", P_ref, amps, sims, which_list,
                 os.path.join(args.output_dir, "platoon_syn.png"), args.window, win=win)
    for f in ["phase_decomp_f2_Car_B_sample.png", "phase_decomp_f2_Car_B_segment.png",
              "trace_f2_Car_B.png", "hysteresis_f2_Car_B_segment.png", "platoon_syn.png"]:
        assert os.path.exists(os.path.join(args.output_dir, f)), f"missing {f}"
    assert sum(1 for p in prows if p["objective"] == "segment") == 2, "segment rows"
    print(f"\nSelf-test PASS: both objectives validated, platoon overlays sample+segment, "
          f"A_i/Gamma_i on busiest window, figures + metrics written to {args.output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ACC platoon string-instability validation comparing "
                    "sample-based vs segment-based calibrations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pairs-dir", help="Directory with pair CSVs + manifest.csv.")
    p.add_argument("--calib-dir", help="Directory with calib_*.json.")
    p.add_argument("-o", "--output-dir", default="acc_string_stability")
    p.add_argument("--which", choices=["sample", "segment", "both"], default="both",
                   help="Which calibration objective(s) to validate/overlay.")
    p.add_argument("--run", default=None,
                   help="Select ONE run by filename substring (match the calibration run).")
    p.add_argument("--manifest-name", default="manifest.csv")
    p.add_argument("--only", default=None, help="Comma-separated follower positions.")
    p.add_argument("--window", type=float, default=120.0,
                   help="Busiest-window length (s) for the platoon/trace figures AND "
                        "the A_i / Gamma_i amplitude metrics.")
    p.add_argument("--platoon-warmup-sec", type=float, default=15.0,
                   help="Seconds of pre-split warm-up for the closed-loop held-out "
                        "platoon (settles the chain; not scored).")
    p.add_argument("--hysteresis", action="store_true",
                   help="Also emit per-make spacing-hysteresis figures.")

    hy = p.add_argument_group("hysteresis binning (only with --hysteresis)")
    hy.add_argument("--speed-bin", type=float, default=1.0)
    hy.add_argument("--min-count", type=int, default=5)
    hy.add_argument("--smooth", type=int, default=5)

    ph = p.add_argument_group("phase-regime (match calibration segmentation)")
    ph.add_argument("--penalty", type=float, default=75.0)
    ph.add_argument("--min-seg", type=int, default=20)
    ph.add_argument("--cusum-threshold", type=float, default=2.1)
    ph.add_argument("--cusum-drift", type=float, default=0.3)
    ph.add_argument("--cp-tol", type=float, default=0.5,
                    help="Tolerance (s) for matching observed/simulated critical points.")
    ph.add_argument("--phase-window", type=float, default=60.0,
                    help="Window (s) shown in the phase-decomposition figure.")

    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
