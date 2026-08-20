#!/usr/bin/env python3
"""
aggregate_astazero.py
=====================
Run-AGGREGATED versions of the two single-file ACC summaries:

    fig_xcorr  -> signal-based response lag tau* per pair
    fig_peaks  -> peak accel / decel per follower

computed over EVERY ASta_*_platoon*.csv in a folder instead of one run.

Aggregation unit (settled)
--------------------------
ONE value per run per group. Each run contributes a single tau*, a single r, a
single peak-accel and a single peak-decel to each group, so n = number of runs
-- NOT number of windows. No within-run pseudoreplication. The figures show
central tendency + spread + the individual run points, so n is read directly.

Grouping key (D1-A: make)
-------------------------
* peaks : follower vehicle (make+model, e.g. "Tesla Model 3").
* lag   : ordered pair leader->follower (e.g. "Audi A8 -> Tesla Model 3").
Because a make can sit at different platoon positions in different runs, a
make x position cross-tab is printed and written so the confounding is explicit.

Spread (D2-A): bar = median, whisker = IQR (Q1-Q3), dots = per-run values,
diamond = mean. Lag form (D3-A): tau* distribution per pair (median peak
correlation r annotated).

This is a THIN driver: load_openacc_asta / analyze_pair are imported verbatim
from acc_controller_behavior.py, and resolve_files / default_analysis_args from
survey_astazero.py. Only the aggregation + the two figures are new here.

Note on r: a run's tau* is only counted toward a pair when its cross-correlation
peak r is finite (a real peak); runs with no finite r are dropped from the lag
stats. --min-r raises that floor.

Usage
-----
  python aggregate_astazero.py /path/to/AstaZero        # aggregate + render
  python aggregate_astazero.py /path/to/AstaZero --min-r 0.5
  python aggregate_astazero.py --self-test              # synthetic multi-run check
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

import acc_controller_behavior as acb          # verbatim analysis + constants + fonts
from survey_astazero import resolve_files, default_analysis_args

import matplotlib.pyplot as plt                # backend already set to Agg by acb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

LAG_BAR = "#5b7fa6"
LAG_DOT = "#2c3e50"
BAND_MID = "#f39c12"
GREY = "#7f8c8d"


# --------------------------------------------------------------------------- #
# Stats helper
# --------------------------------------------------------------------------- #
def group_stats(values) -> dict:
    """median/mean/IQR/min/max over the finite values, plus the finite array."""
    v = np.asarray([x for x in np.asarray(values, float) if np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, q1=np.nan, q3=np.nan,
                    vmin=np.nan, vmax=np.nan, values=v)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                vmin=float(v.min()), vmax=float(v.max()), values=v)


def _ordered_groups(long: pd.DataFrame, name_col: str, pos_col: str) -> List[str]:
    """Group labels sorted by median follower position (leader -> tail)."""
    med = long.groupby(name_col)[pos_col].median().sort_values()
    return list(med.index)


# --------------------------------------------------------------------------- #
# Extraction: one scalar row per (run, pair) + the per-run xcorr curve
# --------------------------------------------------------------------------- #
def extract_all(files: List[str], analysis_args, grid_step: float = 0.05):
    """Single pass over the files. Returns:
        long  : DataFrame, one scalar row per (run, pair)
        grid  : common lag axis (0..tau_max) the curves are interpolated onto
        curves: list of dicts, each the per-run corr(dv, a_f(t+tau)) curve
    Curves are captured here (not re-run) so files are segmented only once."""
    rows, curves = [], []
    grid = np.arange(0.0, float(analysis_args.tau_max) + 1e-9, grid_step)
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            pairs, _info = acb.load_openacc_asta(path)
            analyses = [acb.analyze_pair(p, analysis_args) for p in pairs]
        except Exception as e:                                   # robust scan
            print(f"[skip] {os.path.basename(path)}: {e!r}")
            continue
        for pa in analyses:
            s = pa.summary
            r = float(s["xcorr_r"])
            r = r if np.isfinite(r) else np.nan
            rows.append(dict(
                run=name,
                leader_name=s["leader"], follower_name=s["follower"],
                leader_pos=int(s["leader_veh"]), follower_pos=int(s["follower_veh"]),
                pair=f"{s['leader']} \u2192 {s['follower']}",
                tau=float(s["xcorr_lag"]), r=r,
                peak_accel=float(s["peak_accel"]) if np.isfinite(s["peak_accel"]) else np.nan,
                peak_decel_mag=float(s["peak_decel_mag"]) if np.isfinite(s["peak_decel_mag"]) else np.nan,
            ))
            # pa.xcorr = (tau*, r*, lags, r_curve); interp onto the common grid.
            tau_s, _r_s, lags, rs = pa.xcorr
            lags = np.asarray(lags, float); rs = np.asarray(rs, float)
            fin = np.isfinite(rs)
            curve = (np.interp(grid, lags[fin], rs[fin]) if fin.sum() >= 2
                     else np.full(grid.shape, np.nan))
            curves.append(dict(
                run=name, follower_name=s["follower"], leader_name=s["leader"],
                follower_pos=int(s["follower_veh"]),
                pair=f"{s['leader']} \u2192 {s['follower']}",
                tau=float(tau_s), r=r, curve=curve))
        print(f"[ok]   {os.path.basename(path):32s} pairs={len(analyses)}")
    return pd.DataFrame(rows), grid, curves


# --------------------------------------------------------------------------- #
# Aggregation tables
# --------------------------------------------------------------------------- #
def aggregate_peaks(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in _ordered_groups(long, "follower_name", "follower_pos"):
        sub = long[long["follower_name"] == g]
        a, d = group_stats(sub["peak_accel"]), group_stats(sub["peak_decel_mag"])
        pos = sorted(sub["follower_pos"].unique().tolist())
        rows.append(dict(
            follower=g, positions="/".join(map(str, pos)), n_runs=sub.shape[0],
            accel_median=round(a["median"], 3), accel_mean=round(a["mean"], 3),
            accel_q1=round(a["q1"], 3), accel_q3=round(a["q3"], 3),
            accel_min=round(a["vmin"], 3), accel_max=round(a["vmax"], 3),
            decel_median=round(d["median"], 3), decel_mean=round(d["mean"], 3),
            decel_q1=round(d["q1"], 3), decel_q3=round(d["q3"], 3),
            decel_min=round(d["vmin"], 3), decel_max=round(d["vmax"], 3)))
    return pd.DataFrame(rows)


def aggregate_lag(long: pd.DataFrame, min_r: float = 0.0,
                  by: str = "follower") -> pd.DataFrame:
    """Aggregate tau* by follower make (default) or by leader->follower pair.
    tau* is a follower-response property, so 'follower' pools over leaders and
    raises n per group; 'pair' keeps the directed make-to-make view."""
    gcol = "follower_name" if by == "follower" else "pair"
    label = "follower" if by == "follower" else "pair"
    rows = []
    for g in _ordered_groups(long, gcol, "follower_pos"):
        sub = long[(long[gcol] == g) & np.isfinite(long["r"]) & (long["r"] >= min_r)]
        st = group_stats(sub["tau"])
        r_med = float(np.nanmedian(sub["r"])) if sub["r"].notna().any() else np.nan
        pos = sorted(long[long[gcol] == g]["follower_pos"].unique().tolist())
        rows.append({
            label: g, "positions": "/".join(map(str, pos)), "n_runs": st["n"],
            "tau_median": round(st["median"], 3), "tau_mean": round(st["mean"], 3),
            "tau_q1": round(st["q1"], 3), "tau_q3": round(st["q3"], 3),
            "tau_min": round(st["vmin"], 3), "tau_max": round(st["vmax"], 3),
            "r_median": round(r_med, 3)})
    return pd.DataFrame(rows)


def make_position_crosstab(long: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(long["follower_name"], long["follower_pos"])
    order = _ordered_groups(long, "follower_name", "follower_pos")
    return ct.reindex([g for g in order if g in ct.index])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_aggregate_peaks(long: pd.DataFrame, out: str, rng_seed: int = 0) -> str:
    groups = _ordered_groups(long, "follower_name", "follower_pos")
    if not groups:
        return out
    rng = np.random.default_rng(rng_seed)
    x = np.arange(len(groups)); w = 0.38

    stats, allvals = {}, []
    for g in groups:
        sub = long[long["follower_name"] == g]
        a, d = group_stats(sub["peak_accel"]), group_stats(sub["peak_decel_mag"])
        stats[g] = (a, d)
        allvals += list(a["values"]) + list(d["values"])
    ymax = max((max(allvals) if allvals else acb.HARSH_LEVEL) * 1.15 + 0.5,
               acb.HARSH_LEVEL + 0.5)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.9 * len(groups) + 3.0), 6.0))
    ax.axhspan(0, acb.COMFORT_ACCEL, color=acb.C_ACCEL, alpha=0.06)
    ax.axhspan(acb.COMFORT_ACCEL, acb.HARSH_LEVEL, color=BAND_MID, alpha=0.06)
    ax.axhspan(acb.HARSH_LEVEL, ymax, color=acb.C_DECEL, alpha=0.06)
    ax.axhline(acb.COMFORT_ACCEL, color=GREY, ls="--", lw=1)
    ax.axhline(acb.HARSH_LEVEL, color=acb.C_DECEL, ls="--", lw=1)

    def draw(xc, st, col):
        ax.bar(xc, st["median"], w, color=col, alpha=0.30, edgecolor=col,
               linewidth=1.2, zorder=2)
        if st["n"] > 0 and np.isfinite(st["q1"]):
            ax.plot([xc, xc], [st["q1"], st["q3"]], color=col, lw=2.2,
                    zorder=4, solid_capstyle="round")
        if st["values"].size:
            jj = rng.uniform(-0.06, 0.06, size=st["values"].size)
            ax.scatter(xc + jj, st["values"], s=20, color=col, edgecolor="white",
                       linewidth=0.4, alpha=0.85, zorder=5)
        if np.isfinite(st["mean"]):
            ax.scatter(xc, st["mean"], marker="D", s=34, facecolor="white",
                       edgecolor=col, linewidth=1.4, zorder=6)

    for i, g in enumerate(groups):
        a, d = stats[g]
        draw(x[i] - w / 2, a, acb.C_ACCEL)
        draw(x[i] + w / 2, d, acb.C_DECEL)

    labels = [f"{g}\n(n={long[long['follower_name'] == g].shape[0]})" for g in groups]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("peak |acceleration| (m/s$^2$)"); ax.set_ylim(0, ymax)
    ax.set_title("ACC peak accel / decel per follower, aggregated across runs\n"
                 "(bar = median, whisker = IQR, dots = runs, "
                 "$\\diamond$ = mean; comfort bands reference only)", wrap=True)
    handles = [
        Patch(facecolor=acb.C_ACCEL, alpha=0.5, label="peak acceleration"),
        Patch(facecolor=acb.C_DECEL, alpha=0.5, label="peak deceleration (|.|)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="w",
               markeredgecolor="k", label="mean", markersize=8),
        Line2D([0], [0], color="k", lw=2.2, label="IQR"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k",
               label="per-run", markersize=6),
        Line2D([0], [0], color=GREY, ls="--", label=f"comfortable ~{acb.COMFORT_ACCEL:.1f}"),
        Line2D([0], [0], color=acb.C_DECEL, ls="--", label=f"harsh > {acb.HARSH_LEVEL:.1f}")]
    ax.legend(handles=handles, fontsize=acb.FONT["legend"], ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    return out


def fig_aggregate_lag(long: pd.DataFrame, out: str, min_r: float = 0.0,
                      by: str = "follower", rng_seed: int = 1) -> str:
    gcol = "follower_name" if by == "follower" else "pair"
    groups = _ordered_groups(long, gcol, "follower_pos")
    if not groups:
        return out
    rng = np.random.default_rng(rng_seed)
    x = np.arange(len(groups)); w = 0.5

    fig, ax = plt.subplots(figsize=(max(9.0, 1.95 * len(groups) + 3.0), 5.8))
    tops = []
    for i, g in enumerate(groups):
        sub = long[(long[gcol] == g) & np.isfinite(long["r"]) & (long["r"] >= min_r)]
        st = group_stats(sub["tau"])
        if st["n"] == 0:
            continue
        r_med = float(np.nanmedian(sub["r"])) if sub["r"].notna().any() else np.nan
        ax.bar(x[i], st["median"], w, color=LAG_BAR, alpha=0.30, edgecolor=LAG_BAR,
               linewidth=1.2, zorder=2)
        if np.isfinite(st["q1"]):
            ax.plot([x[i], x[i]], [st["q1"], st["q3"]], color=LAG_BAR, lw=2.4,
                    zorder=4, solid_capstyle="round")
        jj = rng.uniform(-0.08, 0.08, size=st["values"].size)
        ax.scatter(x[i] + jj, st["values"], s=22, color=LAG_DOT, edgecolor="white",
                   linewidth=0.4, alpha=0.85, zorder=5)
        if np.isfinite(st["mean"]):
            ax.scatter(x[i], st["mean"], marker="D", s=36, facecolor="white",
                       edgecolor=LAG_DOT, linewidth=1.4, zorder=6)
        top = max(st["q3"], st["vmax"], st["median"])
        tops.append(top)
        ax.text(x[i], top + 0.10, f"r={r_med:.2f}\nn={st['n']}", ha="center",
                va="bottom", fontsize=acb.FONT["tag"])

    ymax = (max(tops) if tops else 3.0) * 1.20 + 0.4
    if by == "follower":
        labels = list(groups)                       # single-line make labels
        gtxt = "follower"
    else:
        labels = [g.replace(" \u2192 ", "\n\u2192 ") for g in groups]
        gtxt = "leader\u2192follower pair"
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=acb.FONT["tick"])
    ax.set_ylabel(r"signal-based response lag $\tau^*$ (s)"); ax.set_ylim(0, ymax)
    ax.set_title(f"Signal-based response lag per {gtxt}, aggregated across runs\n"
                 "(bar = median, whisker = IQR, dots = runs, "
                 "$\\diamond$ = mean; r = median peak correlation)", wrap=True)
    handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="w",
               markeredgecolor=LAG_DOT, label="mean", markersize=8),
        Line2D([0], [0], color=LAG_BAR, lw=2.4, label="IQR"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LAG_DOT,
               label=r"per-run $\tau^*$", markersize=6)]
    ax.legend(handles=handles, fontsize=acb.FONT["legend"], framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    return out


def fig_aggregate_xcorr(grid: np.ndarray, curves: List[dict], out: str,
                        by: str = "follower", min_r: float = 0.0,
                        ncols: int = 2, show_runs: bool = True,
                        band: bool = False) -> str:
    """Aggregated cross-correlation curves in the style of the single-file
    fig_xcorr: one small-multiple panel per group (follower make by default),
    with per-run corr(dv, a_f(t+tau)) curves thin, the run-mean curve bold, and
    the mean curve's peak marked as tau*. At n=1 a panel is the single-file
    curve; with several runs the bold curve is their mean and the thin curves
    show the spread across runs (different leaders / positions)."""
    from collections import defaultdict
    gkey = "follower_name" if by == "follower" else "pair"
    byg = defaultdict(list)
    for c in curves:
        if np.isfinite(c["r"]) and c["r"] >= min_r and np.isfinite(c["curve"]).any():
            byg[c[gkey]].append(c)
    if not byg:
        return out
    groups = sorted(byg.keys(),
                    key=lambda g: np.median([c["follower_pos"] for c in byg[g]]))

    n = len(groups)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(6.2 * ncols, 3.9 * nrows))
    for k, g in enumerate(groups):
        ax = axes[k // ncols][k % ncols]
        M = np.vstack([r["curve"] for r in byg[g]])          # n_runs x len(grid)
        mean_curve = np.nanmean(M, axis=0)
        if show_runs and M.shape[0] > 1:
            for row in M:
                ax.plot(grid, row, color=acb.C_FOLL, lw=0.9, alpha=0.28, zorder=2)
        if band and M.shape[0] > 1:
            q1 = np.nanpercentile(M, 25, axis=0)
            q3 = np.nanpercentile(M, 75, axis=0)
            ax.fill_between(grid, q1, q3, color=acb.C_FOLL, alpha=0.12, zorder=1)
        ax.plot(grid, mean_curve, color=acb.C_FOLL, lw=2.4, zorder=4)
        ipk = int(np.nanargmax(mean_curve))
        tau_star, r_pk = float(grid[ipk]), float(mean_curve[ipk])
        ax.axvline(tau_star, color=acb.C_DECEL, ls="--", lw=1.4, zorder=5,
                   label=f"$\\tau^*$={tau_star:.2f}s (r={r_pk:.2f}), n={M.shape[0]}")
        ax.set_title(g, fontsize=acb.FONT["title"])
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=acb.FONT["legend"], framealpha=0.9)
        if k // ncols == nrows - 1:
            ax.set_xlabel(r"lag $\tau$ (s)")
        if k % ncols == 0:
            ax.set_ylabel(r"corr$(\Delta v(t),\, a_f(t+\tau))$")
    for j in range(n, nrows * ncols):                        # hide unused axes
        axes[j // ncols][j % ncols].axis("off")

    gtxt = "follower" if by == "follower" else "leader\u2192follower pair"
    fig.suptitle(f"Signal-based response lag per {gtxt}, aggregated across runs\n"
                 "(bold = mean cross-correlation, thin = per-run, "
                 "dashed = $\\tau^*$ of the mean)",
                 fontsize=acb.FONT["title"] * 1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=300); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Shared report + write
# --------------------------------------------------------------------------- #
def report_and_write(long: pd.DataFrame, grid: np.ndarray, curves: List[dict],
                     outdir: str, min_r: float, prefix: str = "",
                     lag_by: str = "follower") -> List[str]:
    long_path = os.path.join(outdir, f"{prefix}aggregate_long.csv")
    long.to_csv(long_path, index=False)

    lag_first = "follower" if lag_by == "follower" else "pair"
    peaks = aggregate_peaks(long)
    lag = aggregate_lag(long, min_r, by=lag_by)
    ct = make_position_crosstab(long)
    peaks_csv = os.path.join(outdir, f"{prefix}aggregate_peaks_summary_make.csv")
    lag_csv = os.path.join(outdir, f"{prefix}aggregate_lag_summary_{lag_by}.csv")
    ct_csv = os.path.join(outdir, f"{prefix}aggregate_make_position_crosstab.csv")
    peaks.to_csv(peaks_csv, index=False)
    lag.to_csv(lag_csv, index=False)
    ct.to_csv(ct_csv)

    n_runs = long["run"].nunique()
    print(f"\n[aggregate] runs={n_runs}  pairs(rows)={len(long)}  "
          f"min_r={min_r}  lag_by={lag_by}")
    print("\n[make x follower-position cross-tab]  (cells = run count)")
    print(ct.to_string())
    print("\n[peaks by follower]")
    print(peaks[["follower", "positions", "n_runs", "accel_median",
                 "decel_median", "decel_max"]].to_string(index=False))
    print(f"\n[lag by {lag_by}]")
    print(lag[[lag_first, "positions", "n_runs", "tau_median", "tau_q1",
               "tau_q3", "r_median"]].to_string(index=False))

    xcorr_png = fig_aggregate_xcorr(
        grid, curves,
        os.path.join(outdir, f"{prefix}aggregate_xcorr_by_{lag_by}.png"),
        by=lag_by, min_r=min_r)
    peaks_png = fig_aggregate_peaks(
        long, os.path.join(outdir, f"{prefix}aggregate_peaks_by_make.png"))
    lag_png = fig_aggregate_lag(
        long, os.path.join(outdir, f"{prefix}aggregate_lag_by_{lag_by}.png"),
        min_r, by=lag_by)
    return [xcorr_png, lag_png, peaks_png, peaks_csv, lag_csv, ct_csv, long_path]


# --------------------------------------------------------------------------- #
# Self-test: fabricate multi-run data (two vehicle orders) to exercise the
# spread / dots / cross-tab paths that a single real file cannot.
# --------------------------------------------------------------------------- #
def _synthetic_curve(grid: np.ndarray, tau_star: float, r_peak: float,
                     width: float, floor: float, rng) -> np.ndarray:
    """A concave example-like curve peaking at tau_star with value r_peak."""
    c = r_peak - ((grid - tau_star) / width) ** 2 * (r_peak - floor)
    c = np.clip(c, floor - 0.03, r_peak) + rng.normal(0, 0.004, size=grid.shape)
    return c


def _fabricate_synthetic(seed: int = 7, grid_step: float = 0.05,
                         tau_max: float = 5.0):
    rng = np.random.default_rng(seed)
    orders = {
        "runA": ["Audi A8", "Tesla Model 3", "BMW X5", "Audi A6", "Mercedes A-Class"],
        "runB": ["Audi A8", "Tesla Model 3", "BMW X5", "Audi A6", "Mercedes A-Class"],
        "runC": ["Audi A8", "BMW X5", "Tesla Model 3", "Audi A6", "Mercedes A-Class"],
        "runD": ["Audi A8", "BMW X5", "Tesla Model 3", "Audi A6", "Mercedes A-Class"],
    }
    accel_mu = {"Tesla Model 3": 2.4, "BMW X5": 1.6, "Audi A6": 1.5,
                "Mercedes A-Class": 1.9}
    decel_mu = {"Tesla Model 3": 2.0, "BMW X5": 2.3, "Audi A6": 2.8,
                "Mercedes A-Class": 3.7}
    grid = np.arange(0.0, tau_max + 1e-9, grid_step)
    rows, curves = [], []
    for run, order in orders.items():
        for i in range(1, len(order)):
            L, F = order[i - 1], order[i]
            tau = float(np.clip(rng.normal(1.6 + 0.25 * i, 0.30), 0.5, 4.5))
            r = float(np.clip(rng.normal(0.83, 0.04), 0.60, 0.95))
            rows.append(dict(
                run=run, leader_name=L, follower_name=F,
                leader_pos=i, follower_pos=i + 1, pair=f"{L} \u2192 {F}",
                tau=tau, r=r,
                peak_accel=float(max(0.3, rng.normal(accel_mu[F], 0.25))),
                peak_decel_mag=float(max(0.3, rng.normal(decel_mu[F], 0.30)))))
            curves.append(dict(
                run=run, follower_name=F, leader_name=L, follower_pos=i + 1,
                pair=f"{L} \u2192 {F}", tau=tau, r=r,
                curve=_synthetic_curve(grid, tau, r, width=1.7, floor=0.60, rng=rng)))
    return pd.DataFrame(rows), grid, curves


def _self_test(outdir: str, min_r: float, lag_by: str = "follower") -> None:
    print("[self-test] fabricating 4 synthetic runs (two vehicle orders)")
    long, grid, curves = _fabricate_synthetic()
    outs = report_and_write(long, grid, curves, outdir, min_r,
                            prefix="selftest_", lag_by=lag_by)
    print(f"\n[self-test] wrote {len(outs)} output(s):")
    for o in outs:
        print(f"           {o}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default="../../Dataset/OpenACC/AstaZero",
                   help="AstaZero folder, a glob, or a single CSV.")
    p.add_argument("--outdir", default="astazero-outputs")
    p.add_argument("--min-r", type=float, default=0.0,
                   help="Min cross-correlation peak r for a run's tau* to count.")
    p.add_argument("--lag-by", choices=["follower", "pair"], default="follower",
                   help="Group the lag figure by follower make (default) or by "
                        "leader->follower pair. Peaks are always by follower make.")
    # segmentation knobs forwarded to analyze_pair (SI defaults from acb)
    p.add_argument("--penalty", type=float, default=55.0)
    p.add_argument("--min-seg", type=int, default=20)
    p.add_argument("--cusum-thresh", type=float, default=acb.SI_CUSUM_THRESH)
    p.add_argument("--cusum-drift", type=float, default=acb.SI_CUSUM_DRIFT)
    p.add_argument("--sg-window", type=int, default=15)
    p.add_argument("--sg-poly", type=int, default=2)
    p.add_argument("--tau-max", type=float, default=5.0)
    p.add_argument("--deadband", type=float, default=0.2)
    p.add_argument("--eps-stable", type=float, default=0.15)
    p.add_argument("--tmin", type=float, default=3.0)
    p.add_argument("--font-scale", type=float, default=1.0)
    p.add_argument("--self-test", action="store_true",
                   help="Render from fabricated multi-run data and exit.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    acb.set_font_scale(args.font_scale)
    os.makedirs(args.outdir, exist_ok=True)

    if args.self_test:
        _self_test(args.outdir, args.min_r, lag_by=args.lag_by)
        return

    files = resolve_files(args.folder)
    if not files:
        sys.exit("error: no ASta_*_platoon*.csv found; pass a folder/glob/CSV.")
    print(f"[aggregate] {len(files)} file(s), min_r={args.min_r}, "
          f"lag_by={args.lag_by}")

    analysis_args = default_analysis_args(dict(
        penalty=args.penalty, min_seg=args.min_seg,
        cusum_thresh=args.cusum_thresh, cusum_drift=args.cusum_drift,
        sg_window=args.sg_window, sg_poly=args.sg_poly, tau_max=args.tau_max,
        deadband=args.deadband, eps_stable=args.eps_stable, tmin=args.tmin))

    long, grid, curves = extract_all(files, analysis_args)
    if long.empty:
        sys.exit("error: no CF pairs extracted from any file.")
    outs = report_and_write(long, grid, curves, args.outdir, args.min_r,
                            lag_by=args.lag_by)

    if long["run"].nunique() == 1:
        print("\n[note] only 1 run present -> each group has n=1 (no spread). "
              "Point at the full AstaZero folder for the real aggregation.")
    print(f"\n[done] wrote {len(outs)} output(s):")
    for o in outs:
        print(f"        {o}")


if __name__ == "__main__":
    main()
