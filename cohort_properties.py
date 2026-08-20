#!/usr/bin/env python3
"""
cohort_properties.py
====================
Model-free behavioural contrast between vehicle cohorts (e.g. ACC-equipped vs
human-driven followers) on phase-segmented car-following pairs.

The point of this module is *not* to calibrate a model. It uses the SAME PELT+
phase decomposition that anchors the calibration objective
(``phase_segmentation.segment_trajectory``) as a measurement instrument, and
reports two families of statistics that a car-following model must reproduce to
be credible for CAV-impact simulation:

  A. Phase-transition statistics  (oscillation behaviour -> flow efficiency)
       critical-point rate (per minute, per km), per-phase speed amplitude and
       rate split by accel/decel, the accel<->decel asymmetry, and the transient
       (non-quasi-steady) time fraction; plus phase-localised surrogate-safety
       measures computed on *closing* samples -- minimum TTC, maximum DRAC, peak
       deceleration, and the hard-deceleration-phase rate.

  B. Hysteresis (speed-spacing loops)  (capacity drop / approach tightness)
       the accel-branch vs decel-branch spacing separation Delta_s(v) at matched
       speed (loop width), and a signed loop-area proxy between the two branch
       curves.

Cohorts come from the ``extract_cf_pairs`` manifest (``cohort``,
``follower_type``, ``leader_type``, ``is_av_follower``, ``is_av_leader``).
Between-cohort tests are UNPAIRED (different vehicles), so contrasts use the
Mann-Whitney U test with a bootstrap CI on the median difference and Cliff's
delta effect size. Given the small number of ACC followers, CIs and effect
sizes are the headline; p-values are secondary.

Units: TGSIM is native SI (m, m/s). PELT+'s ``normal_var`` cost makes ``penalty``
scale-invariant, but the CUSUM ``threshold``/``drift`` are in velocity units and
have been rescaled here from the NGSIM ft/s tuning (7.0 / 1.0) to the m/s
starting values 2.1 / 0.3 (decision D4). Sweep them on real data before trusting
absolute phase counts; cohort *contrasts* are far less sensitive to the exact
knob than absolute rates are.

Usage
-----
    python cohort_properties.py --root cf_tgsim --by cohort -o cohort_out
    python cohort_properties.py --root cf_tgsim --by follower_type \
        --cusum-threshold 2.1 --cusum-drift 0.3 -o cohort_out
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# NumPy 2.x removed np.trapz in favour of np.trapezoid. Restore the alias so
# this module and the downstream phase_segmentation run on either NumPy.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid   # type: ignore[attr-defined]

from phase_segmentation import segment_trajectory, SegmentationResult

FT = 0.3048  # feet -> metres

# Metrics summarised for every group.
SUMMARY_METRICS = [
    "cp_rate_min", "cp_rate_km", "n_phases",
    "mean_accel_rate", "mean_decel_rate", "asym_rate",
    "mean_accel_dur", "mean_decel_dur", "asym_dur",
    "mean_accel_dv", "mean_decel_dv", "transient_frac",
    "min_ttc", "max_drac", "peak_decel", "hard_decel_rate_min",
    "hyst_dS", "hyst_loop_area",
]
# Shorter list for pairwise between-cohort contrasts (keeps the CSV readable).
CONTRAST_METRICS = [
    "cp_rate_min", "asym_rate", "mean_decel_rate",
    "hyst_dS", "min_ttc", "peak_decel",
]


# --------------------------------------------------------------------------- #
# small numerics
# --------------------------------------------------------------------------- #
def _ols_slope(t: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on t (the phase's accel/decel rate)."""
    if len(t) < 2:
        return 0.0
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tc = t - t.mean()
    den = float(np.sum(tc * tc))
    if den <= 0.0:
        return 0.0
    return float(np.sum(tc * (y - y.mean())) / den)


def _mean(x: Sequence[float]) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def _hysteresis(t: np.ndarray, v: np.ndarray, s: np.ndarray,
                phases, speed_bin: float
                ) -> Tuple[float, float, int, Dict[str, np.ndarray]]:
    """Accel- vs decel-branch spacing separation Delta_s(v).

    Label every sample by the kind of the phase it belongs to, then, over the
    speed range covered by *both* branches, compare mean spacing per speed bin.
    Returns (count-weighted mean Delta_s, signed loop-area proxy, n matched
    bins, pooled branch samples for plotting).
    """
    branch = np.array(["steady"] * len(t), dtype=object)
    for ph in phases:
        if ph.kind in ("accel", "decel"):
            branch[ph.i_start:ph.i_end + 1] = ph.kind
    am = branch == "accel"
    dm = branch == "decel"
    pool = {"va": v[am], "sa": s[am], "vd": v[dm], "sd": s[dm]}
    if am.sum() < 2 or dm.sum() < 2:
        return np.nan, np.nan, 0, pool

    vlo = max(float(np.min(v[am])), float(np.min(v[dm])))
    vhi = min(float(np.max(v[am])), float(np.max(v[dm])))
    if not (vhi > vlo):
        return np.nan, np.nan, 0, pool

    edges = np.arange(vlo, vhi + speed_bin, speed_bin)
    dS: List[float] = []
    wts: List[float] = []
    dvw: List[float] = []
    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j + 1]
        aa = am & (v >= lo) & (v < hi)
        dd = dm & (v >= lo) & (v < hi)
        if aa.sum() >= 1 and dd.sum() >= 1:
            dS.append(float(np.mean(s[dd]) - np.mean(s[aa])))
            wts.append(float(aa.sum() + dd.sum()))
            dvw.append(float(hi - lo))
    if not dS:
        return np.nan, np.nan, 0, pool
    dS_arr = np.asarray(dS)
    dS_agg = float(np.average(dS_arr, weights=np.asarray(wts)))
    area = float(np.sum(dS_arr * np.asarray(dvw)))   # integral (s_d - s_a) dv
    return dS_agg, area, len(dS), pool


# --------------------------------------------------------------------------- #
# per-pair metrics
# --------------------------------------------------------------------------- #
def per_pair_metrics(df: pd.DataFrame, seg: SegmentationResult, cfg: dict
                     ) -> Tuple[dict, Dict[str, np.ndarray]]:
    t = df["t"].to_numpy(float)
    v = df["v_follower"].to_numpy(float)
    a = df["a_follower"].to_numpy(float)
    s = df["spacing"].to_numpy(float)
    dv = df["dv"].to_numpy(float)          # +ve = closing (follower faster)
    n = len(t)
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    path = float(np.trapz(v, t)) if n > 1 else 0.0

    acc_rate: List[float] = []
    dec_rate: List[float] = []
    acc_dur: List[float] = []
    dec_dur: List[float] = []
    acc_dv: List[float] = []
    dec_dv: List[float] = []
    dec_min_a: List[float] = []
    transient = 0.0
    for ph in seg.phases:
        i0, i1 = ph.i_start, ph.i_end
        if i1 <= i0:
            continue
        slope = _ols_slope(t[i0:i1 + 1], v[i0:i1 + 1])
        amp = abs(float(v[i1] - v[i0]))
        if abs(slope) > cfg["slope_eps"]:
            transient += ph.duration
        if ph.kind == "accel":
            acc_rate.append(abs(slope)); acc_dur.append(ph.duration); acc_dv.append(amp)
        elif ph.kind == "decel":
            dec_rate.append(abs(slope)); dec_dur.append(ph.duration); dec_dv.append(amp)
            dec_min_a.append(float(np.nanmin(a[i0:i1 + 1])))

    m_acc_rate = _mean(acc_rate)
    m_dec_rate = _mean(dec_rate)
    asym_rate = (m_dec_rate / m_acc_rate) if (acc_rate and m_acc_rate > 0) else np.nan
    m_acc_dur, m_dec_dur = _mean(acc_dur), _mean(dec_dur)
    asym_dur = (m_dec_dur / m_acc_dur) if (acc_dur and m_acc_dur > 0) else np.nan
    hard_ph = sum(1 for mn in dec_min_a if mn < cfg["hard_decel"])

    closing = (dv > cfg["dv_eps"]) & (s > 0.0)
    if closing.any():
        ttc = s[closing] / dv[closing]
        drac = dv[closing] ** 2 / (2.0 * s[closing])
        min_ttc = float(np.min(ttc))
        max_drac = float(np.max(drac))
    else:
        min_ttc = np.nan
        max_drac = np.nan
    peak_decel = float(-np.nanmin(a)) if n else np.nan

    hyst_dS, hyst_area, nbins, pool = _hysteresis(
        t, v, s, seg.phases, cfg["speed_bin"])

    row = dict(
        n_points=n, duration_s=dur, path_m=path,
        n_phases=len(seg.phases), n_critical=len(seg.critical_points),
        cp_rate_min=(len(seg.critical_points) / (dur / 60.0)) if dur > 0 else np.nan,
        cp_rate_km=(len(seg.critical_points) / (path / 1000.0)) if path > 0 else np.nan,
        mean_accel_rate=m_acc_rate, mean_decel_rate=m_dec_rate, asym_rate=asym_rate,
        mean_accel_dur=m_acc_dur, mean_decel_dur=m_dec_dur, asym_dur=asym_dur,
        mean_accel_dv=_mean(acc_dv), mean_decel_dv=_mean(dec_dv),
        transient_frac=(transient / dur) if dur > 0 else np.nan,
        min_ttc=min_ttc, max_drac=max_drac, peak_decel=peak_decel,
        hard_decel_rate_min=(hard_ph / (dur / 60.0)) if dur > 0 else np.nan,
        hyst_dS=hyst_dS, hyst_loop_area=hyst_area, hyst_bins=nbins,
    )
    return row, pool


# --------------------------------------------------------------------------- #
# between-cohort contrasts (unpaired)
# --------------------------------------------------------------------------- #
def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / (len(a) * len(b)))


def _boot_median_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int
                      ) -> Tuple[float, float, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float(np.median(a) - np.median(b)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        d[k] = (np.median(rng.choice(a, len(a), replace=True))
                - np.median(rng.choice(b, len(b), replace=True)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.median(a) - np.median(b)), float(lo), float(hi)


def contrasts(per_pair: pd.DataFrame, by: str, baseline: str,
              n_boot: int, seed: int) -> pd.DataFrame:
    try:
        from scipy.stats import mannwhitneyu
    except Exception:                     # pragma: no cover
        mannwhitneyu = None

    groups = sorted(per_pair[by].dropna().unique().tolist())
    if baseline and baseline in groups:
        pairs = [(baseline, g) for g in groups if g != baseline]
    else:
        pairs = [(groups[i], groups[j])
                 for i in range(len(groups)) for j in range(i + 1, len(groups))]

    rows: List[dict] = []
    for g1, g2 in pairs:
        for met in CONTRAST_METRICS:
            a = per_pair.loc[per_pair[by] == g1, met].dropna().to_numpy(float)
            b = per_pair.loc[per_pair[by] == g2, met].dropna().to_numpy(float)
            if len(a) == 0 or len(b) == 0:
                continue
            if mannwhitneyu is not None and len(a) >= 1 and len(b) >= 1:
                try:
                    U, p = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    U, p = np.nan, np.nan
            else:
                U, p = np.nan, np.nan
            md, lo, hi = _boot_median_diff(a, b, n_boot, seed)
            rows.append(dict(
                metric=met, group_a=g1, group_b=g2, n_a=len(a), n_b=len(b),
                median_a=float(np.median(a)), median_b=float(np.median(b)),
                median_diff=md, ci_lo=lo, ci_hi=hi,
                cliffs_delta=_cliffs_delta(a, b), mann_whitney_U=float(U), p_value=float(p),
            ))
    return pd.DataFrame(rows)


def summarise(per_pair: pd.DataFrame, by: str) -> pd.DataFrame:
    rows: List[dict] = []
    for g, sub in per_pair.groupby(by):
        for met in SUMMARY_METRICS:
            x = sub[met].dropna().to_numpy(float)
            rows.append(dict(
                group=g, metric=met, n=int(x.size),
                median=float(np.median(x)) if x.size else np.nan,
                q25=float(np.percentile(x, 25)) if x.size else np.nan,
                q75=float(np.percentile(x, 75)) if x.size else np.nan,
                mean=float(np.mean(x)) if x.size else np.nan,
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures (optional; guarded so a plotting hiccup never loses the CSVs)
# --------------------------------------------------------------------------- #
def _figures(per_pair: pd.DataFrame, pools: Dict[str, dict], by: str,
             speed_bin: float, outdir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: List[str] = []
    groups = sorted(per_pair[by].dropna().unique().tolist())

    # 1) critical-point rate by group
    try:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        data = [per_pair.loc[per_pair[by] == g, "cp_rate_min"].dropna().to_numpy()
                for g in groups]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=20)
        ax.set_ylabel("critical-point rate (per min)")
        ax.set_title("Phase-transition rate by cohort")
        fig.tight_layout()
        p = os.path.join(outdir, "phase_cp_rate.png")
        fig.savefig(p, dpi=300); plt.close(fig); made.append(p)
    except Exception as e:      # pragma: no cover
        print(f"[fig] phase_cp_rate skipped: {e}")

    # 2) hysteresis: accel vs decel mean spacing branches per group
    try:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        cmap = plt.get_cmap("tab10")
        for gi, g in enumerate(groups):
            va = pools[g]["va"]; sa = pools[g]["sa"]
            vd = pools[g]["vd"]; sd = pools[g]["sd"]
            if len(va) < 2 or len(vd) < 2:
                continue
            vlo = max(va.min(), vd.min()); vhi = min(va.max(), vd.max())
            if not (vhi > vlo):
                continue
            edges = np.arange(vlo, vhi + speed_bin, speed_bin)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            ma = [np.mean(sa[(va >= edges[j]) & (va < edges[j + 1])])
                  if np.any((va >= edges[j]) & (va < edges[j + 1])) else np.nan
                  for j in range(len(edges) - 1)]
            md = [np.mean(sd[(vd >= edges[j]) & (vd < edges[j + 1])])
                  if np.any((vd >= edges[j]) & (vd < edges[j + 1])) else np.nan
                  for j in range(len(edges) - 1)]
            c = cmap(gi % 10)
            ax.plot(ctr, ma, "-o", ms=3, color=c, label=f"{g} accel")
            ax.plot(ctr, md, "--s", ms=3, color=c, label=f"{g} decel")
        ax.set_xlabel("follower speed (m/s)")
        ax.set_ylabel("net spacing (m)")
        ax.set_title("Speed-spacing hysteresis branches")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        p = os.path.join(outdir, "hysteresis_branches.png")
        fig.savefig(p, dpi=300); plt.close(fig); made.append(p)
    except Exception as e:      # pragma: no cover
        print(f"[fig] hysteresis_branches skipped: {e}")

    # 3) safety: ECDF of minimum TTC by group
    try:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for g in groups:
            x = per_pair.loc[per_pair[by] == g, "min_ttc"].dropna().to_numpy()
            if x.size == 0:
                continue
            xs = np.sort(x)
            ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", label=g)
        ax.set_xlabel("minimum TTC in closing phases (s)")
        ax.set_ylabel("ECDF")
        ax.set_title("Surrogate safety by cohort")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, "safety_min_ttc.png")
        fig.savefig(p, dpi=300); plt.close(fig); made.append(p)
    except Exception as e:      # pragma: no cover
        print(f"[fig] safety_min_ttc skipped: {e}")

    return made


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _resolve_units(manifest: pd.DataFrame, override: str) -> str:
    if override in ("si", "feet", "ft"):
        return "m" if override == "si" else "ft"
    if "units" in manifest.columns and manifest["units"].notna().any():
        u = str(manifest["units"].dropna().iloc[0]).lower()
        return "ft" if u.startswith("f") else "m"
    return "m"   # TGSIM default


def run(args: argparse.Namespace) -> int:
    man_path = os.path.join(args.root, args.manifest)
    if not os.path.isfile(man_path):
        print(f"error: manifest not found: {man_path}")
        return 2
    manifest = pd.read_csv(man_path)
    if args.by not in manifest.columns:
        print(f"error: --by '{args.by}' not a manifest column. "
              f"available: {list(manifest.columns)}")
        return 2

    units = _resolve_units(manifest, args.units)
    scale = FT if units == "ft" else 1.0
    cfg = dict(speed_bin=args.speed_bin, slope_eps=args.slope_eps,
               hard_decel=args.hard_decel, dv_eps=args.dv_eps)
    os.makedirs(args.out, exist_ok=True)

    rows: List[dict] = []
    pools: Dict[str, dict] = {}
    n_ok = n_skip = 0
    for _, meta in manifest.iterrows():
        rel = meta.get("path")
        if not isinstance(rel, str):
            n_skip += 1; continue
        fp = os.path.join(args.root, rel)
        if not os.path.isfile(fp):
            n_skip += 1; continue
        df = pd.read_csv(fp)
        if len(df) < args.min_points:
            n_skip += 1; continue
        # to SI for physically-meaningful safety metrics + m/s CUSUM knobs
        if scale != 1.0:
            for c in ("x_follower", "v_follower", "a_follower", "x_leader",
                      "v_leader", "a_leader", "leader_length", "spacing", "dv"):
                if c in df:
                    df[c] = df[c] * scale

        t = df["t"].to_numpy(float)
        seg = segment_trajectory(
            t, df["x_follower"].to_numpy(float), df["v_follower"].to_numpy(float),
            df["spacing"].to_numpy(float),
            penalty=args.penalty, min_segment_length=args.min_segment_length,
            cusum_threshold=args.cusum_threshold, cusum_drift=args.cusum_drift)

        row, pool = per_pair_metrics(df, seg, cfg)
        grp = meta.get(args.by)
        row["group"] = grp
        row[args.by] = grp                      # guarantee the grouping column
        row["pair_id"] = os.path.splitext(os.path.basename(rel))[0]
        for k in ("cohort", "follower_type", "leader_type",
                  "is_av_follower", "is_av_leader", "split",
                  "vehicle_model", "leader_model", "campaign", "headway_setting"):
            if k in manifest.columns:
                row[k] = meta.get(k)
        rows.append(row)

        pl = pools.setdefault(grp, {"va": [], "sa": [], "vd": [], "sd": []})
        for k in ("va", "sa", "vd", "sd"):
            pl[k].append(pool[k])
        n_ok += 1

    if not rows:
        print("error: no usable pairs found under --root.")
        return 1
    per_pair = pd.DataFrame(rows)
    pools = {g: {k: (np.concatenate(v) if v else np.array([]))
                 for k, v in d.items()} for g, d in pools.items()}

    per_pair_path = os.path.join(args.out, "cohort_metrics_per_pair.csv")
    per_pair.to_csv(per_pair_path, index=False)
    summary = summarise(per_pair, args.by)
    summary.to_csv(os.path.join(args.out, "cohort_summary.csv"), index=False)
    contr = contrasts(per_pair, args.by, args.baseline, args.n_boot, args.seed)
    contr.to_csv(os.path.join(args.out, "cohort_contrasts.csv"), index=False)

    figs: List[str] = []
    if not args.no_figures:
        figs = _figures(per_pair, pools, args.by, args.speed_bin, args.out)

    # --- console summary -----------------------------------------------------
    print(f"units={units}  pairs used={n_ok}  skipped={n_skip}  "
          f"groups({args.by})={per_pair[args.by].nunique()}")
    counts = per_pair[args.by].value_counts().to_dict()
    print("cohort counts:", counts)
    print("\nmedian of two headline metrics by cohort:")
    for g, sub in per_pair.groupby(args.by):
        cp = sub["cp_rate_min"].median()
        ds = sub["hyst_dS"].median()
        print(f"  {g:<20s}  cp_rate_min={cp:6.2f}   hyst_dS={ds:7.3f} m   (n={len(sub)})")
    print(f"\nwrote: {per_pair_path}")
    print(f"       {os.path.join(args.out, 'cohort_summary.csv')}")
    print(f"       {os.path.join(args.out, 'cohort_contrasts.csv')}")
    for p in figs:
        print(f"       {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="extract_cf_pairs output dir (manifest.csv + per-pair CSVs)")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--by", default="cohort",
                   help="grouping column: cohort | follower_type | leader_type")
    p.add_argument("--baseline", default="HDV_follows_HDV",
                   help="if present in the data, contrasts are computed vs this "
                        "group only; otherwise all pairwise contrasts are done")
    p.add_argument("--units", default="auto", choices=["auto", "si", "feet", "ft"],
                   help="auto reads the manifest 'units' column (TGSIM -> SI)")
    # PELT+ knobs (penalty scale-invariant; threshold/drift are m/s -> D4)
    p.add_argument("--penalty", type=float, default=50.0)
    p.add_argument("--min-segment-length", type=int, default=20)
    p.add_argument("--cusum-threshold", type=float, default=5,
                   help="m/s; = NGSIM 7.0 ft/s rescaled. Sweep on real data.")
    p.add_argument("--cusum-drift", type=float, default=0.1,
                   help="m/s; = NGSIM 1.0 ft/s rescaled.")
    # metric knobs
    p.add_argument("--speed-bin", type=float, default=0.5,
                   help="speed-bin width (m/s) for hysteresis branch matching")
    p.add_argument("--slope-eps", type=float, default=0.1,
                   help="|dv/dt| (m/s^2) above which a phase counts as transient")
    p.add_argument("--hard-decel", type=float, default=-3.0,
                   help="a_follower (m/s^2) below which a decel phase is 'hard'")
    p.add_argument("--dv-eps", type=float, default=0.1,
                   help="closing-speed threshold (m/s) for TTC/DRAC samples")
    p.add_argument("--min-points", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("-o", "--out", default="cohort_out")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
