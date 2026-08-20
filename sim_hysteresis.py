#!/usr/bin/env python3
"""
sim_hysteresis.py
=================
Simulated speed-spacing hysteresis, ACC-equipped vs human-driven, *after*
phase-anchored calibration.

This is the after-calibration companion to ``cohort_properties.py`` (which
contrasts observed loops). Here the loops come from each cohort's calibrated
car-following model run open-loop -- the contrast a downstream CAV-impact
simulation actually consumes. It is also the sharpest test of the phase
objective's premise: because that objective is anchored at accel<->decel
boundaries, it fits exactly the turning points that generate the hysteresis
loop, so a phase-calibrated model should reproduce the loop, and the ACC-vs-HDV
loop difference, more faithfully than a pointwise fit.

Per cohort (HDV followers, ACC followers), one global theta is fit to that
cohort's TRAINING pairs with the pooled phase-anchored objective
(``calibrate_pairs(..., objective="phase", polish=False)``); the SAME model form
is used for both cohorts so the loop difference is a *parameter* effect. Each
cohort's held-out pairs are then free-simulated with its theta
(``evaluate.free_simulate``), the simulated follower is phase-segmented, and the
accel/decel branch separation Delta_s(v) is measured with the identical
``cohort_properties._hysteresis`` routine used on observed data. Collided /
barrier-clamped simulations are excluded from loop statistics and counted.

Modes
-----
* matched (default): each cohort's pairs under its own theta -> "simulated CAV
  vs simulated HDV".
* --counterfactual: additionally simulate every pair under *every* cohort's
  theta (tagged ``theta_source``), so a fixed set of leaders can be driven by
  both the ACC and the HDV parameter vector -- isolating the parameter effect
  from the leader/scenario confound.

Outputs (to --out): per-pair metrics, per-cohort summary (simulated + observed
loop width/area + calibrated params + J*), between-cohort contrast on the
simulated loop (Mann-Whitney U, bootstrap median-diff CI, Cliff's delta), the
fitted thetas as JSON, and a branch-curve figure overlaying simulated (solid)
and observed (dashed) hysteresis per cohort.

Units: pairs are SI (TGSIM native). CUSUM threshold/drift default to the m/s
values 2.1 / 0.3 (NGSIM ft/s 7.0 / 1.0 rescaled, decision D4) and drive BOTH the
observed segmentation feeding calibration and the simulated segmentation feeding
the loop. penalty is scale-invariant (normal_var cost).

Usage
-----
    python sim_hysteresis.py --root cf_tgsim --model idm --by cohort -o simhyst_out
    python sim_hysteresis.py --root cf_tgsim --by follower_type --counterfactual \
        --maxiter 80 --popsize 15 -o simhyst_out
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Restore np.trapz for downstream modules on NumPy 2.x (also done in
# cohort_properties / objectives; harmless if already present).
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid   # type: ignore[attr-defined]

from cf_data import load_folder, PairData
from cf_models import get_model
from calibrate import calibrate_pairs
from evaluate import free_simulate, pair_metrics
from phase_segmentation import segment_trajectory
from cohort_properties import _hysteresis, _cliffs_delta, _boot_median_diff

# feet-aware observed segmentation (native ft + tuned ft/s knobs); SI is handled
# locally with the m/s knobs.
try:
    from run_experiment import segment_pair as _segment_pair_feet
except Exception:                       # pragma: no cover
    _segment_pair_feet = None

CONTRAST_METRICS = ["dS_sim", "area_sim"]


@dataclass
class Cfg:
    features: Tuple[str, ...]
    penalty: float
    minseg: int
    cusum_threshold: float
    cusum_drift: float
    speed_bin: float
    maxiter: int
    popsize: int
    seed: int
    workers: int


# --------------------------------------------------------------------------- #
# segmentation helpers (one definition, feet-aware)
# --------------------------------------------------------------------------- #
def segment_obs(pair: PairData, cfg: Cfg):
    """Segment the OBSERVED follower. Feet pairs use the tuned native-ft path;
    SI pairs (TGSIM) use the m/s CUSUM knobs."""
    if pair.units_source.startswith("feet") and _segment_pair_feet is not None:
        return _segment_pair_feet(pair)
    return segment_trajectory(
        pair.t, pair.x_follower, pair.v_follower, pair.spacing,
        penalty=cfg.penalty, min_segment_length=cfg.minseg,
        cusum_threshold=cfg.cusum_threshold, cusum_drift=cfg.cusum_drift)


def loop_from_sim(sim, cfg: Cfg):
    """Segment a SIMULATED trajectory (always SI) and return its hysteresis."""
    seg = segment_trajectory(
        sim.t, sim.x, sim.v, sim.s,
        penalty=cfg.penalty, min_segment_length=cfg.minseg,
        cusum_threshold=cfg.cusum_threshold, cusum_drift=cfg.cusum_drift)
    return _hysteresis(sim.t, sim.v, sim.s, seg.phases, cfg.speed_bin)


# --------------------------------------------------------------------------- #
# contrasts on the simulated loop (unpaired, matched theta only)
# --------------------------------------------------------------------------- #
def contrasts(matched: pd.DataFrame, by: str, baseline: str,
              n_boot: int, seed: int) -> pd.DataFrame:
    try:
        from scipy.stats import mannwhitneyu
    except Exception:                   # pragma: no cover
        mannwhitneyu = None
    groups = sorted(matched[by].dropna().unique().tolist())
    if baseline and baseline in groups:
        pairs = [(baseline, g) for g in groups if g != baseline]
    else:
        pairs = [(groups[i], groups[j])
                 for i in range(len(groups)) for j in range(i + 1, len(groups))]
    rows: List[dict] = []
    for g1, g2 in pairs:
        for met in CONTRAST_METRICS:
            a = matched.loc[matched[by] == g1, met].dropna().to_numpy(float)
            b = matched.loc[matched[by] == g2, met].dropna().to_numpy(float)
            if len(a) == 0 or len(b) == 0:
                continue
            if mannwhitneyu is not None:
                try:
                    U, p = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    U, p = np.nan, np.nan
            else:
                U, p = np.nan, np.nan
            md, lo, hi = _boot_median_diff(a, b, n_boot, seed)
            rows.append(dict(metric=met, group_a=g1, group_b=g2,
                             n_a=len(a), n_b=len(b),
                             median_a=float(np.median(a)), median_b=float(np.median(b)),
                             median_diff=md, ci_lo=lo, ci_hi=hi,
                             cliffs_delta=_cliffs_delta(a, b),
                             mann_whitney_U=float(U), p_value=float(p)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figure: simulated (solid) vs observed (dashed) branches per cohort
# --------------------------------------------------------------------------- #
def _binmean(v: np.ndarray, s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan)
    for j in range(len(edges) - 1):
        m = (v >= edges[j]) & (v < edges[j + 1])
        if np.any(m):
            out[j] = float(np.mean(s[m]))
    return out


def _plot_branches(ax, pool: dict, w: float, color, solid: bool, label: Optional[str]):
    va, sa = pool["va"], pool["sa"]
    vd, sd = pool["vd"], pool["sd"]
    if len(va) < 2 or len(vd) < 2:
        return
    vlo = max(float(va.min()), float(vd.min()))
    vhi = min(float(va.max()), float(vd.max()))
    if not (vhi > vlo):
        return
    edges = np.arange(vlo, vhi + w, w)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ls = "-" if solid else "--"
    al = 1.0 if solid else 0.45
    ax.plot(ctr, _binmean(va, sa, edges), ls, color=color, alpha=al, marker="o",
            ms=3, label=(f"{label} accel" if label else None))
    ax.plot(ctr, _binmean(vd, sd, edges), ls, color=color, alpha=al, marker="s",
            ms=3, label=(f"{label} decel" if label else None))


def make_figure(sim_pool: Dict[str, dict], obs_pool: Dict[str, dict],
                speed_bin: float, overlay: bool, outdir: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as e:              # pragma: no cover
        print(f"[fig] skipped (matplotlib unavailable): {e}")
        return None
    groups = sorted(sim_pool.keys())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for gi, g in enumerate(groups):
        c = cmap(gi % 10)
        _plot_branches(ax, sim_pool[g], speed_bin, c, solid=True, label=g)
        if overlay and g in obs_pool:
            _plot_branches(ax, obs_pool[g], speed_bin, c, solid=False, label=None)
    ax.set_xlabel("follower speed (m/s)")
    ax.set_ylabel("net spacing (m)")
    ax.set_title("Speed-spacing hysteresis after calibration\n"
                 "solid = simulated, dashed = observed")
    handles, labels = ax.get_legend_handles_labels()
    if overlay:
        handles += [Line2D([0], [0], color="k", ls="-"),
                    Line2D([0], [0], color="k", ls="--", alpha=0.45)]
        labels += ["simulated", "observed"]
    ax.legend(handles, labels, fontsize=7, ncol=2)
    fig.tight_layout()
    p = os.path.join(outdir, "sim_hysteresis_branches.png")
    fig.savefig(p, dpi=300)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def run(args: argparse.Namespace) -> int:
    man_path = os.path.join(args.root, "manifest.csv")
    if not os.path.isfile(man_path):
        print(f"error: manifest not found: {man_path}")
        return 2
    man = pd.read_csv(man_path)
    if args.by not in man.columns:
        print(f"error: --by '{args.by}' not in manifest. cols: {list(man.columns)}")
        return 2
    man["_stem"] = man["path"].map(_stem)
    grp_map = dict(zip(man["_stem"], man[args.by]))

    cfg = Cfg(features=tuple(args.features.split(",")),
              penalty=args.penalty, minseg=args.min_segment_length,
              cusum_threshold=args.cusum_threshold, cusum_drift=args.cusum_drift,
              speed_bin=args.speed_bin, maxiter=args.maxiter, popsize=args.popsize,
              seed=args.seed, workers=args.workers)
    os.makedirs(args.out, exist_ok=True)
    model = get_model(args.model)

    # --- load & group pairs --------------------------------------------------
    if args.simset == "all":
        allp = load_folder(args.root)
        train_all, sim_all = allp, allp
    else:
        train_all = load_folder(args.root, split="train")
        sim_all = load_folder(args.root, split="test")

    def group(pairs: Sequence[PairData]) -> Dict[str, List[PairData]]:
        d: Dict[str, List[PairData]] = {}
        for p in pairs:
            g = grp_map.get(p.name)
            if g is not None:
                d.setdefault(g, []).append(p)
        return d

    train_by, sim_by = group(train_all), group(sim_all)
    groups = sorted(train_by.keys())
    if not groups:
        print("error: no training pairs mapped to groups.")
        return 1

    # --- per-cohort phase calibration ---------------------------------------
    thetas: Dict[str, List[float]] = {}
    calib: Dict[str, dict] = {}
    for g in groups:
        tp = train_by[g]
        segs = [segment_obs(p, cfg) for p in tp]
        res = calibrate_pairs(
            model, tp, "phase", segmentations=segs,
            obj_kwargs=dict(feature_keys=tuple(cfg.features)),
            maxiter=cfg.maxiter, popsize=cfg.popsize, seed=cfg.seed,
            polish=False, workers=cfg.workers)
        thetas[g] = [float(x) for x in res.theta]
        calib[g] = dict(param_dict={k: float(v) for k, v in res.param_dict.items()},
                        J_star=float(res.fun), n_train=len(tp), success=bool(res.success))
        print(f"[calib] {g:<20s} J*={res.fun:.4g}  n_train={len(tp)}  {res.param_dict}")

    # --- simulate & measure simulated loops ---------------------------------
    rows: List[dict] = []
    obs_cache: Dict[str, tuple] = {}
    sim_pool: Dict[str, dict] = {}
    obs_pool: Dict[str, dict] = {}

    def _simset_for(g: str) -> List[PairData]:
        s = sim_by.get(g, [])
        if not s:
            s = train_by.get(g, [])
            if s:
                print(f"[warn] cohort '{g}' has no test pairs; simulating its "
                      f"{len(s)} train pair(s).")
        return s

    for g in groups:
        for p in _simset_for(g):
            if p.name not in obs_cache:
                seg_o = segment_obs(p, cfg)
                obs_cache[p.name] = _hysteresis(
                    p.t, p.v_follower, p.spacing, seg_o.phases, cfg.speed_bin)
            dS_o, area_o, _nb_o, pool_o = obs_cache[p.name]

            srcs = [g] if not args.counterfactual else list(dict.fromkeys([g] + groups))
            for src in srcs:
                try:
                    sim = free_simulate(model, thetas[src], p)
                except Exception as e:          # pragma: no cover
                    rows.append(dict(cohort=g, theta_source=src, matched=(src == g),
                                     pair=p.name, failed=True, collided=None, n_barrier=None,
                                     dS_sim=np.nan, area_sim=np.nan, bins_sim=0,
                                     dS_obs=dS_o, area_obs=area_o, spacing_rmse=np.nan))
                    continue
                pm = pair_metrics(p, sim)
                if sim.collided:
                    dS_s = area_s = np.nan; nb_s = 0; pool_s = None
                else:
                    dS_s, area_s, nb_s, pool_s = loop_from_sim(sim, cfg)
                rows.append(dict(cohort=g, theta_source=src, matched=(src == g),
                                 pair=p.name, failed=False,
                                 collided=bool(sim.collided), n_barrier=int(sim.n_barrier),
                                 dS_sim=dS_s, area_sim=area_s, bins_sim=nb_s,
                                 dS_obs=dS_o, area_obs=area_o,
                                 spacing_rmse=pm["spacing_rmse"]))
                if src == g and not sim.collided and pool_s is not None:
                    sp = sim_pool.setdefault(g, {"va": [], "sa": [], "vd": [], "sd": []})
                    op = obs_pool.setdefault(g, {"va": [], "sa": [], "vd": [], "sd": []})
                    for k in ("va", "sa", "vd", "sd"):
                        sp[k].append(pool_s[k]); op[k].append(pool_o[k])

    if not rows:
        print("error: no simulations produced.")
        return 1
    per = pd.DataFrame(rows)
    sim_pool = {g: {k: (np.concatenate(v) if v else np.array([]))
                    for k, v in d.items()} for g, d in sim_pool.items()}
    obs_pool = {g: {k: (np.concatenate(v) if v else np.array([]))
                    for k, v in d.items()} for g, d in obs_pool.items()}

    # --- aggregate + write ---------------------------------------------------
    matched = per[per["matched"]].copy()

    def _q(x, q):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        return float(np.percentile(x, q)) if x.size else np.nan

    summ: List[dict] = []
    for g in groups:
        mg = matched[matched["cohort"] == g]
        coll = int(mg["collided"].fillna(False).sum())
        row = dict(cohort=g, n_sim=len(mg), n_collided=coll,
                   J_star=calib[g]["J_star"], n_train=calib[g]["n_train"])
        for metric, col in (("dS_sim", "dS_sim"), ("area_sim", "area_sim"),
                            ("dS_obs", "dS_obs"), ("area_obs", "area_obs"),
                            ("spacing_rmse", "spacing_rmse")):
            row[f"{metric}_median"] = float(np.nanmedian(mg[col])) if len(mg) else np.nan
            row[f"{metric}_q25"] = _q(mg[col], 25)
            row[f"{metric}_q75"] = _q(mg[col], 75)
        for k, v in calib[g]["param_dict"].items():
            row[f"param.{k}"] = v
        summ.append(row)
    summary = pd.DataFrame(summ)
    contr = contrasts(matched, "cohort", args.baseline, args.n_boot, args.seed)

    per_path = os.path.join(args.out, "sim_hysteresis_per_pair.csv")
    per.to_csv(per_path, index=False)
    summary.to_csv(os.path.join(args.out, "sim_hysteresis_cohort.csv"), index=False)
    contr.to_csv(os.path.join(args.out, "sim_hysteresis_contrasts.csv"), index=False)
    with open(os.path.join(args.out, "cohort_thetas.json"), "w") as fh:
        json.dump({g: {"theta": thetas[g], **calib[g]} for g in groups}, fh, indent=2)
    fig = None
    if not args.no_figures:
        fig = make_figure(sim_pool, obs_pool, cfg.speed_bin,
                          overlay=not args.no_overlay, outdir=args.out)

    # --- console -------------------------------------------------------------
    print(f"\nmodel={model.name}  simset={args.simset}  groups({args.by})={len(groups)}"
          f"  counterfactual={args.counterfactual}")
    print("simulated vs observed loop width (median Delta_s, m), matched theta:")
    for g in groups:
        mg = matched[matched["cohort"] == g]
        ds_s = float(np.nanmedian(mg["dS_sim"])) if len(mg) else np.nan
        ds_o = float(np.nanmedian(mg["dS_obs"])) if len(mg) else np.nan
        coll = int(mg["collided"].fillna(False).sum())
        print(f"  {g:<20s} sim={ds_s:7.3f}  obs={ds_o:7.3f}  "
              f"(n={len(mg)}, collided={coll})")
    if not contr.empty:
        print("\nbetween-cohort contrast on simulated loop width (dS_sim):")
        for _, r in contr[contr["metric"] == "dS_sim"].iterrows():
            print(f"  {r['group_a']} vs {r['group_b']}: "
                  f"median_diff={r['median_diff']:.3f} "
                  f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  delta={r['cliffs_delta']:.2f}")
    print(f"\nwrote: {per_path}")
    print(f"       {os.path.join(args.out, 'sim_hysteresis_cohort.csv')}")
    print(f"       {os.path.join(args.out, 'sim_hysteresis_contrasts.csv')}")
    print(f"       {os.path.join(args.out, 'cohort_thetas.json')}")
    if fig:
        print(f"       {fig}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="extract_cf_pairs output dir (manifest.csv + per-pair CSVs)")
    p.add_argument("--model", default="idm", help="idm | gipps | ovm")
    p.add_argument("--by", default="cohort",
                   help="cohort grouping column: cohort | follower_type | leader_type")
    p.add_argument("--baseline", default="HDV_follows_HDV",
                   help="if present, contrasts are vs this cohort; else all pairwise")
    p.add_argument("--simset", default="test", choices=["test", "all"],
                   help="'test' = calibrate on train, simulate held-out test "
                        "(falls back to a cohort's train pairs if it has no test); "
                        "'all' = calibrate and simulate on all pairs")
    p.add_argument("--counterfactual", action="store_true",
                   help="also simulate every pair under every cohort's theta")
    p.add_argument("--features", default="v_end,s_end",
                   help="phase feature keys for the objective")
    # DE knobs
    p.add_argument("--maxiter", type=int, default=60)
    p.add_argument("--popsize", type=int, default=15)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    # PELT+ knobs (penalty scale-invariant; threshold/drift m/s -> D4)
    p.add_argument("--penalty", type=float, default=75.0)
    p.add_argument("--min-segment-length", type=int, default=20)
    p.add_argument("--cusum-threshold", type=float, default=2.1)
    p.add_argument("--cusum-drift", type=float, default=0.3)
    # hysteresis
    p.add_argument("--speed-bin", type=float, default=0.5)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-overlay", action="store_true",
                   help="do not draw the observed-loop dashed reference")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("-o", "--out", default="simhyst_out")
    return p


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
