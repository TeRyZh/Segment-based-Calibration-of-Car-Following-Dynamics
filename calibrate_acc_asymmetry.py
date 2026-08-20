#!/usr/bin/env python3
"""
calibrate_acc_asymmetry.py   (basic IDM)
========================================
Pooled (control) vs regime-separated (treatment) calibration of the STANDARD IDM
on make-differentiated OpenACC pairs -- the ACC-asymmetry study, Mechanism B.

The Delayed-IDM route was dropped: on this ACC data a response-lag term did not
earn its complexity (pooled d-IDM under-produced the observed spacing-hysteresis
loop MORE than memoryless IDM, and the phase-switched d-IDM diverged in full-run
open loop). We therefore use the plain IDM (5 params: v0, T, a_max, b, s0) and ask
a narrower, better-posed question: does splitting the calibration by behavioural
regime expose an accel/decel asymmetry in the IDM parameters and improve fit?

  Phase 1  POOLED (control)          one IDM parameter set over the whole run.
  Phase 2  REGIME-SEPARATED (treat)  theta_dec on deceleration phases, theta_acc
                                     on acceleration phases, fit separately.

Regimes come from PELT+ critical points (SI CUSUM). Each phase is fit by an
open-loop ``simulate.simulate`` over that phase's slice, initialised at the
observed follower state at the phase start (IDM is memoryless, so no warm-start
buffer is needed). Objective: spacing RMSE. The SAME ACC-specific bounds are used
for all three fits, so any asymmetry (expected b_dec > b_acc, i.e. firmer
braking) is produced by the data, not the bounds.

Reframing note: with no lag term, IDM -- pooled or regime-separated -- still
under-produces the observed hysteresis loop width. That is a structural ceiling
of memoryless car-following (see the validation script), not a failure of this
calibration; the contribution here is the *parameter asymmetry* and the fit gain
from regime separation.

Outputs (to --output-dir)
    calib_f{pos}_{make}.json     theta_pooled / theta_dec / theta_acc, RMSEs,
                                 phase counts, dec-acc parameter deltas
    asymmetry_summary.csv        one flat row per pair
    calib_config.json            the exact arguments used
"""
from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# NumPy 2.x removed np.trapz (renamed np.trapezoid); phase_segmentation.py still
# calls np.trapz. Alias it BEFORE importing that module. (Renaming it in
# phase_segmentation.py / phase_multiscale.py is the permanent fix.)
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import pandas as pd
from scipy.optimize import differential_evolution

import simulate
from cf_models import get_model
from phase_segmentation import segment_trajectory

# ACC-specific IDM box (min-gap following). Defined here so the IDM class keeps
# its own NGSIM bounds for the highway pipeline. Same box for pooled + regimes.
ACC_IDM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "v0": (10.0, 40.0), "T": (0.3, 3.0), "a_max": (0.2, 3.0),
    "b": (0.2, 5.0), "s0": (0.5, 8.0),
}



# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _filesafe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def load_pair_arrays(csv_path: str) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    return {
        "t": df["t"].to_numpy(float),
        "x_f": df["x_follower"].to_numpy(float),
        "v_f": df["v_follower"].to_numpy(float),
        "a_f": df["a_follower"].to_numpy(float),
        "x_l": df["x_leader"].to_numpy(float),
        "v_l": df["v_leader"].to_numpy(float),
        "L": float(df["leader_length"].iloc[0]),
        "s": df["spacing"].to_numpy(float),
        "dv": df["dv"].to_numpy(float),
    }


def _meta_from_name(csv_path: str) -> Dict[str, object]:
    base = os.path.basename(csv_path)
    m = re.match(r"oa_pair_(?:.+_)?f(\d+)_(.+)\.csv$", base)
    if m:
        return {"follower_position": int(m.group(1)),
                "follower_make": m.group(2).replace("_", " "), "leader_make": ""}
    return {"follower_position": 0, "follower_make": os.path.splitext(base)[0],
            "leader_make": ""}


def load_pairs(args) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    inp = args.input
    out: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    if os.path.isdir(inp):
        man = pd.read_csv(os.path.join(inp, args.manifest_name))
        for _, row in man.iterrows():
            meta = row.to_dict()
            path = str(row.get("path", ""))
            local = os.path.join(inp, os.path.basename(path)) if path else ""
            csv = local if os.path.exists(local) else path
            if not os.path.exists(csv):
                print(f"  ! missing pair csv for f{meta.get('follower_position')}; skip")
                continue
            out.append((meta, load_pair_arrays(csv)))
    else:
        out.append((_meta_from_name(inp), load_pair_arrays(inp)))
    if args.only:
        keep = {p.strip() for p in str(args.only).split(",")}
        out = [(m, a) for (m, a) in out if str(m.get("follower_position")) in keep]
    return out


# --------------------------------------------------------------------------- #
# Segmentation -> regime phases
# --------------------------------------------------------------------------- #
def segment_regimes(A: Dict[str, object], args
                    ) -> Tuple[object, List[Tuple[int, int]], List[Tuple[int, int]]]:
    seg = segment_trajectory(
        A["t"], A["x_f"], A["v_f"], A["s"],
        penalty=args.penalty, min_segment_length=args.min_seg,
        cusum_threshold=args.cusum_threshold, cusum_drift=args.cusum_drift)
    v = A["v_f"]
    dec, acc = [], []
    for ph in seg.phases:
        i0, i1 = ph.i_start, ph.i_end
        if i1 <= i0:
            continue
        (dec if v[i1] < v[i0] else acc).append((i0, i1))
    return seg, dec, acc


# --------------------------------------------------------------------------- #
# Evaluation (spacing RMSE + barrier count) and objectives
# --------------------------------------------------------------------------- #
def pooled_eval(model, A: Dict[str, object], theta) -> Tuple[float, int, int]:
    r = simulate.simulate(model, theta, A["t"], A["x_l"], A["v_l"], A["L"],
                          float(A["x_f"][0]), float(A["v_f"][0]))
    resid = r.s - A["s"]
    return float(np.sqrt(np.mean(resid * resid))), int(r.n_barrier), int(len(A["t"]))


def regime_eval(model, A: Dict[str, object], phases: List[Tuple[int, int]], theta
                ) -> Tuple[float, int, int]:
    t, xl, vl, L = A["t"], A["x_l"], A["v_l"], A["L"]
    xf, vf, s = A["x_f"], A["v_f"], A["s"]
    sse = 0.0
    barr = 0
    total = 0
    for (i0, i1) in phases:
        sl = slice(i0, i1 + 1)
        r = simulate.simulate(model, theta, t[sl], xl[sl], vl[sl], L,
                              float(xf[i0]), float(vf[i0]))
        resid = r.s - s[sl]
        sse += float(np.dot(resid, resid))
        barr += int(r.n_barrier)
        total += (i1 - i0 + 1)
    return float(np.sqrt(sse / max(total, 1))), barr, total


# Module-level objectives (picklable via functools.partial -> --workers safe).
def _pooled_obj(theta, model, A, cp):
    rmse, nb, n = pooled_eval(model, A, theta)
    return rmse + cp * (nb / max(n, 1))


def _regime_obj(theta, model, A, phases, cp):
    rmse, nb, n = regime_eval(model, A, phases, theta)
    return rmse + cp * (nb / max(n, 1))


# --------------------------------------------------------------------------- #
# DE wrapper + helpers
# --------------------------------------------------------------------------- #
def acc_bounds(model) -> List[Tuple[float, float]]:
    return [ACC_IDM_BOUNDS[p] for p in model.param_names]


def _clip(theta, bounds) -> np.ndarray:
    return np.array([min(max(float(v), lo), hi) for v, (lo, hi) in zip(theta, bounds)],
                    dtype=float)


def _de(objective, bounds, args):
    params = inspect.signature(differential_evolution).parameters
    rng_kw = {"rng": args.seed} if "rng" in params else {"seed": args.seed}
    updating = "immediate" if args.workers == 1 else "deferred"
    return differential_evolution(
        objective, bounds, maxiter=args.maxiter, popsize=args.popsize,
        tol=args.tol, mutation=(0.5, 1.0), recombination=0.7, polish=False,
        workers=args.workers, updating=updating, **rng_kw)


# --------------------------------------------------------------------------- #
# Per-pair calibration
# --------------------------------------------------------------------------- #
def calibrate_pair(model, A: Dict[str, object], meta: Dict[str, object], args) -> Dict:
    pnames = list(model.param_names)                 # v0,T,a_max,b,s0
    bounds = acc_bounds(model)
    dt = float(A["t"][1] - A["t"][0])
    cp = args.collision_penalty

    seg, dec_ph, acc_ph = segment_regimes(A, args)
    print(f"    segmentation: {len(seg.critical_points)} critical points -> "
          f"{len(dec_ph)} decel / {len(acc_ph)} accel phases")

    # Phase 1 -- pooled (control)
    res_p = _de(functools.partial(_pooled_obj, model=model, A=A, cp=cp), bounds, args)
    th_p = _clip(res_p.x, bounds)
    rmse_p, nb_p, _ = pooled_eval(model, A, th_p)

    # Phase 2 -- regime-separated (treatment)
    def _fit(phases):
        if not phases:
            return None, float("nan"), 0
        res = _de(functools.partial(_regime_obj, model=model, A=A, phases=phases, cp=cp),
                  bounds, args)
        th = _clip(res.x, bounds)
        rmse, nb, _ = regime_eval(model, A, phases, th)
        return th, rmse, nb

    th_d, rmse_d, nb_d = _fit(dec_ph)
    th_a, rmse_a, nb_a = _fit(acc_ph)

    def _named(th):
        return None if th is None else {n: float(v) for n, v in zip(pnames, th)}

    idx = {k: pnames.index(k) for k in pnames}

    def _delta(k):
        if th_d is None or th_a is None:
            return float("nan")
        return float(th_d[idx[k]] - th_a[idx[k]])

    result = {
        "model": model.name,
        "follower_make": meta.get("follower_make", ""),
        "leader_make": meta.get("leader_make", ""),
        "follower_position": int(meta.get("follower_position", 0) or 0),
        "distance_setting": meta.get("distance_setting", ""),
        "n_frames": int(len(A["t"])), "dt": dt,
        "n_critical_points": int(len(seg.critical_points)),
        "n_decel_phases": len(dec_ph), "n_accel_phases": len(acc_ph),
        "theta_pooled": _named(th_p),
        "theta_dec": _named(th_d),
        "theta_acc": _named(th_a),
        "rmse_pooled_m": rmse_p, "rmse_dec_m": rmse_d, "rmse_acc_m": rmse_a,
        "n_barrier_pooled": nb_p, "n_barrier_dec": nb_d, "n_barrier_acc": nb_a,
        "d_b": _delta("b"), "d_T": _delta("T"), "d_a_max": _delta("a_max"),
        "d_s0": _delta("s0"), "d_v0": _delta("v0"),
        "bounds": {k: [lo, hi] for k, (lo, hi) in ACC_IDM_BOUNDS.items()},
        "collision_penalty": cp,
        "de": {"maxiter": args.maxiter, "popsize": args.popsize, "seed": args.seed,
               "polish": False},
    }
    return result


def _flatten(r: Dict) -> Dict:
    keep = ["model", "follower_make", "leader_make", "follower_position",
            "distance_setting", "n_frames", "n_critical_points",
            "n_decel_phases", "n_accel_phases",
            "rmse_pooled_m", "rmse_dec_m", "rmse_acc_m",
            "d_b", "d_T", "d_a_max", "d_s0", "d_v0"]
    flat = {k: r.get(k, float("nan")) for k in keep}
    for grp in ("pooled", "dec", "acc"):
        th = r.get(f"theta_{grp}")
        if th:
            for pn, val in th.items():
                flat[f"{pn}_{grp}"] = val
    return flat


def _print_pair(r: Dict) -> None:
    td, ta = r["theta_dec"], r["theta_acc"]
    print(f"    RMSE  pooled={r['rmse_pooled_m']:.3f}  dec={r['rmse_dec_m']:.3f}  "
          f"acc={r['rmse_acc_m']:.3f} m")
    if td and ta:
        print(f"    b   dec={td['b']:.2f}  acc={ta['b']:.2f}  (dec-acc={r['d_b']:+.2f})   "
              f"T  dec={td['T']:.2f}  acc={ta['T']:.2f}  (dec-acc={r['d_T']:+.2f})")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args) -> None:
    if args.self_test:
        _selftest(args)
        return
    if not args.input:
        sys.exit("error: an input (pairs directory or pair csv) is required.")

    pairs = load_pairs(args)
    if not pairs:
        sys.exit("No pairs loaded (check the manifest / --only filter).")
    model = get_model(args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for meta, A in pairs:
        pos = meta.get("follower_position")
        print(f"[f{pos}] calibrating {meta.get('follower_make','?')} "
              f"(leader {meta.get('leader_make','?')}), n={len(A['t'])} ...")
        r = calibrate_pair(model, A, meta, args)
        jpath = os.path.join(args.output_dir,
                             f"calib_f{r['follower_position']}_"
                             f"{_filesafe(r['follower_make'])}.json")
        with open(jpath, "w") as f:
            json.dump(r, f, indent=2)
        _print_pair(r)
        rows.append(_flatten(r))

    summary = pd.DataFrame(rows)
    spath = os.path.join(args.output_dir, "asymmetry_summary.csv")
    summary.to_csv(spath, index=False)
    with open(os.path.join(args.output_dir, "calib_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"Wrote {len(rows)} calibration(s) -> {args.output_dir}")
    cols = ["follower_make", "rmse_pooled_m", "rmse_dec_m", "rmse_acc_m",
            "b_dec", "b_acc", "d_b", "T_dec", "T_acc", "d_T"]
    have = [c for c in cols if c in summary.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[have].to_string(index=False))
    print(f"Summary: {spath}")
    print("=" * 70)


def _selftest(args) -> None:
    """Structural smoke test: synthesise a pair, confirm the pipeline runs and
    both regimes fit with finite RMSE (parameter recovery needs full --maxiter)."""
    import tempfile
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "oa_pair_synth_f2_Synth_Car.csv")
    model = get_model(args.model)
    dt, n = 0.1, 1000
    t = np.round(np.arange(n) * dt, 4)
    vL = np.clip(16 + 6 * np.sin(2 * np.pi * t / 40.0), 0, None)
    xL = np.concatenate([[300.0], 300.0 + np.cumsum(0.5 * (vL[1:] + vL[:-1]) * dt)])
    r = simulate.simulate(model, [28, 1.2, 1.5, 2.2, 2.5], t, xL, vL, 0.0,
                          xL[0] - 25.0, 16.0)
    pd.DataFrame({
        "t": t, "x_follower": r.x, "v_follower": r.v, "a_follower": r.a,
        "x_leader": xL, "v_leader": vL, "a_leader": np.gradient(vL, t),
        "leader_length": 0.0, "spacing": r.s, "dv": r.v - vL,
    }).to_csv(csv, index=False)

    A = load_pair_arrays(csv)
    args.maxiter, args.popsize, args.workers = 5, 5, 1
    res = calibrate_pair(model, A, _meta_from_name(csv), args)
    assert res["theta_pooled"] and res["theta_dec"] and res["theta_acc"], "missing fits"
    for g in ("theta_pooled", "theta_dec", "theta_acc"):
        assert set(res[g]) == set(model.param_names), f"{g} params mismatch"
    assert np.isfinite(res["rmse_pooled_m"]) and np.isfinite(res["rmse_dec_m"])
    assert res["n_decel_phases"] > 0 and res["n_accel_phases"] > 0, "no phases"
    _print_pair(res)
    print("\nSelf-test PASS: IDM pipeline runs, both regimes fit, output well-formed.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pooled vs regime-separated basic-IDM calibration for the ACC "
                    "asymmetry study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?",
                   help="OpenACC pairs directory (with manifest.csv) or a pair csv.")
    p.add_argument("-o", "--output-dir", default="acc_calib")
    p.add_argument("--model", default="idm", help="Car-following model (idm).")
    p.add_argument("--manifest-name", default="manifest.csv")
    p.add_argument("--only", default=None,
                   help="Comma-separated follower positions to calibrate (e.g. '2,3').")

    seg = p.add_argument_group("segmentation (PELT+ on the follower, SI)")
    seg.add_argument("--penalty", type=float, default=75.0)
    seg.add_argument("--min-seg", type=int, default=20)
    seg.add_argument("--cusum-threshold", type=float, default=2.1,
                     help="SI CUSUM threshold (m/s); 2.1 for OpenACC vs 7.0 for NGSIM.")
    seg.add_argument("--cusum-drift", type=float, default=0.3,
                     help="SI CUSUM drift (m/s); 0.3 for OpenACC vs 1.0 for NGSIM.")

    de = p.add_argument_group("differential evolution")
    de.add_argument("--maxiter", type=int, default=60)
    de.add_argument("--popsize", type=int, default=15)
    de.add_argument("--seed", type=int, default=42)
    de.add_argument("--tol", type=float, default=1e-6)
    de.add_argument("--workers", type=int, default=1,
                    help="DE parallel workers (-1 = all cores). Objectives picklable.")
    de.add_argument("--collision-penalty", type=float, default=5.0,
                    help="Added to RMSE as penalty*(barrier_frames/frames).")

    p.add_argument("--self-test", action="store_true",
                   help="Run on a synthetic pair (no real files needed).")
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
