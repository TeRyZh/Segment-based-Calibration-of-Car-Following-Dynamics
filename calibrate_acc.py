#!/usr/bin/env python3
"""
calibrate_acc.py
================
Calibrate ONE IDM parameter set per make on a SINGLE OpenACC run, using a
train/held-out temporal split, under the two competing objectives:

  sample    pointwise: spacing RMSE at EVERY timestep of the open-loop simulation.
  segment   segment-based (Mechanism A): the follower is simulated continuously,
            but spacing error is evaluated ONLY at the observed PELT+ critical
            points (accel<->decel phase boundaries). '--objective phase' is
            accepted as a backward-compatible alias for 'segment'.

Design (per the study):
  * ONE run only (select with --run if the pairs directory holds several); one
    parameter set per make -- never run-specific parameters.
  * TRAIN on the first --train-frac of each pair's trajectory; HELD-OUT is the
    remaining tail. Every fitted theta is evaluated on BOTH segments and BOTH
    metrics, so the head-line number is generalisation: held-out spacing RMSE
    (full and at held-out critical points) for the sample vs segment fits.
  * Wider-than-human ACC bounds; no decel/accel split.

Outputs (to --output-dir)
    calib_f{pos}_{make}.json   theta_sample / theta_segment (+ theta_phase alias)
    calib_summary.csv          one flat row per make (held-out metrics up front)
    calib_config.json
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

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import pandas as pd
from scipy.optimize import differential_evolution

import simulate
from cf_models import get_model
from phase_segmentation import segment_trajectory

ACC_IDM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "v0": (5.0, 45.0), "T": (0.2, 3.0), "a_max": (0.2, 4.0),
    "b": (0.2, 10.0), "s0": (0.1, 8.0),
}

ACC_BOUNDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "idm":  {"v0": (5.0, 45.0), "T": (0.2, 3.0), "a_max": (0.2, 4.0),
             "b": (0.2, 6.0), "s0": (0.1, 8.0)},
    "ovm":  {"kappa": (0.1, 3.0), "v_max": (5.0, 50.0), "s_c": (2.0, 40.0),
             "w": (1.0, 50.0)},
    "fvdm": {"kappa": (0.1, 3.0), "v_max": (5.0, 50.0), "s_c": (2.0, 40.0),
             "w": (1.0, 50.0), "lambda": (0.0, 5.0)},
}

OBJECTIVES = ("sample", "segment")   # 'segment' is the segment-based objective

# --------------------------------------------------------------------------- #
# IO helpers (shared with the validator)
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


def slice_arrays(A: Dict[str, object], i0: int, i1: int) -> Dict[str, object]:
    return {k: (v[i0:i1].copy() if isinstance(v, np.ndarray) else v) for k, v in A.items()}


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
        if args.run:
            man = man[man["run"].astype(str).str.contains(str(args.run))]
        runs = sorted(set(man["run"].astype(str))) if "run" in man.columns else []
        if len(runs) > 1:
            sys.exit(f"Multiple runs found ({runs}); this study uses ONE run. "
                     f"Select one with --run (e.g. --run {runs[0]}).")
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
# Critical points + evaluation
# --------------------------------------------------------------------------- #
def critical_points(A: Dict[str, object], args) -> np.ndarray:
    seg = segment_trajectory(
        A["t"], A["x_f"], A["v_f"], A["s"],
        penalty=args.penalty, min_segment_length=args.min_seg,
        cusum_threshold=args.cusum_threshold, cusum_drift=args.cusum_drift)
    return np.asarray(seg.critical_points, dtype=int)


def sim_eval(model, Aseg: Dict[str, object], theta, cp: np.ndarray
             ) -> Tuple[float, float, int]:
    """Open-loop sim over one segment; return (full RMSE, critical-point RMSE, barriers)."""
    r = simulate.simulate(model, theta, Aseg["t"], Aseg["x_l"], Aseg["v_l"], Aseg["L"],
                          float(Aseg["x_f"][0]), float(Aseg["v_f"][0]))
    resid = r.s - Aseg["s"]
    rmse_full = float(np.sqrt(np.mean(resid * resid)))
    rmse_cp = float(np.sqrt(np.mean(resid[cp] ** 2))) if cp.size else float("nan")
    return rmse_full, rmse_cp, int(r.n_barrier)


def _obj(theta, model, Atrain, cp_train, penalty, which):
    rf, rc, nb = sim_eval(model, Atrain, theta, cp_train)
    base = rf if which == "sample" else rc
    return base + penalty * (nb / max(len(Atrain["t"]), 1))


# --------------------------------------------------------------------------- #
# DE + helpers
# --------------------------------------------------------------------------- #
# def acc_bounds(model) -> List[Tuple[float, float]]:
#     return [ACC_IDM_BOUNDS[p] for p in model.param_names]

def acc_bounds(model) -> List[Tuple[float, float]]:
    if model.name not in ACC_BOUNDS:
        sys.exit(f"No ACC bounds for model '{model.name}'; add them to ACC_BOUNDS "
                 f"(known: {sorted(ACC_BOUNDS)}).")
    b = ACC_BOUNDS[model.name]
    return [b[p] for p in model.param_names]


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
# Per-make calibration (train fit + train/held-out cross-eval)
# --------------------------------------------------------------------------- #
def calibrate_pair(model, A, meta, args, which_list) -> Dict:
    pnames = list(model.param_names)
    bounds = acc_bounds(model)

    n = len(A["t"])
    split = int(round(args.train_frac * n))
    split = min(max(split, args.min_seg + 1), n - (args.min_seg + 1))   # keep both usable
    Atr = slice_arrays(A, 0, split)
    Aho = slice_arrays(A, split, n)
    cp_tr = critical_points(Atr, args)
    cp_ho = critical_points(Aho, args)

    thetas: Dict[str, Optional[np.ndarray]] = {"sample": None, "segment": None}
    for which in which_list:
        res = _de(functools.partial(_obj, model=model, Atrain=Atr, cp_train=cp_tr,
                                    penalty=args.collision_penalty, which=which),
                  bounds, args)
        thetas[which] = _clip(res.x, bounds)

    evals: Dict[str, dict] = {}
    for which, th in thetas.items():
        if th is None:
            continue
        rf_tr, rc_tr, nb_tr = sim_eval(model, Atr, th, cp_tr)
        rf_ho, rc_ho, nb_ho = sim_eval(model, Aho, th, cp_ho)
        evals[which] = {
            "train": {"rmse_full_m": rf_tr, "rmse_cp_m": rc_tr, "n_barrier": nb_tr},
            "heldout": {"rmse_full_m": rf_ho, "rmse_cp_m": rc_ho, "n_barrier": nb_ho}}

    def _named(th):
        return None if th is None else {k: float(v) for k, v in zip(pnames, th)}

    return {
        "model": model.name,
        "follower_make": meta.get("follower_make", ""),
        "leader_make": meta.get("leader_make", ""),
        "follower_position": int(meta.get("follower_position", 0) or 0),
        "distance_setting": meta.get("distance_setting", ""),
        "run": meta.get("run", ""), "pair_id": meta.get("pair_id", ""),
        "dt": float(A["t"][1] - A["t"][0]),
        "split": {"train_frac": args.train_frac, "n": n, "n_train": split,
                  "n_heldout": n - split, "split_index": split,
                  "n_cp_train": int(cp_tr.size), "n_cp_heldout": int(cp_ho.size)},
        "theta_sample": _named(thetas["sample"]),
        "theta_segment": _named(thetas["segment"]),
        "theta_phase": _named(thetas["segment"]),   # backward-compat alias

        "eval": evals,
        "bounds": {k: [lo, hi] for k, (lo, hi) in ACC_BOUNDS[model.name].items()},
        "de": {"maxiter": args.maxiter, "popsize": args.popsize, "seed": args.seed,
               "polish": False},
    }


def _flatten(r: Dict) -> Dict:
    sp = r.get("split", {})
    flat = {"model": r.get("model"), "follower_make": r.get("follower_make"),
            "follower_position": r.get("follower_position"), "run": r.get("run"),
            "n_train": sp.get("n_train"), "n_heldout": sp.get("n_heldout"),
            "n_cp_heldout": sp.get("n_cp_heldout")}
    ev = r.get("eval", {})
    for which in OBJECTIVES:
        e = ev.get(which, {})
        for seg in ("heldout", "train"):
            s = e.get(seg, {})
            flat[f"rmse_full_{seg}_{which}"] = s.get("rmse_full_m")
            flat[f"rmse_cp_{seg}_{which}"] = s.get("rmse_cp_m")
        th = r.get(f"theta_{which}")
        if th:
            for pn, v in th.items():
                flat[f"{pn}_{which}"] = v
    # backward-compat: mirror every new 'segment' column under the legacy 'phase' suffix
    for key in [k for k in flat if k.endswith("_segment")]:
        flat[key[:-len("_segment")] + "_phase"] = flat[key]
    return flat


def _print_pair(r: Dict) -> None:
    sp = r["split"]; ev = r.get("eval", {})
    print(f"    split: train n={sp['n_train']} ({sp['n_cp_train']} CPs) / "
          f"held-out n={sp['n_heldout']} ({sp['n_cp_heldout']} CPs)")
    print(f"    {'objective':8s} {'train full/cp':>16s} {'HELD-OUT full/cp':>18s}")
    for which in OBJECTIVES:
        if which not in ev:
            continue
        tr, ho = ev[which]["train"], ev[which]["heldout"]
        print(f"    {which:8s} {tr['rmse_full_m']:7.3f}/{tr['rmse_cp_m']:<7.3f} "
              f"   {ho['rmse_full_m']:7.3f}/{ho['rmse_cp_m']:<7.3f}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args) -> None:
    if args.self_test:
        _selftest(args)
        return
    if not args.input:
        sys.exit("error: an input (pairs directory or pair csv) is required.")
    obj = "segment" if args.objective == "phase" else args.objective  # 'phase' alias
    which_list = OBJECTIVES if obj == "both" else (obj,)

    pairs = load_pairs(args)
    if not pairs:
        sys.exit("No pairs loaded (check the manifest / --run / --only filter).")
    model = get_model(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for meta, A in pairs:
        pos = meta.get("follower_position")
        print(f"[f{pos}] calibrating {meta.get('follower_make','?')} "
              f"(run {meta.get('run','?')}), objective={args.objective}, "
              f"train_frac={args.train_frac} ...")
        r = calibrate_pair(model, A, meta, args, which_list)
        jpath = os.path.join(args.output_dir,
                             f"calib_f{r['follower_position']}_"
                             f"{_filesafe(r['follower_make'])}.json")
        with open(jpath, "w") as f:
            json.dump(r, f, indent=2)
        _print_pair(r)
        rows.append(_flatten(r))

    summary = pd.DataFrame(rows)
    spath = os.path.join(args.output_dir, "calib_summary.csv")
    summary.to_csv(spath, index=False)
    with open(os.path.join(args.output_dir, "calib_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"Wrote {len(rows)} calibration(s) -> {args.output_dir}")
    cols = ["follower_make", "rmse_full_heldout_sample", "rmse_cp_heldout_sample",
            "rmse_full_heldout_segment", "rmse_cp_heldout_segment"]
    have = [c for c in cols if c in summary.columns]
    if len(have) > 1:
        print("HELD-OUT generalisation (rows=make; sample vs segment fits):")
        with pd.option_context("display.width", 200):
            print(summary[have].round(3).to_string(index=False))
    print(f"Summary: {spath}")
    print("=" * 78)


def _selftest(args) -> None:
    import tempfile
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "oa_pair_syn_f2_Synth_Car.csv")
    model = get_model(args.model)
    dt, n = 0.1, 1400
    t = np.round(np.arange(n) * dt, 4)
    vL = np.clip(16 + 6 * np.sin(2 * np.pi * t / 40.0), 0, None)
    xL = np.concatenate([[300.0], 300.0 + np.cumsum(0.5 * (vL[1:] + vL[:-1]) * dt)])
    r = simulate.simulate(model, [30, 1.1, 1.8, 2.4, 2.5], t, xL, vL, 0.0, xL[0] - 25.0, 16.0)
    pd.DataFrame({
        "t": t, "x_follower": r.x, "v_follower": r.v, "a_follower": r.a,
        "x_leader": xL, "v_leader": vL, "a_leader": np.gradient(vL, t),
        "leader_length": 0.0, "spacing": r.s, "dv": r.v - vL,
    }).to_csv(csv, index=False)

    A = load_pair_arrays(csv)
    args.maxiter, args.popsize, args.workers = 15, 10, 1
    res = calibrate_pair(model, A, _meta_from_name(csv), args, OBJECTIVES)
    sp = res["split"]
    assert sp["n_train"] + sp["n_heldout"] == sp["n"], "split accounting"
    assert res["theta_sample"] and res["theta_segment"], "both fits expected"
    assert res["theta_phase"] == res["theta_segment"], "phase alias present"
    for w in OBJECTIVES:
        for seg in ("train", "heldout"):
            assert np.isfinite(res["eval"][w][seg]["rmse_full_m"]), f"{w}/{seg}"
    _print_pair(res)
    print("\nSelf-test PASS: train/held-out split, both objectives, full eval table.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-run train/held-out IDM calibration per make under "
                    "sample-based and phase-anchored objectives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?",
                   help="OpenACC pairs directory (with manifest.csv) or a pair csv.")
    p.add_argument("-o", "--output-dir", default="acc_calib")
    p.add_argument("--model", default="idm", help="Car-following model (idm).")
    p.add_argument("--objective", choices=["sample", "segment", "phase", "both"],
                   default="both",
                   help="'phase' is a backward-compatible alias for 'segment'.")
    p.add_argument("--run", default=None,
                   help="Select ONE run by filename substring (required if the pairs "
                        "directory holds several runs).")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="Fraction of each trajectory (from the start) used for training; "
                        "the remaining tail is held out for validation.")
    p.add_argument("--manifest-name", default="manifest.csv")
    p.add_argument("--only", default=None,
                   help="Comma-separated follower positions to calibrate (e.g. '2,3').")

    seg = p.add_argument_group("phase boundaries (PELT+ on the observed follower, SI)")
    seg.add_argument("--penalty", type=float, default=50.0)
    seg.add_argument("--min-seg", type=int, default=20)
    seg.add_argument("--cusum-threshold", type=float, default=2)
    seg.add_argument("--cusum-drift", type=float, default=0.3)

    de = p.add_argument_group("differential evolution")
    de.add_argument("--maxiter", type=int, default=150)
    de.add_argument("--popsize", type=int, default=15)
    de.add_argument("--seed", type=int, default=42)
    de.add_argument("--tol", type=float, default=1e-6)
    de.add_argument("--workers", type=int, default=-1)
    de.add_argument("--collision-penalty", type=float, default=5.0)

    p.add_argument("--self-test", action="store_true")
    return p


def main(argv=None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
