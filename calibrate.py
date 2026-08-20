#!/usr/bin/env python3
"""
calibrate.py
============
Step 4 of the phase-transition calibration framework: global optimisation of the
model parameters against a chosen objective.

Because the phase-anchored objective smooths the error landscape relative to
sample-based RMSE, a GA-style global search suffices. We use SciPy's
``differential_evolution`` (decision D7: closest to the manuscript's GA
description, no extra dependency) within the model's physical parameter bounds.
The same driver serves the sample-based baseline, so the two calibrations are
produced identically for the head-to-head.

Usage
-----
    # one pair, phase-anchored objective, IDM
    python calibrate.py --pair pair.csv --model idm --objective phase

    # one pair, sample-based RMSE baseline on the gap
    python calibrate.py --pair pair.csv --model idm --objective sample \
        --target spacing --metric rmse

    # a whole train split (single global theta across the set)
    python calibrate.py --dir cf_pairs_out --split train --model gipps \
        --objective phase --maxiter 80 --out theta_gipps_phase.json

Reports theta* (named, SI), the final objective, and -- for a set -- the
per-pair scores at theta*.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import differential_evolution

from cf_models import CFModel, get_model
from cf_data import PairData, load_pair, load_folder, discover_pair
from phase_segmentation import SegmentationResult, segment_trajectory
from objectives import (AggregateObjective, PhaseAnchoredObjective,
                        SampleObjective, make_objective)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class CalibResult:
    model: str
    objective: str
    param_names: List[str]
    theta: List[float]
    param_dict: Dict[str, float]
    fun: float
    success: bool
    n_iter: int
    n_eval: int
    elapsed_s: float
    n_pairs: int
    pair_names: List[str] = field(default_factory=list)
    per_pair: Optional[List[float]] = None
    objective_config: Dict = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def summary(self) -> str:
        lines = [
            f"model      : {self.model}",
            f"objective  : {self.objective}  {self.objective_config}",
            f"pairs      : {self.n_pairs}"
            + ("" if self.n_pairs == 1 else f"  ({', '.join(self.pair_names[:4])}"
               + (" ..." if self.n_pairs > 4 else "") + ")"),
            f"final obj  : {self.fun:.6g}   "
            f"({'converged' if self.success else 'stopped'}, "
            f"{self.n_iter} iters, {self.n_eval} evals, {self.elapsed_s:.1f}s)",
            "theta*     :",
        ]
        for name, val in self.param_dict.items():
            lines.append(f"    {name:8s} = {val:.4f}")
        if self.per_pair is not None and self.n_pairs > 1:
            lines.append("per-pair obj at theta*:")
            for nm, v in zip(self.pair_names, self.per_pair):
                lines.append(f"    {nm:32s} {v:.6g}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SciPy version shim (seed -> rng across releases)
# --------------------------------------------------------------------------- #
def _rng_kwarg(seed: Optional[int]) -> Dict:
    params = inspect.signature(differential_evolution).parameters
    if "rng" in params:
        return {"rng": seed}
    return {"seed": seed}


# --------------------------------------------------------------------------- #
# Core drivers
# --------------------------------------------------------------------------- #
def _run_de(objective, bounds, *, maxiter: int, popsize: int, seed: Optional[int],
            tol: float, mutation, recombination: float, polish: bool,
            workers: int):
    de_kw = dict(maxiter=maxiter, popsize=popsize, tol=tol, mutation=mutation,
                 recombination=recombination, polish=polish, workers=workers,
                 init="latinhypercube")
    if workers != 1:
        de_kw["updating"] = "deferred"      # required for parallel evaluation
    de_kw.update(_rng_kwarg(seed))
    return differential_evolution(objective, bounds, **de_kw)


def calibrate_pairs(model: CFModel,
                    pairs: Sequence[PairData],
                    objective: str = "phase",
                    *,
                    segmentations: Optional[Sequence[SegmentationResult]] = None,
                    obj_kwargs: Optional[Dict] = None,
                    maxiter: int = 60, popsize: int = 15,
                    seed: Optional[int] = 42, tol: float = 1e-6,
                    mutation=(0.5, 1.0), recombination: float = 0.7,
                    polish: bool = True, workers: int = 1) -> CalibResult:
    """Calibrate a single global parameter vector across one or more pairs."""
    if not pairs:
        raise ValueError("no pairs to calibrate on")
    obj_kwargs = dict(obj_kwargs or {})

    # build per-pair objectives (segment observed follower once for 'phase')
    per_pair_objs = []
    for i, pair in enumerate(pairs):
        seg = None
        if objective.lower().startswith(("phase", "critical")):
            seg = (segmentations[i] if segmentations is not None
                   else segment_trajectory(pair.t, pair.x_follower,
                                           pair.v_follower, pair.spacing))
        per_pair_objs.append(make_objective(objective, model, pair,
                                            segmentation=seg, **obj_kwargs))

    obj = per_pair_objs[0] if len(per_pair_objs) == 1 \
        else AggregateObjective(per_pair_objs)

    t0 = time.time()
    res = _run_de(obj, model.bounds(), maxiter=maxiter, popsize=popsize,
                  seed=seed, tol=tol, mutation=mutation,
                  recombination=recombination, polish=polish, workers=workers)
    elapsed = time.time() - t0

    theta = model.clip(res.x)                       # guard tiny polish overshoot
    per_pair = (AggregateObjective(per_pair_objs).per_pair(theta)
                if len(per_pair_objs) > 1 else None)

    cfg = _objective_config(objective, obj_kwargs, per_pair_objs[0])
    return CalibResult(
        model=model.name, objective=objective,
        param_names=list(model.param_names),
        theta=[float(x) for x in theta],
        param_dict={k: float(v) for k, v in model.param_dict(theta).items()},
        fun=float(res.fun), success=bool(res.success),
        n_iter=int(getattr(res, "nit", -1)), n_eval=int(getattr(res, "nfev", -1)),
        elapsed_s=elapsed, n_pairs=len(pairs),
        pair_names=[p.name for p in pairs], per_pair=per_pair,
        objective_config=cfg)


def calibrate_pair(model: CFModel, pair: PairData, objective: str = "phase",
                   *, segmentation: Optional[SegmentationResult] = None,
                   **kw) -> CalibResult:
    """Convenience wrapper for a single pair."""
    segs = [segmentation] if segmentation is not None else None
    return calibrate_pairs(model, [pair], objective, segmentations=segs, **kw)


def _objective_config(objective: str, obj_kwargs: Dict, example) -> Dict:
    if isinstance(example, PhaseAnchoredObjective):
        return {"features": list(example.keys),
                "weighting": obj_kwargs.get("weighting", "zscore")}
    if isinstance(example, SampleObjective):
        return {"target": example.target, "metric": example.metric}
    return dict(obj_kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_pairs_from_args(args) -> List[PairData]:
    if args.dir:
        pairs = load_folder(args.dir, split=args.split, units=args.units,
                            limit=args.limit)
        if not pairs:
            sys.exit(f"No pairs found under {args.dir} "
                     f"(split={args.split!r}).")
        return pairs
    path = discover_pair(args.pair)
    if path is None:
        sys.exit("No pair CSV given/found. Use --pair PATH or --dir FOLDER.")
    return [load_pair(path, units=args.units)]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase-transition (and baseline) CF calibration driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("input")
    src.add_argument("--pair", help="A single per-pair CSV.")
    src.add_argument("--dir", help="An extract_cf_pairs.py output folder.")
    src.add_argument("--split", choices=["train", "test"], default=None,
                     help="Restrict --dir to this split (uses manifest.csv).")
    src.add_argument("--limit", type=int, default=None,
                     help="Cap number of pairs loaded from --dir.")
    src.add_argument("--units", choices=["auto", "si", "feet"], default="auto",
                     help="Unit handling (D1). 'feet' is always safe for NGSIM.")

    mdl = p.add_argument_group("model / objective")
    mdl.add_argument("--model", choices=["idm", "gipps", "ovm"], default="idm")
    mdl.add_argument("--objective", choices=["phase", "sample"], default="phase")
    mdl.add_argument("--features", default="v_end,s_end",
                     help="Phase objective feature keys (comma-sep) from "
                          "v_end,s_end,dist (decision D3).")
    mdl.add_argument("--weighting", choices=["zscore", "none"], default="zscore")
    mdl.add_argument("--target", choices=["spacing", "speed", "position"],
                     default="spacing", help="Sample objective target series.")
    mdl.add_argument("--metric", choices=["rmse", "mae"], default="rmse",
                     help="Sample objective metric.")
    mdl.add_argument("--collision-penalty", type=float, default=1.0)

    opt = p.add_argument_group("optimiser (differential_evolution)")
    opt.add_argument("--maxiter", type=int, default=60)
    opt.add_argument("--popsize", type=int, default=15)
    opt.add_argument("--seed", type=int, default=42)
    opt.add_argument("--tol", type=float, default=1e-6)
    opt.add_argument("--recombination", type=float, default=0.7)
    opt.add_argument("--no-polish", action="store_true",
                     help="Skip the final local L-BFGS-B refinement.")
    opt.add_argument("--workers", type=int, default=1,
                     help="Parallel workers for DE (-1 = all cores).")

    p.add_argument("--out", help="Write the CalibResult to this JSON path.")
    return p


def _obj_kwargs_from_args(args) -> Dict:
    if args.objective == "phase":
        return {"feature_keys": tuple(k.strip() for k in args.features.split(",")
                                      if k.strip()),
                "weighting": args.weighting,
                "collision_penalty": args.collision_penalty}
    return {"target": args.target, "metric": args.metric,
            "collision_penalty": args.collision_penalty}


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    model = get_model(args.model)
    pairs = _load_pairs_from_args(args)

    print(f"Loaded {len(pairs)} pair(s); units: {pairs[0].units_source}. "
          f"Calibrating {model.name} against the '{args.objective}' objective ...")
    res = calibrate_pairs(
        model, pairs, args.objective,
        obj_kwargs=_obj_kwargs_from_args(args),
        maxiter=args.maxiter, popsize=args.popsize, seed=args.seed,
        tol=args.tol, recombination=args.recombination,
        polish=not args.no_polish, workers=args.workers)

    print("\n" + "=" * 60)
    print(res.summary())
    print("=" * 60)
    if args.out:
        res.to_json(args.out)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
