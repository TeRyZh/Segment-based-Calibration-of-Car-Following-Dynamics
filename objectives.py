#!/usr/bin/env python3
"""
objectives.py
=============
Step 3 of the phase-transition calibration framework: the objective functions
the optimiser minimises. Two are provided so the manuscript's central
head-to-head -- phase-anchored vs. sample-based calibration -- runs on the same
pair, the same model, and the same forward simulator (decision D9).

1. PhaseAnchoredObjective  (the contribution)
   -------------------------------------------
   Scores the simulated follower at the phases delimited by the critical points
   found by segmenting the *observed* follower once, up front:

       J(theta) = sum_k  || W^(1/2) ( Phi_sim,k(theta) - Phi_obs,k ) ||^2
                = sum_k  sum_j  w_j ( phi_sim,k,j(theta) - phi_obs,k,j )^2

   The segmentation is fixed (independent of theta); theta only moves the
   simulated features evaluated over those same spans. Phi is built from the
   feature registry in phase_segmentation -- the same computer runs on the
   observed and the simulated trajectory, so the two cannot drift apart. Which
   features enter Phi is an argument, not a hard-wire: pass `feature_keys`
   (or --features on run_experiment's CLI) with any subset of

       v_end  s_end  dist  phase_speed  phase_speed_ols  duration

   so feature sets are swappable for ablation. Boundary-only keys (v_end,
   s_end, phase_speed, duration) read the trajectory at the critical points;
   dist and phase_speed_ols summarise the samples within a phase. W is diagonal
   and z-scores each feature by its observed cross-phase standard deviation, so
   features of mixed units (m, m/s, s) are commensurable after standardising.

2. SampleObjective  (the baseline)
   --------------------------------
   The conventional sample-by-sample error over every 0.1 s sample. Target
   series defaults to spacing (Punzo et al.'s recommended gap-based objective);
   speed / position are selectable. Metric defaults to RMSE (the named
   baseline); MAE is selectable. RMSE-vs-MAE and the spacing-based choice are
   flagged open decisions -- both are exposed here, neither is hard-wired.

Both objectives share the continuous simulator (simulate.simulate) with
identical physical settings, add a soft collision penalty proportional to the
number of no-overtake-barrier hits, and expose ``simulate(theta)`` and
``details(theta)`` for reporting. AggregateObjective sums the per-pair scores
(mean over pairs) to calibrate a single global parameter set across a set.

All quantities SI (decision D1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cf_models import CFModel
from cf_data import PairData
from phase_segmentation import (SegmentationResult, segment_trajectory,
                                FEATURE_KEYS, phase_feature_matrix,
                                validate_feature_keys)
from simulate import (SimResult, simulate,
                      A_MIN_DEFAULT, A_MAX_DEFAULT, S_MIN_HARD)

_EPS = 1e-9

# The per-phase feature menu lives in phase_segmentation's registry and is used
# for the observed AND the simulated trajectory, so the two cannot drift apart.
# The *selection* is an argument (feature_keys below / --features on the CLI).
# Registered keys: phase_segmentation.FEATURE_KEYS.
FEATURE_KEYS_DEFAULT: Tuple[str, ...] = ("v_end", "s_end")


# --------------------------------------------------------------------------- #
# Common base
# --------------------------------------------------------------------------- #
class _BaseObjective:
    """Shared plumbing: bound simulator settings + a single simulate() path."""

    def __init__(self, model: CFModel, pair: PairData,
                 collision_penalty: float = 1.0,
                 a_min: float = A_MIN_DEFAULT, a_max: float = A_MAX_DEFAULT,
                 s_min_hard: float = S_MIN_HARD):
        self.model = model
        self.pair = pair
        self.collision_penalty = float(collision_penalty)
        self._sim_kw = dict(a_min=a_min, a_max=a_max, s_min_hard=s_min_hard)

    def simulate(self, theta: Sequence[float]) -> SimResult:
        p = self.pair
        return simulate(self.model, theta, t=p.t,
                        x_leader=p.x_leader, v_leader=p.v_leader,
                        leader_length=p.leader_length,
                        x0=float(p.x_follower[0]), v0=float(p.v_follower[0]),
                        **self._sim_kw)

    # subclasses implement these two
    def _terms(self, sim: SimResult) -> Tuple[float, int]:
        """Return (sum_of_penalised_squared_terms, n_terms) for aggregation."""
        raise NotImplementedError

    def __call__(self, theta: Sequence[float]) -> float:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1) Phase-anchored objective
# --------------------------------------------------------------------------- #
class PhaseAnchoredObjective(_BaseObjective):
    """Phase-boundary objective J(theta) (framework Step 3, the contribution)."""

    kind = "phase"

    def __init__(self, model: CFModel, pair: PairData,
                 segmentation: SegmentationResult,
                 feature_keys: Sequence[str] = FEATURE_KEYS_DEFAULT,
                 weighting: str = "zscore",
                 weights: Optional[Sequence[float]] = None,
                 collision_penalty: float = 1.0,
                 **sim_kw):
        super().__init__(model, pair, collision_penalty, **sim_kw)
        self.seg = segmentation
        self.keys: Tuple[str, ...] = validate_feature_keys(feature_keys)

        # boundaries and observed features (fixed; theta-independent)
        self.bounds: List[Tuple[int, int]] = [(ph.i_start, ph.i_end)
                                              for ph in segmentation.phases]
        self.phi_obs = segmentation.feature_matrix(self.keys)     # (K, F)

        # diagonal weights W (per feature)
        F = len(self.keys)
        if weights is not None:
            w = np.asarray(weights, float)
            if w.shape != (F,):
                raise ValueError(f"weights must have length {F}")
        elif weighting == "zscore":
            std = self.phi_obs.std(axis=0)                        # cross-phase std
            std = np.where(std > _EPS, std, 1.0)                  # guard K==1/const
            w = 1.0 / (std ** 2)
        elif weighting == "none":
            w = np.ones(F)
        else:
            raise ValueError("weighting must be 'zscore' | 'none' or pass weights")
        self.w = w

    def _phi_sim(self, sim: SimResult) -> np.ndarray:
        """Phi_sim (K, F) -- same computer, same spans, same keys as Phi_obs."""
        return phase_feature_matrix(sim.t, sim.x, sim.v, sim.s,
                                    self.bounds, self.keys)

    def _terms(self, sim: SimResult) -> Tuple[float, int]:
        resid = self._phi_sim(sim) - self.phi_obs                    # (K, F)
        wsq = float(np.sum(self.w * resid ** 2))
        if sim.collided:
            wsq += self.collision_penalty * sim.n_barrier
        return wsq, self.phi_obs.size

    def __call__(self, theta: Sequence[float]) -> float:
        val, _ = self._terms(self.simulate(theta))
        return val

    # -- reporting -----------------------------------------------------------
    def per_phase_table(self, theta: Sequence[float]) -> List[Dict]:
        """Per-phase observed vs simulated features with weighted residuals."""
        sim = self.simulate(theta)
        phi_sim = self._phi_sim(sim)
        rows = []
        for k, ph in enumerate(self.seg.phases):
            row = {"k": k, "kind": ph.kind,
                   "i0": ph.i_start, "i1": ph.i_end,
                   "t0": ph.t_start, "t1": ph.t_end}
            for j, key in enumerate(self.keys):
                o, s = float(self.phi_obs[k, j]), float(phi_sim[k, j])
                row[f"{key}_obs"] = o
                row[f"{key}_sim"] = s
                row[f"{key}_res"] = s - o
                row[f"{key}_wsq"] = float(self.w[j] * (s - o) ** 2)
            rows.append(row)
        return rows

    def details(self, theta: Sequence[float]) -> Dict:
        sim = self.simulate(theta)
        val, n = self._terms(sim)
        return {"kind": self.kind, "J": val, "n_terms": n,
                "n_phases": self.seg.n_phases, "collided": sim.collided,
                "n_barrier": sim.n_barrier, "keys": self.keys,
                "weights": self.w.tolist()}


# --------------------------------------------------------------------------- #
# 2) Sample-based baseline
# --------------------------------------------------------------------------- #
class SampleObjective(_BaseObjective):
    """Conventional sample-by-sample error over every sample (the baseline)."""

    kind = "sample"
    _TARGETS = ("spacing", "speed", "position")
    _METRICS = ("rmse", "mae")

    def __init__(self, model: CFModel, pair: PairData,
                 target: str = "spacing", metric: str = "rmse",
                 collision_penalty: float = 1.0, **sim_kw):
        super().__init__(model, pair, collision_penalty, **sim_kw)
        if target not in self._TARGETS:
            raise ValueError(f"target must be one of {self._TARGETS}")
        if metric not in self._METRICS:
            raise ValueError(f"metric must be one of {self._METRICS}")
        self.target = target
        self.metric = metric
        self._obs = {"spacing": pair.spacing,
                     "speed": pair.v_follower,
                     "position": pair.x_follower}[target]

    def _sim_series(self, sim: SimResult) -> np.ndarray:
        return {"spacing": sim.s, "speed": sim.v, "position": sim.x}[self.target]

    def _resid(self, sim: SimResult) -> np.ndarray:
        return self._sim_series(sim) - self._obs

    def _terms(self, sim: SimResult) -> Tuple[float, int]:
        r = self._resid(sim)
        # for pooled aggregation we return sum of squares + count (RMSE-consistent)
        sse = float(np.sum(r ** 2))
        if sim.collided:
            sse += (self.collision_penalty ** 2) * sim.n_barrier
        return sse, r.size

    def __call__(self, theta: Sequence[float]) -> float:
        sim = self.simulate(theta)
        r = self._resid(sim)
        if self.metric == "rmse":
            val = float(np.sqrt(np.mean(r ** 2)))
        else:
            val = float(np.mean(np.abs(r)))
        if sim.collided:
            val += self.collision_penalty * sim.n_barrier / max(1, r.size)
        return val

    def details(self, theta: Sequence[float]) -> Dict:
        sim = self.simulate(theta)
        return {"kind": self.kind, "target": self.target, "metric": self.metric,
                "value": self.__call__(theta), "collided": sim.collided,
                "n_barrier": sim.n_barrier}


# --------------------------------------------------------------------------- #
# Multi-pair aggregate (single global parameter set across a set)
# --------------------------------------------------------------------------- #
class AggregateObjective:
    """Combine per-pair objectives into one scalar (mean over pairs).

    For phase objectives this is the mean per-pair J; for sample objectives it
    is the pooled RMSE/MAE across all samples of all pairs -- both are the
    natural aggregate for their metric and are monotone for the optimiser.
    """

    def __init__(self, objectives: Sequence[_BaseObjective]):
        if not objectives:
            raise ValueError("AggregateObjective needs >= 1 objective")
        self.objs = list(objectives)
        self.kind = self.objs[0].kind
        self._pooled_sample = isinstance(self.objs[0], SampleObjective)
        self._sample_metric = (self.objs[0].metric
                               if self._pooled_sample else None)

    def __call__(self, theta: Sequence[float]) -> float:
        if self._pooled_sample:
            tot, n = 0.0, 0
            for o in self.objs:
                sse, cnt = o._terms(o.simulate(theta))
                tot += sse
                n += cnt
            pooled = tot / max(1, n)
            return float(np.sqrt(pooled)) if self._sample_metric == "rmse" \
                else float(pooled)
        # phase (or generic): mean of per-pair scalars
        return float(np.mean([o(theta) for o in self.objs]))

    def per_pair(self, theta: Sequence[float]) -> List[float]:
        return [float(o(theta)) for o in self.objs]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_objective(kind: str, model: CFModel, pair: PairData,
                   segmentation: Optional[SegmentationResult] = None,
                   **kw) -> _BaseObjective:
    """Build a per-pair objective: kind in {'phase', 'sample'}.

    For 'phase', a SegmentationResult must be supplied (segment the observed
    follower once, up front) or it is computed here from the pair with defaults.
    """
    k = str(kind).strip().lower()
    if k in ("phase", "phase_anchored", "critical", "critical_point"):
        if segmentation is None:
            segmentation = segment_trajectory(pair.t, pair.x_follower,
                                              pair.v_follower, pair.spacing)
        return PhaseAnchoredObjective(model, pair, segmentation, **kw)
    
    if k in ("sample", "rmse", "sample_rmse", "continuous"):
        return SampleObjective(model, pair, **kw)
    raise ValueError(f"unknown objective kind {kind!r} (use 'phase' or 'sample')")


# --------------------------------------------------------------------------- #
# Self-test: known-theta* pair -> J(theta*) near 0, off-theta -> larger
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    import glob
    from cf_models import get_model
    from cf_data import load_pair, discover_pair

    print("=" * 70)
    print("objectives.py self-test")
    print("=" * 70)

    p = discover_pair() or (glob.glob("/tmp/pair_clean.csv") or [None])[0]
    if p is None:
        print("no pair CSV available - skipping data-dependent checks")
        return
    pair = load_pair(p)
    idm = get_model("idm")
    theta_star = np.array([16.5, 1.2, 1.4, 2.2, 2.0])     # generator's truth
    theta_off = np.array([30.0, 2.5, 1.0, 4.0, 5.0])      # deliberately wrong
    
    seg = segment_trajectory(pair.t, pair.x_follower, pair.v_follower, pair.spacing)
    print(f"\npair={pair.name}  units={pair.units_source}  "
          f"phases={seg.n_phases}  cps={len(seg.critical_points)}")

    phase = PhaseAnchoredObjective(idm, pair, seg)
    samp = SampleObjective(idm, pair, target="spacing", metric="rmse")

    Jp_star, Jp_off = phase(theta_star), phase(theta_off)
    Rs_star, Rs_off = samp(theta_star), samp(theta_off)
    print(f"\n[1] phase-anchored J:  theta*={Jp_star:.4f}   off={Jp_off:.4f}")
    print(f"[2] sample RMSE(gap):  theta*={Rs_star:.4f} m theta_off={Rs_off:.4f} m")
    assert Jp_off > Jp_star, "phase objective should prefer theta*"
    assert Rs_off > Rs_star, "sample objective should prefer theta*"
    print("    both objectives rank theta* below the wrong vector -> PASS")

    print("\n[3] per-phase table at theta* (first 3 phases)")
    for row in phase.per_phase_table(theta_star)[:3]:
        print(f"    k={row['k']} {row['kind']:6s} "
              f"v_end obs/sim={row['v_end_obs']:.2f}/{row['v_end_sim']:.2f} "
              f"s_end obs/sim={row['s_end_obs']:.2f}/{row['s_end_sim']:.2f}")

    print("\n[4] aggregate over [pair, pair] equals single-pair value")
    agg = AggregateObjective([phase, PhaseAnchoredObjective(idm, pair, seg)])
    assert abs(agg(theta_star) - Jp_star) < 1e-9, "aggregate mean mismatch"
    print(f"    agg={agg(theta_star):.4f} == single={Jp_star:.4f} -> PASS")

    print("\nAll objective checks passed.")


if __name__ == "__main__":
    _selftest()
