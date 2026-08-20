#!/usr/bin/env python3
"""
evaluate.py
===========
Held-out evaluation for the phase-transition calibration study.

A calibrated parameter vector theta (one global set, from a *pooled* fit over the
training pairs) is graded on the test pairs by **free (open-loop) simulation**:
given only each test pair's initial follower state and its observed leader
trajectory, the follower is integrated forward for the whole pair with no reset,
so errors accumulate -- the honest generalisation test, and the one aligned with
the trajectory-level philosophy of the method.

Metric neutrality
-----------------
The grading metric must not be either objective's own training target, or the
comparison is circular.  The primary metric is therefore held-out **spacing
error** (position-level: raw, not differentiated -- see the NGSIM noise argument)
reported as RMSE (primary) and MAE.  Spacing RMSE is nonetheless the sample
baseline's training target, so on it the phase method faces a *non-inferiority*
bar, not a superiority bar.  Secondary metrics: follower position RMSE, speed
RMSE (its observed side is differentiation-contaminated -- reported for
literature comparability only), plus a feasibility screen (collision / barrier).

Aggregation is by-pair (each test pair equal weight) with a normalised form so
regimes with very different gaps are comparable; pooled-residual RMSE and the
median are reported as robustness checks.  Because both methods are graded on the
*same* test pairs, the head-to-head is a **paired** comparison (Wilcoxon signed
rank) with a pre-registered non-inferiority margin, evaluated against both an
absolute margin (metres) and a relative margin (fraction of the sample error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cf_models import CFModel
from cf_data import PairData
from simulate import simulate, SimResult


# --------------------------------------------------------------------------- #
# Free simulation of one held-out pair
# --------------------------------------------------------------------------- #
def free_simulate(model: CFModel, theta: Sequence[float], pair: PairData,
                  **sim_kw) -> SimResult:
    """Open-loop follower simulation from the pair's initial state.

    The observed leader (position/speed) is the exogenous input; the follower is
    integrated forward for the full horizon with the single global ``theta`` and
    no re-initialisation.
    """
    return simulate(model, theta, pair.t, pair.x_leader, pair.v_leader,
                    pair.leader_length,
                    x0=float(pair.x_follower[0]), v0=float(pair.v_follower[0]),
                    **sim_kw)


# --------------------------------------------------------------------------- #
# Per-pair metrics on a simulated trajectory
# --------------------------------------------------------------------------- #
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def pair_metrics(pair: PairData, sim: SimResult) -> Dict[str, float]:
    """Neutral held-out metrics for one pair (all SI).

    Position-level primary (spacing); position/speed secondary; feasibility
    screen.  ``spacing_rmspe`` normalises spacing RMSE by the pair's mean
    observed gap so pairs of different regimes can be pooled fairly.
    """
    obs_s = np.asarray(pair.spacing, float)
    obs_x = np.asarray(pair.x_follower, float)
    obs_v = np.asarray(pair.v_follower, float)

    s_rmse = _rmse(sim.s, obs_s)
    mean_gap = float(np.mean(obs_s))
    return {
        "pair": pair.name,
        "n": int(pair.n),
        "mean_gap": mean_gap,
        # ---- primary: spacing (position-level) ----
        "spacing_rmse": s_rmse,
        "spacing_mae": _mae(sim.s, obs_s),
        "spacing_rmspe": float(s_rmse / mean_gap) if mean_gap > 0 else np.nan,
        # ---- secondary ----
        "pos_rmse": _rmse(sim.x, obs_x),
        "speed_rmse": _rmse(sim.v, obs_v),   # obs side differentiation-contaminated
        # ---- feasibility screen ----
        "collided": bool(sim.collided),
        "n_barrier": int(sim.n_barrier),
    }


# --------------------------------------------------------------------------- #
# Evaluate one theta over a set of pairs
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    label: str
    per_pair: List[Dict[str, float]]
    aggregate: Dict[str, float]

    def rmse_vector(self, metric: str = "spacing_rmse") -> np.ndarray:
        return np.array([r[metric] for r in self.per_pair], dtype=float)


def evaluate_theta(model: CFModel, theta: Sequence[float],
                   pairs: Sequence[PairData], *, label: str = "",
                   **sim_kw) -> EvalResult:
    """Free-simulate every pair with ``theta`` and collect per-pair + aggregate."""
    rows: List[Dict[str, float]] = []
    pooled_sse = 0.0
    pooled_n = 0
    for pair in pairs:
        sim = free_simulate(model, theta, pair, **sim_kw)
        m = pair_metrics(pair, sim)
        rows.append(m)
        # pooled spacing residuals (sample-weighted)
        resid = np.asarray(sim.s, float) - np.asarray(pair.spacing, float)
        pooled_sse += float(np.sum(resid ** 2))
        pooled_n += resid.size

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    agg = {
        "label": label,
        "n_pairs": len(rows),
        # by-pair headline (each pair equal weight)
        "spacing_rmse_mean": float(np.mean(col("spacing_rmse"))),
        "spacing_rmse_median": float(np.median(col("spacing_rmse"))),
        "spacing_mae_mean": float(np.mean(col("spacing_mae"))),
        "spacing_rmspe_mean": float(np.nanmean(col("spacing_rmspe"))),
        "pos_rmse_mean": float(np.mean(col("pos_rmse"))),
        "speed_rmse_mean": float(np.mean(col("speed_rmse"))),
        # pooled (sample-weighted) robustness check
        "spacing_rmse_pooled": float(np.sqrt(pooled_sse / max(1, pooled_n))),
        # feasibility
        "collision_free_frac": float(np.mean(~col("collided").astype(bool))),
        "pairs_with_barrier": int(np.sum(col("n_barrier") > 0)),
    }
    return EvalResult(label=label, per_pair=rows, aggregate=agg)


# --------------------------------------------------------------------------- #
# Paired head-to-head + non-inferiority
# --------------------------------------------------------------------------- #
def _bootstrap_ci(d: np.ndarray, *, n_boot: int = 10000, alpha: float = 0.05,
                  seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_compare(eval_a: EvalResult, eval_b: EvalResult, *,
                   metric: str = "spacing_rmse",
                   delta_abs: Optional[float] = None,
                   delta_rel: Optional[float] = None,
                   gate: str = "and",
                   seed: int = 0) -> Dict:
    """Paired comparison of A (e.g. phase) vs B (e.g. sample) on ``metric``.

    Differences are d_i = A_i - B_i over the shared pairs (lower metric = better,
    so negative d favours A).  Reports the paired-difference summary, a Wilcoxon
    signed-rank test, and non-inferiority verdicts against an absolute and a
    relative margin (A is non-inferior to B if the upper CI bound on the mean
    difference falls below the margin).  ``gate`` in {'and','or'} combines the
    two margin verdicts.
    """
    names_a = [r["pair"] for r in eval_a.per_pair]
    names_b = [r["pair"] for r in eval_b.per_pair]
    if names_a != names_b:
        # align on common pairs, preserving A's order
        b_by_name = {r["pair"]: r for r in eval_b.per_pair}
        common = [n for n in names_a if n in b_by_name]
        a = np.array([next(r[metric] for r in eval_a.per_pair if r["pair"] == n)
                      for n in common], float)
        b = np.array([b_by_name[n][metric] for n in common], float)
    else:
        a = eval_a.rmse_vector(metric)
        b = eval_b.rmse_vector(metric)

    d = a - b
    d_mean = float(np.mean(d))
    d_median = float(np.median(d))
    lo, hi = _bootstrap_ci(d, seed=seed)

    # Wilcoxon signed-rank (two-sided); guard degenerate all-equal case
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        if nz.size >= 1:
            w_stat, w_p = wilcoxon(a, b)
            w_stat, w_p = float(w_stat), float(w_p)
        else:
            w_stat, w_p = float("nan"), 1.0
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    mean_b = float(np.mean(b))
    out = {
        "metric": metric,
        "n_pairs": int(len(d)),
        "A_mean": float(np.mean(a)), "B_mean": mean_b,
        "diff_mean": d_mean, "diff_median": d_median,
        "diff_ci95": [lo, hi],
        "A_better_frac": float(np.mean(d < 0)),   # share of pairs where A wins
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
    }

    verdicts = {}
    if delta_abs is not None:
        verdicts["abs"] = {"margin": float(delta_abs),
                           "noninferior": bool(hi < delta_abs)}
    if delta_rel is not None:
        margin_rel = float(delta_rel * mean_b)
        verdicts["rel"] = {"delta_rel": float(delta_rel),
                           "margin": margin_rel,
                           "noninferior": bool(hi < margin_rel)}
    if verdicts:
        flags = [v["noninferior"] for v in verdicts.values()]
        gated = all(flags) if gate == "and" else any(flags)
        out["noninferiority"] = {"gate": gate, "verdicts": verdicts,
                                 "noninferior": bool(gated)}
    return out
