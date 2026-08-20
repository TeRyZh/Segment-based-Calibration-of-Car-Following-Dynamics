#!/usr/bin/env python3
"""
phase_segmentation.py
=====================
Step 1 of the phase-transition calibration framework.

Wraps the project's `PELTPlusDetection` to turn a follower trajectory into a set
of behavioural phases separated by *critical points* (acceleration<->deceleration
switches), and extracts the observed per-phase feature vectors Phi_obs,k that the
phase-anchored objective scores against.

We reuse PELT+ verbatim (CUSUM candidates on velocity, PELT cost on position -
the piecewise-linear / critical-point definition used in the manuscript). Only
the small segment -> critical-point classification is lifted from analyzer_v6's
`process_single_trajectory` (the rest of analyzer_v6 is shockwave clustering and
depends on modules unrelated to calibration).

A "phase" k spans observed sample indices [i_{k-1}, i_k] (boundaries are the
critical points, plus the trajectory endpoints). Phi_obs,k holds the per-phase
features the objective scores against; the *menu* lives in the registry below
and the *selection* is an argument (objectives.PhaseAnchoredObjective's
`feature_keys`, run_experiment's `--features`), so feature sets can be swapped
freely for ablation.

Registered features (all SI):

    key                level        support        value over span [i0, i1]
    ---------------    ----------   ------------   ---------------------------
    v_end              velocity     1 sample       v(t_k)
    s_end              position     1 sample       s(t_k)
    dist               velocity     all interior   integral_{t_k-1}^{t_k} v dt
    phase_speed        position     2 endpoints    (x_k - x_k-1)/(t_k - t_k-1)
    phase_speed_ols    position     all interior   OLS slope of x on t
    duration           time         2 endpoints    t_k - t_k-1

TO ADD A FEATURE: write one function and decorate it with @feature("my_key")
(see the registry below). It becomes available everywhere at once -- the
observed side, the simulated side, and the CLI -- with no other edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from pelt_plus_class import PELTPlusDetection

_EPS = 1e-9
# numpy 2.x removed np.trapz; alias for both (must match objectives.py).
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


# --------------------------------------------------------------------------- #
# Per-phase feature registry -- the single source of truth
# --------------------------------------------------------------------------- #
# Every per-phase feature, observed AND simulated, is computed by the functions
# below and nowhere else, so Phi_obs and Phi_sim cannot drift apart.
#
# TO ADD A FEATURE:
#
#     @feature("my_key")
#     def _f_my_key(t, x, v, s, i0, i1):
#         """One-line description (goes in the docs)."""
#         return float(...)
#
# and it is immediately selectable via PhaseAnchoredObjective(feature_keys=...)
# and run_experiment's --features flag. No other file needs to change.

FeatureFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int],
                     float]
_FEATURE_FNS: Dict[str, FeatureFn] = {}          # insertion-ordered


def feature(key: str) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator: register a per-phase feature function under `key`."""
    def _register(fn: FeatureFn) -> FeatureFn:
        if key in _FEATURE_FNS:
            raise KeyError(f"feature key {key!r} is already registered")
        _FEATURE_FNS[key] = fn
        return fn
    return _register


# -- velocity-level (differentiated channel) -------------------------------- #
@feature("v_end")
def _f_v_end(t, x, v, s, i0, i1) -> float:
    """Terminal follower speed v(t_k), read off the differentiated channel."""
    return float(v[i1])


# -- position-level --------------------------------------------------------- #
@feature("s_end")
def _f_s_end(t, x, v, s, i0, i1) -> float:
    """Terminal net spacing s(t_k)."""
    return float(s[i1])


@feature("dist")
def _f_dist(t, x, v, s, i0, i1) -> float:
    """Distance travelled in the phase, integral_{t_k-1}^{t_k} v dt."""
    return float(_trapz(v[i0:i1 + 1], t[i0:i1 + 1]))


# -- phase-level ------------------------------------------------------------ #
@feature("phase_speed")
def _f_phase_speed(t, x, v, s, i0, i1) -> float:
    """Mean speed across the phase from its endpoints, (x_k - x_k-1)/dt.

    A secant slope of position over the whole phase: an *averaging* operator,
    so position noise is suppressed by the phase duration rather than amplified
    by 1/dt as it is when reading the differentiated channel (v_end).
    """
    dt = float(t[i1] - t[i0])
    return float((x[i1] - x[i0]) / dt) if dt > _EPS else float(v[i1])


@feature("phase_speed_ols")
def _f_phase_speed_ols(t, x, v, s, i0, i1) -> float:
    """Mean speed across the phase as the OLS slope of x on t (all samples)."""
    dt = float(t[i1] - t[i0])
    if i1 - i0 < 1 or dt <= _EPS:
        return float(v[i1])
    tt = np.asarray(t[i0:i1 + 1], float)
    xx = np.asarray(x[i0:i1 + 1], float)
    tc = tt - tt.mean()
    den = float(tc @ tc)
    if den <= _EPS:
        return float(v[i1])
    # closed form == np.polyfit(tt, xx, 1)[0], without the per-call lstsq/SVD
    return float(tc @ (xx - xx.mean()) / den)


@feature("duration")
def _f_duration(t, x, v, s, i0, i1) -> float:
    """Phase duration t_k - t_k-1.

    NOTE: under a fixed segmentation the spans are shared by the observed and
    the simulated trajectory, so this feature's residual is identically zero
    and it contributes nothing to J. Registered for the event-alignment
    variant, where phases are re-detected on the simulation.
    """
    return float(t[i1] - t[i0])


# Registration order; `feature_matrix()` and `_phase_features()` use all of it.
FEATURE_KEYS: Tuple[str, ...] = tuple(_FEATURE_FNS)


def validate_feature_keys(keys: Sequence[str]) -> Tuple[str, ...]:
    """Return `keys` as a tuple, raising early on unknown entries."""
    keys = tuple(keys)
    if not keys:
        raise ValueError("need at least one feature key")
    unknown = [k for k in keys if k not in _FEATURE_FNS]
    if unknown:
        raise KeyError(f"unknown feature key(s) {unknown}; registered: "
                       f"{', '.join(FEATURE_KEYS)}")
    return keys


def _phase_vals(t: np.ndarray, x: np.ndarray, v: np.ndarray, s: np.ndarray,
                spans: Sequence[Sequence[int]], key: str) -> np.ndarray:
    """Value of `key` for every phase span [i0, i1] (inclusive), SI."""
    try:
        fn = _FEATURE_FNS[key]
    except KeyError:
        raise KeyError(f"unknown feature key {key!r}; registered: "
                       f"{', '.join(FEATURE_KEYS)}") from None
    return np.asarray([fn(t, x, v, s, int(i0), int(i1)) for i0, i1 in spans],
                      dtype=float)


def phase_feature_matrix(t: np.ndarray, x: np.ndarray, v: np.ndarray,
                         s: np.ndarray, spans: Sequence[Sequence[int]],
                         keys: Sequence[str]) -> np.ndarray:
    """Stack the selected features over spans -> (n_spans, n_keys), SI.

    The single entry point for BOTH the observed trajectory and any simulated
    one, so Phi_obs and Phi_sim are identical by construction.
    """
    keys = validate_feature_keys(keys)
    spans = [(int(i0), int(i1)) for i0, i1 in spans]
    if not spans:
        return np.empty((0, len(keys)), dtype=float)
    return np.column_stack([_phase_vals(t, x, v, s, spans, k) for k in keys])


def _phase_features(t: np.ndarray, x: np.ndarray, v: np.ndarray,
                    s: np.ndarray, i0: int, i1: int) -> Dict[str, float]:
    """All registered features over the inclusive sample span [i0, i1]."""
    return {key: float(fn(t, x, v, s, int(i0), int(i1)))
            for key, fn in _FEATURE_FNS.items()}


@dataclass
class Phase:
    """One behavioural phase between two critical points (inclusive indices)."""
    k: int
    i_start: int
    i_end: int
    t_start: float
    t_end: float
    kind: str                       # 'accel' | 'decel' | 'single'
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def n_samples(self) -> int:
        return self.i_end - self.i_start + 1


@dataclass
class SegmentationResult:
    critical_points: List[int]      # sample indices of accel<->decel switches
    decel_points: List[int]
    accel_points: List[int]
    phases: List[Phase]
    diagnostics: dict

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def feature_matrix(self, keys: Sequence[str] = FEATURE_KEYS) -> np.ndarray:
        """Stack Phi_obs,k over phases -> (n_phases, len(keys)) array."""
        return np.array([[ph.features[key] for key in keys] for ph in self.phases],
                        dtype=float)


def segment_trajectory(t: np.ndarray,
                       x: np.ndarray,
                       v: np.ndarray,
                       s: np.ndarray,
                       penalty: float = 50.0,
                       min_segment_length: int = 20,
                       cusum_threshold: float = 5.0,
                       cusum_drift: float = 1.0) -> SegmentationResult:
    """Segment a follower trajectory into phases and extract Phi_obs,k.

    Args:
        t, x, v : follower time, position, speed (SI, equal length).
        s       : net spacing series (SI), aligned to t (used for s_end feature).
        penalty, min_segment_length, cusum_threshold, cusum_drift : PELT+ knobs.

    Returns:
        SegmentationResult with critical points, phases, and per-phase features.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    s = np.asarray(s, float)
    n = len(t)

    detector = PELTPlusDetection(penalty=penalty,
                                 min_segment_length=min_segment_length,
                                 cusum_threshold=cusum_threshold,
                                 cusum_drift=cusum_drift)
    change_points, diagnostics = detector.detect(
        {"time": t, "distance": x, "velocity": v})
    segments = diagnostics.get("segments", [])

    # Classify each interior change point as accel/decel by comparing the
    # velocity (segment slope) across it -- lifted from analyzer_v6.
    decel_points: List[int] = []
    accel_points: List[int] = []
    for i in range(len(segments) - 1):
        cp = segments[i + 1]["start_idx"]
        if segments[i + 1]["slope"] < segments[i]["slope"]:
            decel_points.append(cp)
        else:
            accel_points.append(cp)
    critical_points = sorted(set(decel_points) | set(accel_points))

    # Build phases from endpoints + critical points.
    boundaries = [0] + critical_points + [n - 1]
    boundaries = sorted(set(b for b in boundaries if 0 <= b <= n - 1))
    decel_set = set(decel_points)

    phases: List[Phase] = []
    for k in range(len(boundaries) - 1):
        i0, i1 = boundaries[k], boundaries[k + 1]
        if i1 <= i0:
            continue
        if not critical_points:
            kind = "single"
        elif i1 in decel_set:
            kind = "decel"
        elif i1 in set(accel_points):
            kind = "accel"
        else:
            kind = "single"       # phase ending at the trajectory endpoint
        phases.append(Phase(
            k=k, i_start=i0, i_end=i1,
            t_start=float(t[i0]), t_end=float(t[i1]), kind=kind,
            features=_phase_features(t, x, v, s, i0, i1)))

    return SegmentationResult(
        critical_points=critical_points,
        decel_points=decel_points,
        accel_points=accel_points,
        phases=phases,
        diagnostics=diagnostics)


if __name__ == "__main__":
    import glob
    import pandas as pd

    paths = glob.glob("/mnt/user-data/uploads/*.csv") + glob.glob("*.csv")
    if not paths:
        print("no CF-pair CSV found for demo"); raise SystemExit
    df = pd.read_csv(paths[0])
    FT = 0.3048
    k = 1.0 if df["spacing"].max() < 60 else FT
    t = df["t"].to_numpy()
    res = segment_trajectory(t, df["x_follower"].to_numpy() * k,
                             df["v_follower"].to_numpy() * k,
                             df["spacing"].to_numpy() * k)
    print(f"file: {paths[0].split('/')[-1]}")
    print(f"critical points: {res.critical_points}")
    print(f"  decel: {res.decel_points}   accel: {res.accel_points}")
    print(f"phases: {res.n_phases}")
    for ph in res.phases:
        print(f"  k={ph.k} [{ph.i_start:>3d}->{ph.i_end:<3d}] "
              f"{ph.kind:6s} dur={ph.duration:5.1f}s  "
              f"v_end={ph.features['v_end']:6.2f}  "
              f"s_end={ph.features['s_end']:6.2f}  "
              f"dist={ph.features['dist']:7.2f}")
