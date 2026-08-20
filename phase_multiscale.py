#!/usr/bin/env python3
"""
phase_multiscale.py
===================
Multi-penalty extension of Step 1 (phase segmentation).

Two things this adds on top of ``phase_segmentation.segment_trajectory``:

1. **Detector dispatch.**  The critical-point detector becomes a runtime choice
   -- PELT+ (CUSUM candidates on velocity, PELT cost on position), Pure PELT
   (PELT on position), or Pure DP (exact segmented regression on position).  The
   three have inconsistent interfaces; ``_detect`` normalises them.

2. **Penalty-grid segmentation + union merge.**  The penalty beta is the one free
   hyperparameter of the phase objective.  Rather than pick one, segment at every
   beta in a grid and merge the resulting critical-point sets into a single
   boundary set by tolerance clustering.  The merged segmentation is an ordinary
   ``SegmentationResult`` -- the calibration objective, the simulator, and the
   evaluation are all unchanged, and the boundaries remain theta-independent
   (Variant A: fixed observed boundaries).

Merge policy (decided up front, all exposed as arguments):
  * critical points within ``tol`` samples collapse to their median index;
  * clustering is **same-kind only** -- a decel CP at 100 and an accel CP at 103
    are different behavioural decisions and are never merged;
  * the kind of every surviving boundary is **relabelled after merging**, from
    the position-based slope across it, rather than inherited from whichever beta
    contributed it (inherited labels can conflict across levels);
  * ``min_phase_len`` optionally drops boundaries that would create phases
    shorter than a floor.  Default 0 = off (pure union), because the sweep
    should decide whether it is needed.

Units: everything here is index-space and unit-agnostic.  Detection runs on the
caller's arrays (native ft/s for NGSIM -- the PELT+ CUSUM thresholds are tuned
for ft/s); ``resegment_si`` carries the unit-free boundary indices onto SI arrays
and recomputes only the features.

Nothing in ``phase_segmentation.py`` is modified; ``Phase``,
``SegmentationResult`` and ``_phase_features`` are reused verbatim so the feature
definitions cannot drift between the two modules.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from phase_segmentation import (FEATURE_KEYS, Phase, SegmentationResult,
                                _phase_features)
from pelt_plus_class import PELTPlusDetection
from pure_pelt_class import PurePELTDetection

DETECTORS: Tuple[str, ...] = ("pelt_plus", "pelt", "dp")
PENALTY_GRID_DEFAULT: Tuple[float, ...] = (50, 200)
MERGE_TOL_DEFAULT: int = 5          # samples; 1.0 s at 10 Hz


# --------------------------------------------------------------------------- #
# Detector dispatch
# --------------------------------------------------------------------------- #
def _detect(method: str, t: np.ndarray, x: np.ndarray, v: np.ndarray,
            penalty: float, min_segment_length: int,
            cusum_threshold: float, cusum_drift: float
            ) -> Tuple[List[int], List[Dict], Dict]:
    """Run one detector; return (change_points, segments, diagnostics).

    Normalises two inconsistencies between the three modules:
      * Pure DP is a bare function taking (time, data), not a .detect(dict);
      * Pure DP returns no 'segments', so they are rebuilt with analyze_segments.

    ``segmented_regression`` is imported lazily: it hard-depends on numba, and
    that should not block the two PELT paths on installs without it.
    """
    m = str(method).strip().lower()

    if m in ("pelt_plus", "pelt+", "peltplus"):
        det = PELTPlusDetection(penalty=penalty,
                                min_segment_length=min_segment_length,
                                cusum_threshold=cusum_threshold,
                                cusum_drift=cusum_drift)
        cps, diag = det.detect({"time": t, "distance": x, "velocity": v})
        segs = diag.get("segments", [])

    elif m in ("pelt", "pure_pelt", "purepelt"):
        det = PurePELTDetection(penalty=penalty,
                                min_segment_length=min_segment_length)
        cps, diag = det.detect({"time": t, "distance": x, "velocity": v})
        segs = diag.get("segments", [])

    elif m in ("dp", "pure_dp", "puredp"):
        from segmented_regression import segmented_regression_dp, analyze_segments
        cps, diag = segmented_regression_dp(t, x, penalty, min_segment_length)
        cps = list(cps)
        segs = analyze_segments(t, x, list(cps))
        diag = dict(diag)
        diag["segments"] = segs

    else:
        raise ValueError(f"unknown detector {method!r}; use one of {DETECTORS}")

    diag = dict(diag)
    diag["detector"] = m
    diag["penalty_used"] = float(penalty)
    return list(cps), list(segs), diag


# --------------------------------------------------------------------------- #
# Segment -> critical point classification / phase construction
# --------------------------------------------------------------------------- #
def _classify(segments: Sequence[Dict]) -> Tuple[List[int], List[int]]:
    """Label each interior boundary accel/decel from the slope change.

    Mirrors phase_segmentation.segment_trajectory (itself lifted from
    analyzer_v6): the boundary is the *next* segment's start_idx, and a drop in
    position-slope (= speed) across it is a deceleration decision.
    """
    decel: List[int] = []
    accel: List[int] = []
    for i in range(len(segments) - 1):
        cp = int(segments[i + 1]["start_idx"])
        if segments[i + 1]["slope"] < segments[i]["slope"]:
            decel.append(cp)
        else:
            accel.append(cp)
    return decel, accel


def _build_phases(t: np.ndarray, x: np.ndarray, v: np.ndarray, s: np.ndarray,
                  critical_points: Sequence[int],
                  decel_points: Sequence[int], accel_points: Sequence[int],
                  n: int) -> List[Phase]:
    """Phases between endpoints + critical points (identical to Step 1)."""
    boundaries = sorted(set([0] + [int(c) for c in critical_points] + [n - 1]))
    boundaries = [b for b in boundaries if 0 <= b <= n - 1]
    decel_set, accel_set = set(decel_points), set(accel_points)
    cps = set(int(c) for c in critical_points)

    phases: List[Phase] = []
    for k in range(len(boundaries) - 1):
        i0, i1 = boundaries[k], boundaries[k + 1]
        if i1 <= i0:
            continue
        if not cps:
            kind = "single"
        elif i1 in decel_set:
            kind = "decel"
        elif i1 in accel_set:
            kind = "accel"
        else:
            kind = "single"                 # phase ending at the trajectory end
        phases.append(Phase(
            k=len(phases), i_start=i0, i_end=i1,
            t_start=float(t[i0]), t_end=float(t[i1]), kind=kind,
            features=_phase_features(t, x, v, s, i0, i1)))
    return phases


def _slopes_over(t: np.ndarray, x: np.ndarray,
                 spans: Sequence[Tuple[int, int]]) -> List[float]:
    """Position-vs-time slope (= mean speed) on each inclusive span."""
    out: List[float] = []
    for i0, i1 in spans:
        if i1 - i0 < 1:
            out.append(float("nan"))
            continue
        tt = t[i0:i1 + 1]
        xx = x[i0:i1 + 1]
        out.append(float(np.polyfit(tt, xx, 1)[0]))
    return out


# --------------------------------------------------------------------------- #
# Single-penalty segmentation (any detector)
# --------------------------------------------------------------------------- #
def segment_at(t: np.ndarray, x: np.ndarray, v: np.ndarray, s: np.ndarray,
               penalty: float = 75.0,
               method: str = "pelt_plus",
               min_segment_length: int = 20,
               cusum_threshold: float = 7.0,
               cusum_drift: float = 1.0) -> SegmentationResult:
    """Segment at one penalty with the chosen detector.

    With method='pelt_plus' this reproduces
    ``phase_segmentation.segment_trajectory`` exactly (verified by smoke test 1).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    s = np.asarray(s, float)
    n = len(t)

    _, segments, diagnostics = _detect(method, t, x, v, penalty,
                                       min_segment_length,
                                       cusum_threshold, cusum_drift)
    decel, accel = _classify(segments)
    critical_points = sorted(set(decel) | set(accel))
    phases = _build_phases(t, x, v, s, critical_points, decel, accel, n)

    return SegmentationResult(critical_points=critical_points,
                              decel_points=decel, accel_points=accel,
                              phases=phases, diagnostics=diagnostics)


def segment_multi(t: np.ndarray, x: np.ndarray, v: np.ndarray, s: np.ndarray,
                  penalties: Sequence[float] = PENALTY_GRID_DEFAULT,
                  method: str = "pelt_plus",
                  min_segment_length: int = 20,
                  cusum_threshold: float = 7.0,
                  cusum_drift: float = 1.0) -> List[SegmentationResult]:
    """One SegmentationResult per penalty, in the order given."""
    return [segment_at(t, x, v, s, penalty=float(b), method=method,
                       min_segment_length=min_segment_length,
                       cusum_threshold=cusum_threshold, cusum_drift=cusum_drift)
            for b in penalties]


# --------------------------------------------------------------------------- #
# Union merge
# --------------------------------------------------------------------------- #
def cluster_indices(indices: Sequence[int], tol: int) -> List[int]:
    """Collapse indices within ``tol`` samples to their median.

    Greedy, bounded-width: a cluster is closed once an index would sit more than
    ``tol`` from the cluster's *first* member, so no cluster can span more than
    ``tol`` (single-linkage chaining would let clusters drift arbitrarily far).
    """
    idx = sorted(set(int(i) for i in indices))
    if not idx:
        return []
    if tol <= 0:
        return idx
    reps: List[int] = []
    cur: List[int] = [idx[0]]
    for i in idx[1:]:
        if i - cur[0] <= tol:
            cur.append(i)
        else:
            reps.append(int(round(float(np.median(cur)))))
            cur = [i]
    reps.append(int(round(float(np.median(cur)))))
    return reps


def _enforce_min_phase(cps: Sequence[int], n: int,
                       min_phase_len: int) -> Tuple[List[int], int]:
    """Greedily drop CPs that would create a phase shorter than the floor."""
    if min_phase_len <= 0:
        return list(cps), 0
    kept: List[int] = []
    last = 0
    for cp in cps:
        if cp - last >= min_phase_len and (n - 1) - cp >= min_phase_len:
            kept.append(int(cp))
            last = int(cp)
    return kept, len(cps) - len(kept)


def merge_segmentations(segs: Sequence[SegmentationResult],
                        t: np.ndarray, x: np.ndarray, v: np.ndarray,
                        s: np.ndarray,
                        tol: int = MERGE_TOL_DEFAULT,
                        min_phase_len: int = 0) -> SegmentationResult:
    """Merge per-penalty segmentations into one union SegmentationResult.

    Each critical point is counted **once** (decision D-2 option B): the result
    is a single boundary set, a single phase sequence, and hence a single
    ordinary phase-anchored cost -- no per-level weighting is involved.

    NOTE this is deliberately *not* called for a one-element grid.  Merging
    relabels kinds from slopes measured over the merged spans, which need not
    reproduce the detector's own segment slopes bit-for-bit; the single-penalty
    path must stay identical to today's behaviour, so callers should use
    ``segment_at`` directly when len(penalties) == 1.
    """
    if not segs:
        raise ValueError("merge_segmentations needs >= 1 segmentation")
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    s = np.asarray(s, float)
    n = len(t)

    # -- 1. same-kind tolerance clustering --------------------------------- #
    decel_all = [c for sg in segs for c in sg.decel_points]
    accel_all = [c for sg in segs for c in sg.accel_points]
    decel_reps = cluster_indices(decel_all, tol)
    accel_reps = cluster_indices(accel_all, tol)

    union = sorted(set(decel_reps) | set(accel_reps))
    union = [c for c in union if 0 < c < n - 1]

    # -- 2. optional minimum-phase-length floor ---------------------------- #
    union, n_dropped = _enforce_min_phase(union, n, min_phase_len)

    # -- 3. relabel kinds from slopes over the MERGED spans ----------------- #
    boundaries = sorted(set([0] + union + [n - 1]))
    spans = [(boundaries[k], boundaries[k + 1])
             for k in range(len(boundaries) - 1)]
    slopes = _slopes_over(t, x, spans)
    decel: List[int] = []
    accel: List[int] = []
    for k in range(len(spans) - 1):
        cp = boundaries[k + 1]
        if slopes[k + 1] < slopes[k]:
            decel.append(cp)
        else:
            accel.append(cp)
    critical_points = sorted(set(decel) | set(accel))

    phases = _build_phases(t, x, v, s, critical_points, decel, accel, n)

    diagnostics = {
        "merge": {
            "penalties": [sg.diagnostics.get("penalty_used") for sg in segs],
            "detector": segs[0].diagnostics.get("detector"),
            "tol": int(tol),
            "min_phase_len": int(min_phase_len),
            "n_cp_per_level": [len(sg.critical_points) for sg in segs],
            "n_cp_pooled_raw": len(decel_all) + len(accel_all),
            "n_cp_union": len(critical_points),
            "n_cp_dropped_min_phase": int(n_dropped),
            "n_decel": len(decel), "n_accel": len(accel),
        },
        "levels": [sg.diagnostics for sg in segs],
    }
    return SegmentationResult(critical_points=critical_points,
                              decel_points=decel, accel_points=accel,
                              phases=phases, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# Unit transfer (detect in native units, features in SI)
# --------------------------------------------------------------------------- #
def resegment_si(seg_native: SegmentationResult, t: np.ndarray,
                 x_si: np.ndarray, v_si: np.ndarray,
                 s_si: np.ndarray) -> SegmentationResult:
    """Carry natively-detected boundary indices onto SI arrays.

    Boundary *indices* are unit-invariant; only the per-phase features depend on
    units, so only those are recomputed.  (Same contract as
    ``run_experiment._resegment_si``, which can delegate here.)
    """
    phases_si = [
        Phase(k=ph.k, i_start=ph.i_start, i_end=ph.i_end,
              t_start=float(t[ph.i_start]), t_end=float(t[ph.i_end]),
              kind=ph.kind,
              features=_phase_features(t, x_si, v_si, s_si,
                                       ph.i_start, ph.i_end))
        for ph in seg_native.phases
    ]
    return SegmentationResult(critical_points=seg_native.critical_points,
                              decel_points=seg_native.decel_points,
                              accel_points=seg_native.accel_points,
                              phases=phases_si,
                              diagnostics=seg_native.diagnostics)


def segment_arrays(t: np.ndarray, x: np.ndarray, v: np.ndarray, s: np.ndarray,
                   penalties: Sequence[float] = PENALTY_GRID_DEFAULT,
                   method: str = "pelt",
                   merge_tol: int = MERGE_TOL_DEFAULT,
                   min_phase_len: int = 0,
                   min_segment_length: int = 20,
                   cusum_threshold: float = 7.0,
                   cusum_drift: float = 1.0) -> SegmentationResult:
    """Grid -> union in one call.  A one-element grid bypasses the merge."""
    pens = [float(b) for b in penalties]
    if len(pens) == 1:
        return segment_at(t, x, v, s, penalty=pens[0], method=method,
                          min_segment_length=min_segment_length,
                          cusum_threshold=cusum_threshold,
                          cusum_drift=cusum_drift)
    segs = segment_multi(t, x, v, s, penalties=pens, method=method,
                         min_segment_length=min_segment_length,
                         cusum_threshold=cusum_threshold,
                         cusum_drift=cusum_drift)
    return merge_segmentations(segs, t, x, v, s, tol=merge_tol,
                               min_phase_len=min_phase_len)


def parse_penalties(spec: str) -> List[float]:
    """'50,100,150,200' -> [50.0, 100.0, 150.0, 200.0].  Any length, incl. 1."""
    vals = [float(p.strip()) for p in str(spec).split(",") if p.strip()]
    if not vals:
        raise ValueError("--penalties parsed to an empty grid")
    return vals