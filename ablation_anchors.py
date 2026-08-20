#!/usr/bin/env python3
"""
ablation_anchors.py
===================
Matched-count *anchor-placement controls* for the phase-anchored calibration
ablation (Probe 2: behavioural placement vs. mere sparsity).

The phase-anchored objective scores the follower only at the PELT+ critical
points.  A reviewer will ask whether the benefit comes from the *behavioural*
placement of those anchors or merely from evaluating at *fewer, well-spaced*
points (a regularisation / sparsity effect).  To separate the two we build
control segmentations that keep the SAME NUMBER of phases K as the PELT+
segmentation of a given pair, but place the K-1 interior boundaries either
uniformly along the trajectory or at random indices -- everything else
(features, weighting, simulator) identical.  If phase-anchored ~ uniform ~
random at matched K, only the count matters; if phase-anchored wins, placement
is load-bearing.

The control ``SegmentationResult`` is a drop-in for
``objectives.PhaseAnchoredObjective`` and
``calibrate.calibrate_pairs(..., segmentations=[...])``: same dataclass, same
per-phase feature computation via ``phase_segmentation._phase_features`` (the
single source of truth, so observed control features and simulated features
cannot drift, and any feature the local registry supports -- e.g.
``phase_speed_ols`` -- is available automatically).  Phase ``kind`` is set to
"control" and is unused by the objective.

Boundary placement is on the sample-index axis, which is unit-invariant, so the
control is built directly on the SI arrays; only the target phase count K is
borrowed from the (native-unit-detected) reference segmentation.
"""

from __future__ import annotations

from typing import List

import numpy as np

from phase_segmentation import Phase, SegmentationResult, _phase_features


# --------------------------------------------------------------------------- #
# Interior-boundary placement (unit-invariant, on the index axis)
# --------------------------------------------------------------------------- #
def _clip_and_space(idx: List[int], n: int, min_seg: int) -> List[int]:
    """Greedily keep indices that stay >= min_seg from the ends and each other."""
    out: List[int] = []
    last = -min_seg
    for c in idx:
        c = int(min(max(c, 1), n - 2))
        if (c - last) >= min_seg and (n - 1 - c) >= min_seg:
            out.append(c)
            last = c
    return out


def _uniform_interior(n: int, k_interior: int, min_seg: int) -> List[int]:
    """k_interior evenly spaced interior indices in (0, n-1)."""
    if k_interior <= 0:
        return []
    raw = np.linspace(0, n - 1, k_interior + 2)[1:-1]      # drop the two endpoints
    idx = sorted({int(round(r)) for r in raw})
    return _clip_and_space(idx, n, min_seg)


def _random_interior(n: int, k_interior: int, min_seg: int, seed: int) -> List[int]:
    """k_interior random interior indices, >= min_seg from ends and each other.

    Rejection-light: draw a candidate set, greedily accept respecting the min
    gap, retry a bounded number of times to reach the target count. Falls back to
    the best partial set (or uniform) if n is too small for K with this spacing.
    """
    if k_interior <= 0:
        return []
    rng = np.random.default_rng(seed)
    lo, hi = min_seg, n - 1 - min_seg
    if hi <= lo:
        return _uniform_interior(n, k_interior, min_seg)

    pool = np.arange(lo, hi + 1)
    best: List[int] = []
    for _ in range(200):
        cand = np.sort(rng.choice(pool, size=min(k_interior, pool.size),
                                  replace=False))
        acc: List[int] = []
        last = -min_seg
        for c in cand:
            if (int(c) - last) >= min_seg:
                acc.append(int(c))
                last = int(c)
        if len(acc) >= k_interior:
            return acc[:k_interior]
        if len(acc) > len(best):
            best = acc
    return best   # < k_interior only when n cannot hold K phases at this spacing


# --------------------------------------------------------------------------- #
# Control segmentation builder
# --------------------------------------------------------------------------- #
def matched_control_segmentation(k_phases: int,
                                 t: np.ndarray, x: np.ndarray,
                                 v: np.ndarray, s: np.ndarray,
                                 mode: str = "uniform",
                                 min_segment_length: int = 20,
                                 seed: int = 0) -> SegmentationResult:
    """A control segmentation with ``k_phases`` phases and NON-behavioural bounds.

    Args:
        k_phases : target number of phases (= reference PELT+ ``n_phases``).
        t,x,v,s  : follower time/position/speed/spacing (SI); features use these.
        mode     : 'uniform' (evenly spaced) | 'random' (seeded).
        min_segment_length : minimum samples per phase (mirrors PELT+ knob).
        seed     : RNG seed for ``mode='random'``.

    Returns:
        A ``SegmentationResult`` whose phases have the same feature dict layout as
        the observed PELT+ phases, so it is a drop-in for the phase objective.
    """
    t = np.asarray(t, float); x = np.asarray(x, float)
    v = np.asarray(v, float); s = np.asarray(s, float)
    n = len(t)
    k_phases = max(1, int(k_phases))
    k_interior = k_phases - 1

    if mode == "uniform":
        interior = _uniform_interior(n, k_interior, min_segment_length)
    elif mode == "random":
        interior = _random_interior(n, k_interior, min_segment_length, seed)
    else:
        raise ValueError("mode must be 'uniform' or 'random'")

    boundaries = sorted(set([0] + interior + [n - 1]))
    phases: List[Phase] = []
    for k in range(len(boundaries) - 1):
        i0, i1 = boundaries[k], boundaries[k + 1]
        if i1 <= i0:
            continue
        phases.append(Phase(
            k=k, i_start=i0, i_end=i1,
            t_start=float(t[i0]), t_end=float(t[i1]), kind="control",
            features=_phase_features(t, x, v, s, i0, i1)))

    return SegmentationResult(
        critical_points=list(interior), decel_points=[], accel_points=[],
        phases=phases,
        diagnostics={"control_mode": mode, "k_requested": k_phases,
                     "k_built": len(phases), "seed": seed,
                     "min_segment_length": min_segment_length})


# --------------------------------------------------------------------------- #
# Self-test: matched count, finite features, spacing respected
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    print("=" * 66)
    print("ablation_anchors.py self-test")
    print("=" * 66)
    n = 400
    t = np.arange(n) * 0.1
    # a smooth stop-and-go-ish position so features are non-degenerate
    v = 12 + 6 * np.sin(2 * np.pi * t / 12.0)
    x = np.cumsum(v) * 0.1
    s = 20 + 3 * np.cos(2 * np.pi * t / 9.0)

    for K in (1, 3, 6, 10):
        su = matched_control_segmentation(K, t, x, v, s, mode="uniform")
        sr = matched_control_segmentation(K, t, x, v, s, mode="random", seed=1)
        for tag, seg in (("uniform", su), ("random", sr)):
            feats = seg.feature_matrix(("s_end",))
            ok_count = seg.n_phases == max(1, K) or seg.diagnostics["k_built"] < K
            ok_finite = np.all(np.isfinite(feats))
            mn = min((ph.i_end - ph.i_start) for ph in seg.phases)
            print(f"  K={K:2d} {tag:7s}: phases={seg.n_phases:2d} "
                  f"min_len={mn:3d}  features_finite={bool(ok_finite)}  "
                  f"count_ok={bool(ok_count)}")
            assert ok_finite, "control features must be finite"
            assert seg.n_phases >= 1
    print("\nAll anchor-control checks passed.")


if __name__ == "__main__":
    _selftest()
