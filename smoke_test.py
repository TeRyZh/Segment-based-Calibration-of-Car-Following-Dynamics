#!/usr/bin/env python3
"""Smoke tests for phase_multiscale.py + penalty_sweep.py (synthetic pairs)."""
import numpy as np

# The /mnt/project snapshot of phase_segmentation.py still calls np.trapz, which
# numpy 2.x removed. Terry's local copy has the _trapz alias; shim for the test.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from phase_segmentation import segment_trajectory
from phase_multiscale import (cluster_indices, merge_segmentations, segment_at,
                              segment_arrays, segment_multi)
from penalty_sweep import fig_counts, fig_overlap, summarise, sweep_arrays

FT = 0.3048


def synth(seed=0, n=900, dt=0.1):
    """Stop-and-go follower in NATIVE ft / ft-s (what PELT+ is tuned for)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    v = 45.0 + 18.0 * np.sin(2 * np.pi * t / 22.0) + 7.0 * np.sin(2 * np.pi * t / 7.5 + seed)
    v = np.clip(v, 2.0, None)                       # ft/s
    x = np.cumsum(v) * dt + rng.normal(0, 0.25, n)  # ft, NGSIM-ish position noise
    s = 40.0 + 12.0 * np.cos(2 * np.pi * t / 22.0) + rng.normal(0, 0.3, n)
    s = np.clip(s, 8.0, None)
    return t, x, v, s


def eq_seg(a, b):
    if a.critical_points != b.critical_points:
        return f"CPs differ: {a.critical_points} vs {b.critical_points}"
    if a.decel_points != b.decel_points or a.accel_points != b.accel_points:
        return "kind labels differ"
    if a.n_phases != b.n_phases:
        return f"n_phases {a.n_phases} vs {b.n_phases}"
    for p, q in zip(a.phases, b.phases):
        if (p.i_start, p.i_end, p.kind) != (q.i_start, q.i_end, q.kind):
            return f"phase span/kind differ: {p} vs {q}"
        for k in p.features:
            if abs(p.features[k] - q.features[k]) > 1e-12:
                return f"feature {k} differs: {p.features[k]} vs {q.features[k]}"
    return None


# --------------------------------------------------------------------------- #
print("=" * 68)
print("TEST 1  segment_at(pelt_plus, 75) == segment_trajectory(75)")
print("=" * 68)
fails = 0
for seed in (0, 1, 2):
    t, x, v, s = synth(seed)
    old = segment_trajectory(t, x, v, s, penalty=75.0)
    new = segment_at(t, x, v, s, penalty=75.0, method="pelt_plus")
    err = eq_seg(old, new)
    print(f"  seed {seed}: {old.n_phases:2d} phases, {len(old.critical_points):2d} CPs"
          f"   -> {'IDENTICAL' if err is None else 'MISMATCH: ' + err}")
    fails += err is not None
# one-element grid must take the bypass and stay identical
t, x, v, s = synth(0)
err = eq_seg(segment_trajectory(t, x, v, s, penalty=75.0),
             segment_arrays(t, x, v, s, penalties=[75.0], method="pelt_plus"))
print(f"  segment_arrays(grid=[75]) bypass -> {'IDENTICAL' if err is None else err}")
fails += err is not None
print(f"  {'PASS' if fails == 0 else 'FAIL'}  (single-penalty path is unchanged)")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 68)
print("TEST 2  union merge sanity")
print("=" * 68)
print(f"  cluster_indices([10,13,14,60,61,200], tol=10) -> "
      f"{cluster_indices([10, 13, 14, 60, 61, 200], 10)}   (expect [13, 60, 200])")

GRID = [50.0, 100.0, 150.0, 200.0]
TOL = 10
for seed in (0, 1):
    t, x, v, s = synth(seed)
    segs = segment_multi(t, x, v, s, penalties=GRID, method="pelt_plus")
    u = merge_segmentations(segs, t, x, v, s, tol=TOL)
    per_level = [len(sg.critical_points) for sg in segs]

    cps = u.critical_points
    assert cps == sorted(set(cps)), "union CPs not sorted/unique"
    assert all(0 < c < len(t) - 1 for c in cps), "CP at or past an endpoint"
    assert set(u.decel_points) | set(u.accel_points) == set(cps), "label/CP mismatch"
    assert not (set(u.decel_points) & set(u.accel_points)), "CP labelled twice"
    spans = [(p.i_start, p.i_end) for p in u.phases]
    assert all(b[0] == a[1] for a, b in zip(spans, spans[1:])), "phases not contiguous"
    assert spans[0][0] == 0 and spans[-1][1] == len(t) - 1, "phases don't span trajectory"
    # same-kind separation >= tol (cross-kind proximity is allowed by design)
    for kind_pts in (u.decel_points, u.accel_points):
        gaps = np.diff(sorted(kind_pts))
        assert all(g > TOL for g in gaps), f"same-kind CPs within tol: {kind_pts}"

    short = sum(1 for p in u.phases if p.n_samples < 20)
    print(f"  seed {seed}: per-level CPs {per_level} -> union {len(cps)}"
          f"  ({len(u.decel_points)}d/{len(u.accel_points)}a), "
          f"{u.n_phases} phases, {short} shorter than 20 samples")

# min_phase_len actually bites
u0 = merge_segmentations(segs, t, x, v, s, tol=TOL, min_phase_len=0)
u30 = merge_segmentations(segs, t, x, v, s, tol=TOL, min_phase_len=30)
assert len(u30.critical_points) <= len(u0.critical_points)
assert all(p.n_samples >= 30 for p in u30.phases[:-1]), "floor not enforced"
print(f"  min_phase_len 0 -> {len(u0.critical_points)} CPs; "
      f"30 -> {len(u30.critical_points)} CPs "
      f"(dropped {u30.diagnostics['merge']['n_cp_dropped_min_phase']})")
print("  PASS")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 68)
print("TEST 3  detectors + sweep_arrays/summarise/figures")
print("=" * 68)
t, x, v, s = synth(0)
for m in ("pelt_plus", "pelt"):
    sg = segment_at(t, x, v, s, penalty=100.0, method=m)
    print(f"  detector={m:9s}: {len(sg.critical_points):2d} CPs, {sg.n_phases:2d} phases")
try:
    sg = segment_at(t[:200], x[:200], v[:200], s[:200], penalty=100.0, method="dp")
    print(f"  detector=dp       : {len(sg.critical_points):2d} CPs, {sg.n_phases:2d} phases")
except ImportError as e:
    print(f"  detector=dp       : SKIPPED ({e}) -- numba absent here, lazy import held")

rows, cps_by_level = [], {f"{p:g}": [] for p in GRID}
cps_by_level["union"] = []
for seed in (0, 1, 2):
    t, x, v, s = synth(seed)
    r, levels = sweep_arrays(f"pair_{seed}", t, x, v, s,
                             x * FT, v * FT, s * FT,          # SI features
                             penalties=GRID, method="pelt_plus", merge_tol=TOL)
    rows.extend(r)
    for lab, sg in levels.items():
        cps_by_level[lab].append(list(sg.critical_points))

summary = summarise(rows, cps_by_level, GRID, TOL, 20)
print(f"\n  {'level':>7}{'CP/pair':>10}{'phases/pair':>13}{'s_end mean':>12}")
for lab in summary["levels"]:
    pl = summary["per_level"][lab]
    print(f"  {lab:>7}{pl['n_cp_per_pair']['mean']:10.2f}"
          f"{pl['n_phases_per_pair']['mean']:13.2f}"
          f"{pl['features']['s_end']['mean']:12.2f}")

uv = summary["union_vs_finest"]
print(f"\n  union/finest ratio      = {uv['size_ratio_mean']:.3f}")
print(f"  finest covered by union = {uv['frac_finest_covered_by_union']['mean']:.3f}")
print(f"  union adds beyond finest= {uv['cps_union_adds_beyond_finest']['mean']:.2f} CPs/pair")
J = np.array(summary["overlap"]["jaccard_mean"])
print(f"  Jaccard(50,100)={J[0,1]:.2f}  (50,200)={J[0,3]:.2f}  (150,200)={J[2,3]:.2f}")
mh = summary["merge_hygiene"]
print(f"  union phases < 20 samples: {mh['n_phases_below_floor']} "
      f"({mh['frac_phases_below_floor']*100:.1f}%)")

fig_counts(rows, summary["levels"], "/tmp/fig_penalty_counts.png")
fig_overlap(summary, "/tmp/fig_penalty_overlap.png")
import os
print(f"  figures rendered: counts={os.path.getsize('/tmp/fig_penalty_counts.png')}B, "
      f"overlap={os.path.getsize('/tmp/fig_penalty_overlap.png')}B")
print("  PASS")
print("\nall smoke tests done.")
