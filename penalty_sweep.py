#!/usr/bin/env python3
"""
penalty_sweep.py
================
Step 2: what does the penalty beta actually do to the phase features?

The phase-anchored objective's one free hyperparameter is the segmentation
penalty.  This script sweeps a grid, reports the phase features at every level
and at their union, and quantifies how much the levels actually differ.  Three
questions it is built to answer:

1. **Sensitivity.**  How do critical-point count, phase count, phase duration and
   the boundary features {s_end, dist, ...} move with beta?

2. **Nesting.**  Are the coarse boundary sets contained in the fine ones?
   Reported as pairwise tolerance-Jaccard, containment, and Chamfer distance
   between levels.  This is the load-bearing number: *if* C_50 already contains
   most of C_100, C_150, C_200, the union is C_50 with a few strays and the
   multi-penalty framing buys nothing over just picking beta = 50.  The
   ``union_vs_finest`` block reports that ratio directly.

3. **Merge hygiene.**  The union's phase-length distribution, and how many phases
   fall below ``min_segment_length``.  min_segment_length is enforced *within*
   each beta but not *across* them, so the union can contain short phases whose
   features are near-copies of their neighbours'.  ``n_phases_below_floor``
   decides whether ``--min-phase-len`` needs to be turned on.

Detection runs on native units (feet for NGSIM -- the PELT+ CUSUM thresholds are
tuned for ft/s and SI under-detects); reported features are SI, via the same
unit-free index transfer the calibration experiment uses.

Outputs (to --out, default ./penalty_sweep_out)
    penalty_sweep.csv           long form: pair x level x phase x features
    penalty_sweep_summary.json  per-level aggregates + overlap matrices
    fig_penalty_counts.png      CP / phase counts vs beta (+ union)
    fig_penalty_overlap.png     Jaccard + containment heatmaps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_multiscale import (DETECTORS, MERGE_TOL_DEFAULT,
                              PENALTY_GRID_DEFAULT, merge_segmentations,
                              parse_penalties, resegment_si, segment_multi)
from phase_segmentation import SegmentationResult

FT_TO_M = 0.3048


# --------------------------------------------------------------------------- #
# Set-similarity between critical-point sets
# --------------------------------------------------------------------------- #
def _match(A: Sequence[int], B: Sequence[int], tol: int) -> int:
    """Greedy nearest-first one-to-one matching within ``tol`` samples."""
    cand = sorted((abs(a - b), i, j)
                  for i, a in enumerate(A) for j, b in enumerate(B)
                  if abs(a - b) <= tol)
    ua, ub, m = set(), set(), 0
    for _, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); m += 1
    return m


def jaccard_tol(A: Sequence[int], B: Sequence[int], tol: int) -> float:
    if not A and not B:
        return 1.0
    m = _match(A, B, tol)
    denom = len(A) + len(B) - m
    return float(m / denom) if denom > 0 else 1.0


def containment(A: Sequence[int], B: Sequence[int], tol: int) -> float:
    """Fraction of A that has a match in B (nesting: is coarse inside fine?)."""
    if not A:
        return float("nan")
    return float(_match(A, B, tol) / len(A))


def chamfer(A: Sequence[int], B: Sequence[int]) -> float:
    """Symmetric mean nearest-neighbour distance, in samples."""
    if not A or not B:
        return float("nan")
    a = np.asarray(A, float)
    b = np.asarray(B, float)
    d1 = float(np.mean([np.min(np.abs(b - x)) for x in a]))
    d2 = float(np.mean([np.min(np.abs(a - x)) for x in b]))
    return 0.5 * (d1 + d2)


# --------------------------------------------------------------------------- #
# Core sweep on plain arrays (no cf_data dependency -- testable standalone)
# --------------------------------------------------------------------------- #
def sweep_arrays(name: str,
                 t: np.ndarray, x_nat: np.ndarray, v_nat: np.ndarray,
                 s_nat: np.ndarray,
                 x_si: np.ndarray, v_si: np.ndarray, s_si: np.ndarray,
                 penalties: Sequence[float],
                 method: str = "pelt_plus",
                 merge_tol: int = MERGE_TOL_DEFAULT,
                 min_phase_len: int = 0,
                 min_segment_length: int = 20,
                 cusum_threshold: float = 7.0,
                 cusum_drift: float = 1.0
                 ) -> Tuple[List[Dict], Dict[str, SegmentationResult]]:
    """Segment one follower at every beta plus the union.

    ``*_nat`` arrays drive detection (native units); ``*_si`` arrays are what the
    reported features are computed on.  Returns (long-form rows, {level: seg}).
    """
    segs_nat = segment_multi(t, x_nat, v_nat, s_nat, penalties=penalties,
                             method=method,
                             min_segment_length=min_segment_length,
                             cusum_threshold=cusum_threshold,
                             cusum_drift=cusum_drift)
    union_nat = merge_segmentations(segs_nat, t, x_nat, v_nat, s_nat,
                                    tol=merge_tol, min_phase_len=min_phase_len)

    levels: Dict[str, SegmentationResult] = {}
    for b, sg in zip(penalties, segs_nat):
        levels[f"{float(b):g}"] = resegment_si(sg, t, x_si, v_si, s_si)
    levels["union"] = resegment_si(union_nat, t, x_si, v_si, s_si)

    rows: List[Dict] = []
    for label, sg in levels.items():
        for ph in sg.phases:
            row = {"pair": name, "level": label,
                   "n_cp": len(sg.critical_points), "n_phases": sg.n_phases,
                   "k": ph.k, "i_start": ph.i_start, "i_end": ph.i_end,
                   "t_start": round(ph.t_start, 3), "t_end": round(ph.t_end, 3),
                   "kind": ph.kind, "phase_dur_s": round(ph.duration, 3),
                   "n_samples": ph.n_samples}
            row.update({k: float(val) for k, val in ph.features.items()})
            rows.append(row)
    return rows, levels


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _stats(a: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([x for x in a if np.isfinite(x)], float)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def summarise(rows: Sequence[Dict], cps_by_level: Dict[str, List[List[int]]],
              penalties: Sequence[float], merge_tol: int,
              min_segment_length: int) -> Dict:
    labels = [f"{float(b):g}" for b in penalties] + ["union"]
    feature_keys = sorted({k for r in rows
                           for k in r
                           if k not in ("pair", "level", "n_cp", "n_phases",
                                        "k", "i_start", "i_end", "t_start",
                                        "t_end", "kind", "n_samples")})

    # ---- per-level aggregates ---- #
    per_level: Dict[str, Dict] = {}
    for lab in labels:
        sub = [r for r in rows if r["level"] == lab]
        if not sub:
            continue
        per_pair = {}
        for r in sub:
            per_pair.setdefault(r["pair"], r)
        per_level[lab] = {
            "n_cp_per_pair": _stats([r["n_cp"] for r in per_pair.values()]),
            "n_phases_per_pair": _stats([r["n_phases"]
                                         for r in per_pair.values()]),
            "kind_counts": {k: sum(1 for r in sub if r["kind"] == k)
                            for k in ("accel", "decel", "single")},
            "features": {fk: _stats([r[fk] for r in sub if fk in r])
                         for fk in feature_keys},
        }

    # ---- pairwise overlap between beta levels (mean over pairs) ---- #
    blabels = [f"{float(b):g}" for b in penalties]
    n_lv = len(blabels)
    jac = np.full((n_lv, n_lv), np.nan)
    con = np.full((n_lv, n_lv), np.nan)
    cham = np.full((n_lv, n_lv), np.nan)
    for i, li in enumerate(blabels):
        for j, lj in enumerate(blabels):
            js, cs, ch = [], [], []
            for A, B in zip(cps_by_level[li], cps_by_level[lj]):
                js.append(jaccard_tol(A, B, merge_tol))
                cs.append(containment(A, B, merge_tol))
                ch.append(chamfer(A, B))
            jac[i, j] = float(np.nanmean(js)) if js else np.nan
            con[i, j] = float(np.nanmean(cs)) if cs else np.nan
            cham[i, j] = float(np.nanmean(ch)) if ch else np.nan

    # ---- does the union buy anything over the finest level? ---- #
    finest = f"{float(min(penalties)):g}"
    u_sizes = [len(c) for c in cps_by_level["union"]]
    f_sizes = [len(c) for c in cps_by_level[finest]]
    contain_finest_in_union = [containment(F, U, merge_tol)
                               for F, U in zip(cps_by_level[finest],
                                               cps_by_level["union"])]
    union_extra = [len(U) - _match(U, F, merge_tol)
                   for U, F in zip(cps_by_level["union"],
                                   cps_by_level[finest])]

    # ---- merge hygiene: union phases below the within-level floor ---- #
    u_rows = [r for r in rows if r["level"] == "union"]
    n_below = sum(1 for r in u_rows if r["n_samples"] < min_segment_length)

    return {
        "levels": labels,
        "feature_keys": feature_keys,
        "per_level": per_level,
        "overlap": {
            "tol_samples": int(merge_tol),
            "labels": blabels,
            "jaccard_mean": jac.tolist(),
            "containment_mean_row_in_col": con.tolist(),
            "chamfer_mean_samples": cham.tolist(),
        },
        "union_vs_finest": {
            "finest_level": finest,
            "union_size": _stats(u_sizes),
            "finest_size": _stats(f_sizes),
            "size_ratio_mean": (float(np.mean(u_sizes) / np.mean(f_sizes))
                                if np.mean(f_sizes) > 0 else float("nan")),
            "frac_finest_covered_by_union": _stats(contain_finest_in_union),
            "cps_union_adds_beyond_finest": _stats(union_extra),
        },
        "merge_hygiene": {
            "min_segment_length": int(min_segment_length),
            "union_phase_len_samples": _stats([r["n_samples"] for r in u_rows]),
            "union_phase_dur_s": _stats([r["phase_dur_s"] for r in u_rows]),
            "n_phases_below_floor": int(n_below),
            "frac_phases_below_floor": (float(n_below / len(u_rows))
                                        if u_rows else float("nan")),
        },
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_counts(rows: Sequence[Dict], labels: Sequence[str], path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylab in ((axes[0], "n_cp", "critical points per pair"),
                          (axes[1], "n_phases", "phases per pair")):
        data = []
        for lab in labels:
            per_pair = {}
            for r in rows:
                if r["level"] == lab:
                    per_pair[r["pair"]] = r[key]
            data.append(list(per_pair.values()))
        bp = ax.boxplot(data, showmeans=True, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#c0392b" if labels[i] == "union" else "#2471a3")
            patch.set_alpha(0.35)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlabel("penalty  beta")
        ax.set_ylabel(ylab)
    fig.suptitle("Segmentation sensitivity to the penalty", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_overlap(summary: Dict, path: str) -> None:
    ov = summary["overlap"]
    labs = ov["labels"]
    mats = [(np.array(ov["jaccard_mean"], float),
             f"tolerance-Jaccard (tol={ov['tol_samples']} samples)"),
            (np.array(ov["containment_mean_row_in_col"], float),
             "containment: row's CPs found in column")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (M, title) in zip(axes, mats):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
        ax.set_xlabel("beta"); ax.set_ylabel("beta")
        ax.set_title(title, fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Boundary-set persistence across penalties", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="folder containing manifest.csv (+ train/ test subdirs)")
    ap.add_argument("--split", default="train", help="train | test")
    ap.add_argument("--units", default="feet", help="feet | si | auto")
    ap.add_argument("--detector", default="pelt_plus", choices=DETECTORS)
    ap.add_argument("--penalties",
                    default=",".join(f"{p:g}" for p in PENALTY_GRID_DEFAULT),
                    help="comma-separated grid, any length "
                         "(e.g. 50,100,150,200 or 25,50,75,100,150,200,300)")
    ap.add_argument("--merge-tol", type=int, default=MERGE_TOL_DEFAULT,
                    help="samples; CPs closer than this collapse (10 = 1.0 s)")
    ap.add_argument("--min-phase-len", type=int, default=0,
                    help="union-level phase floor in samples; 0 = off")
    ap.add_argument("--min-segment-length", type=int, default=20)
    ap.add_argument("--cusum-threshold", type=float, default=7.0)
    ap.add_argument("--cusum-drift", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="penalty_sweep_out")
    args = ap.parse_args(argv)

    from cf_data import load_folder                      # lazy: heavy import

    os.makedirs(args.out, exist_ok=True)
    penalties = parse_penalties(args.penalties)
    labels = [f"{p:g}" for p in penalties] + ["union"]

    pairs = load_folder(args.root, split=args.split, units=args.units,
                        limit=args.limit)
    if not pairs:
        sys.exit(f"loaded 0 pairs from {args.root} (split={args.split})")
    print(f"[load] {len(pairs)} pairs (split={args.split}, units={args.units})")
    print(f"[sweep] detector={args.detector}  grid={penalties}  "
          f"tol={args.merge_tol}  min_phase_len={args.min_phase_len}")

    rows: List[Dict] = []
    cps_by_level: Dict[str, List[List[int]]] = {lab: [] for lab in labels}

    for n_done, pair in enumerate(pairs, 1):
        if pair.units_source.startswith("feet"):
            inv = 1.0 / FT_TO_M
            nat = (pair.x_follower * inv, pair.v_follower * inv,
                   pair.spacing * inv)
        else:
            nat = (pair.x_follower, pair.v_follower, pair.spacing)
        r, levels = sweep_arrays(
            pair.name, pair.t, nat[0], nat[1], nat[2],
            pair.x_follower, pair.v_follower, pair.spacing,
            penalties=penalties, method=args.detector,
            merge_tol=args.merge_tol, min_phase_len=args.min_phase_len,
            min_segment_length=args.min_segment_length,
            cusum_threshold=args.cusum_threshold, cusum_drift=args.cusum_drift)
        rows.extend(r)
        for lab in labels:
            cps_by_level[lab].append(list(levels[lab].critical_points))
        if n_done % 25 == 0 or n_done == len(pairs):
            print(f"    {n_done}/{len(pairs)} pairs segmented")

    summary = summarise(rows, cps_by_level, penalties, args.merge_tol,
                        args.min_segment_length)
    summary["config"] = {
        "root": args.root, "split": args.split, "units": args.units,
        "detector": args.detector, "penalties": penalties,
        "merge_tol": args.merge_tol, "min_phase_len": args.min_phase_len,
        "min_segment_length": args.min_segment_length,
        "cusum_threshold": args.cusum_threshold,
        "cusum_drift": args.cusum_drift, "n_pairs": len(pairs),
    }

    # ---- console report ---- #
    print(f"\n{'level':>8}{'CP/pair':>12}{'phases/pair':>14}"
          f"{'decel':>8}{'accel':>8}")
    for lab in labels:
        pl = summary["per_level"][lab]
        print(f"{lab:>8}{pl['n_cp_per_pair']['mean']:12.2f}"
              f"{pl['n_phases_per_pair']['mean']:14.2f}"
              f"{pl['kind_counts']['decel']:8d}{pl['kind_counts']['accel']:8d}")

    uv = summary["union_vs_finest"]
    print(f"\n[nesting] union vs finest level (beta={uv['finest_level']}):")
    print(f"    union/finest CP-count ratio  = {uv['size_ratio_mean']:.3f}")
    print(f"    finest covered by union      = "
          f"{uv['frac_finest_covered_by_union']['mean']:.3f}")
    print(f"    CPs union adds beyond finest = "
          f"{uv['cps_union_adds_beyond_finest']['mean']:.2f} per pair")
    print("    -> ratio ~1.0 and few added CPs means the grid is nested and the "
          "union\n       is just the finest level; the multi-penalty framing "
          "would buy nothing.")

    mh = summary["merge_hygiene"]
    print(f"\n[hygiene] union phases shorter than min_segment_length="
          f"{mh['min_segment_length']}: {mh['n_phases_below_floor']} "
          f"({mh['frac_phases_below_floor']*100:.1f}%)")
    print("    -> if this is non-trivial, turn on --min-phase-len.")

    # ---- persist ---- #
    csv_path = os.path.join(args.out, "penalty_sweep.csv")
    cols = (["pair", "level", "n_cp", "n_phases", "k", "i_start", "i_end",
             "t_start", "t_end", "kind", "phase_dur_s", "n_samples"]
            + summary["feature_keys"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "penalty_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig_counts(rows, labels, os.path.join(args.out, "fig_penalty_counts.png"))
    fig_overlap(summary, os.path.join(args.out, "fig_penalty_overlap.png"))

    print(f"\n[out] wrote penalty_sweep.csv, penalty_sweep_summary.json, "
          f"2 figures -> {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
