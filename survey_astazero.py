#!/usr/bin/env python3
"""
survey_astazero.py
==================
Folder-wide survey + selection over the OpenACC AstaZero platoon runs, to pick
ONE experiment file and ONE shared ("common") window for the platoon-propagation
visualisation -- the figure in which a perturbation enters at the lead vehicle
and cascades down the platoon with staggered response lags.

This is a THIN driver. Every substantive computation is imported VERBATIM from
`acc_controller_behavior.py`:

    load_openacc_asta   -- preamble/IVS loader (arbitrary vehicle count)
    analyze_pair        -- per-pair PELT+ critical points + matched connectors
    _common_window      -- best shared [t0, t0+span] (wave-reaches-tail guard)
    _visible_connectors -- connector count with both endpoints inside a window
    _platoon_vref       -- platoon-mean speed for the oblique detrend
    fig_platoon_spacetime -- the platoon-propagation figure

This module only globs a folder, runs the per-file analysis at a fixed span,
ranks the files, writes a transparent per-file ranking CSV, and renders the
platoon figure for the winner.

Selection metric (D1-A: two-tier)
---------------------------------
  Tier 1 : files whose best common window has EVERY consecutive pair
           contributing >=1 matched connector (guard_pass) -- the perturbation
           is seen to reach the tail.
  Tier 2 : the rest (the common window fell back to pure max-sum).
  Within a tier, rank by the richness term (--score):
      'sum'  -- summed visible connectors [default; favours longer platoons]
      'mean' -- mean per-pair connectors  [length-fair across vehicle counts]
  tie-broken by the minimum per-pair connector count (propagation balance),
  then n_vehicles (longer platoon), then filename (determinism).

Span (D2-A) defaults to 60 s to match the single-file tool; tune with --span.
Winner outputs (D3-A): the platoon-propagation PNG + the ranking CSV only.

Only ASta_040719_platoon7.csv ships in the dev sandbox, so a single-file run
trivially selects itself (a full-chain smoke test). Point --folder / the
positional arg at the real AstaZero folder to rank every run in one pass.

Usage
-----
  python survey_astazero.py /path/to/AstaZero            # rank folder, render winner
  python survey_astazero.py /path/to/AstaZero --span 45  # tighter window
  python survey_astazero.py --score mean                 # length-fair richness
  python survey_astazero.py --smoke                      # first file found only
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Importing the analysis module also installs the NumPy-2.x np.trapz alias
# (its header does this before phase_segmentation is imported) and configures
# the Agg backend + fonts. It does NOT run main() (guarded by __main__).
import acc_controller_behavior as acb


# --------------------------------------------------------------------------- #
# Per-file result record
# --------------------------------------------------------------------------- #
@dataclass
class FileResult:
    path: str
    name: str
    ok: bool = False
    error: str = ""
    n_vehicles: int = 0
    order: List[str] = field(default_factory=list)
    duration: float = float("nan")
    dt: float = float("nan")
    n_samples: int = 0
    t0: float = float("nan")
    t1: float = float("nan")
    per_pair_counts: List[int] = field(default_factory=list)
    summed: int = 0
    mean_pair: float = float("nan")
    min_pair: int = 0
    guard_pass: bool = False
    v_lo: float = float("nan")
    v_hi: float = float("nan")
    score: float = float("nan")        # richness term actually used for ranking
    rank: Optional[int] = None
    tier: Optional[int] = None
    # heavy objects kept only to render the winner; not written to CSV
    analyses: object = None
    t: object = None


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
_PLATOON_RE = re.compile(r"platoon(\d+)", re.IGNORECASE)


def _natural_key(path: str):
    """Sort ASta_<date>_platoon<N>.csv so platoon2 precedes platoon10."""
    name = os.path.basename(path)
    m = _PLATOON_RE.search(name)
    n = int(m.group(1)) if m else 10**9
    # keep the date prefix as the primary key, platoon number as secondary
    stem = name[: m.start()] if m else name
    return (stem, n, name)


def resolve_files(target: Optional[str]) -> List[str]:
    """Resolve the positional target into a sorted list of AstaZero CSVs.
    Accepts a directory, a glob, a single file, or None (search fallbacks)."""
    pattern = "ASta_*_platoon*.csv"
    if target is None:
        for d in (".", "/mnt/project", "/mnt/user-data/uploads",
                  os.path.dirname(os.path.abspath(__file__))):
            hits = glob.glob(os.path.join(d, pattern))
            if hits:
                return sorted(hits, key=_natural_key)
        return []
    if os.path.isdir(target):
        hits = glob.glob(os.path.join(target, pattern))
        return sorted(hits, key=_natural_key)
    hits = glob.glob(target)              # glob or single existing file
    if not hits and os.path.exists(target):
        hits = [target]
    return sorted(hits, key=_natural_key)


# --------------------------------------------------------------------------- #
# Per-file processing (all heavy lifting delegated to acb)
# --------------------------------------------------------------------------- #
def default_analysis_args(overrides: Optional[dict] = None):
    """A parsed namespace of acb's own defaults, with optional overrides, so
    analyze_pair() sees exactly the knobs it expects (SI CUSUM etc.)."""
    ns = acb.build_arg_parser().parse_args([])
    for k, v in (overrides or {}).items():
        setattr(ns, k, v)
    return ns


def process_file(path: str, span: float, step: float,
                 analysis_args, score_mode: str) -> FileResult:
    name = os.path.basename(path)
    try:
        pairs, info = acb.load_openacc_asta(path)
        if not pairs:
            raise ValueError("no consecutive CF pairs (need >=2 vehicles)")
        analyses = [acb.analyze_pair(p, analysis_args) for p in pairs]
        t = pairs[0].t

        t0, t1 = acb._common_window(analyses, t, span, step)
        counts = [acb._visible_connectors(pa, t, t0, t1) for pa in analyses]
        summed = int(sum(counts))
        mean_pair = float(np.mean(counts)) if counts else float("nan")
        min_pair = int(min(counts)) if counts else 0
        guard = bool(counts) and all(c >= 1 for c in counts)

        m = (t >= t0) & (t <= t1)
        speeds = [analyses[0].pair.v_leader[m]] + [pa.pair.v_follower[m] for pa in analyses]
        allv = np.concatenate(speeds) if speeds else np.array([np.nan])
        v_lo, v_hi = float(np.nanmin(allv)), float(np.nanmax(allv))

        richness = float(summed if score_mode == "sum" else mean_pair)

        return FileResult(
            path=path, name=name, ok=True,
            n_vehicles=int(info["n_veh"]), order=list(info.get("order", [])),
            duration=float(t[-1] - t[0]), dt=float(info["dt"]),
            n_samples=int(info["n_samples"]),
            t0=float(t0), t1=float(t1),
            per_pair_counts=[int(c) for c in counts], summed=summed,
            mean_pair=mean_pair, min_pair=min_pair, guard_pass=guard,
            v_lo=v_lo, v_hi=v_hi, score=richness,
            analyses=analyses, t=t)
    except Exception as e:                                    # robust folder scan
        return FileResult(path=path, name=name, ok=False, error=repr(e))


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _sort_key(r: FileResult):
    # guard-pass first (0 before 1); then richness, min-pair, n_vehicles all
    # descending (negated); filename ascending for determinism.
    return (0 if r.guard_pass else 1, -r.score, -r.min_pair,
            -r.n_vehicles, r.name)


def rank_results(results: List[FileResult]) -> Tuple[List[FileResult], List[FileResult]]:
    ok = sorted((r for r in results if r.ok), key=_sort_key)
    failed = [r for r in results if not r.ok]
    for i, r in enumerate(ok, 1):
        r.rank = i
        r.tier = 1 if r.guard_pass else 2
    return ok, failed


def build_dataframe(ok: List[FileResult], failed: List[FileResult],
                    span: float, score_mode: str) -> pd.DataFrame:
    rows = []
    for r in ok:
        rows.append(dict(
            rank=r.rank, tier=r.tier, file=r.name, ok=True,
            n_vehicles=r.n_vehicles, order=" > ".join(r.order),
            duration_s=round(r.duration, 1), dt_s=round(r.dt, 3),
            n_samples=r.n_samples, span_s=span,
            t0_s=round(r.t0, 1), t1_s=round(r.t1, 1),
            per_pair_counts="/".join(map(str, r.per_pair_counts)),
            summed_connectors=r.summed, mean_pair=round(r.mean_pair, 2),
            min_pair=r.min_pair, guard_pass=r.guard_pass,
            score_mode=score_mode, score=round(r.score, 3),
            v_lo=round(r.v_lo, 2), v_hi=round(r.v_hi, 2), error=""))
    for r in failed:
        rows.append(dict(
            rank="", tier="", file=r.name, ok=False,
            n_vehicles="", order="", duration_s="", dt_s="", n_samples="",
            span_s=span, t0_s="", t1_s="", per_pair_counts="",
            summed_connectors="", mean_pair="", min_pair="", guard_pass="",
            score_mode=score_mode, score="", v_lo="", v_hi="", error=r.error))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_survey_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", default="../../Dataset/OpenACC/AstaZero",
                   help="AstaZero folder, a glob, or a single CSV. "
                        "Default: search cwd / project / uploads.")
    p.add_argument("--outdir", default="./astazero-outputs")
    # selection / window
    p.add_argument("--span", type=float, default=60.0,
                   help="Common-window width (s). D2 default 60.")
    p.add_argument("--step", type=float, default=5.0,
                   help="Window scan step (s).")
    p.add_argument("--score", choices=["sum", "mean"], default="sum",
                   help="Richness term for ranking. D1 default 'sum'.")
    # segmentation knobs (SI defaults from acb) -- forwarded to analyze_pair
    p.add_argument("--penalty", type=float, default=55.0)
    p.add_argument("--min-seg", type=int, default=20)
    p.add_argument("--cusum-thresh", type=float, default=acb.SI_CUSUM_THRESH)
    p.add_argument("--cusum-drift", type=float, default=acb.SI_CUSUM_DRIFT)
    p.add_argument("--sg-window", type=int, default=15)
    p.add_argument("--sg-poly", type=int, default=2)
    p.add_argument("--tau-max", type=float, default=5.0)
    p.add_argument("--deadband", type=float, default=0.2)
    p.add_argument("--eps-stable", type=float, default=0.15)
    p.add_argument("--tmin", type=float, default=3.0)
    # figure knobs (forwarded to fig_platoon_spacetime)
    p.add_argument("--font-scale", type=float, default=1.0)
    p.add_argument("--overlay-tags", action=argparse.BooleanOptionalAction,
                   default=True, help="Response-lag tags on the overlay.")
    p.add_argument("--overlay-tag-spacing", type=float, default=2.5)
    # dev
    p.add_argument("--smoke", action="store_true",
                   help="Process only the first file found.")
    return p


def main(argv=None) -> None:
    args = build_survey_parser().parse_args(argv)
    acb.set_font_scale(args.font_scale)
    os.makedirs(args.outdir, exist_ok=True)

    files = resolve_files(args.folder)
    if not files:
        sys.exit("error: no ASta_*_platoon*.csv found; pass a folder/glob/CSV.")
    if args.smoke:
        files = files[:1]

    analysis_args = default_analysis_args(dict(
        penalty=args.penalty, min_seg=args.min_seg,
        cusum_thresh=args.cusum_thresh, cusum_drift=args.cusum_drift,
        sg_window=args.sg_window, sg_poly=args.sg_poly, tau_max=args.tau_max,
        deadband=args.deadband, eps_stable=args.eps_stable, tmin=args.tmin))

    print(f"[survey] {len(files)} file(s), span={args.span:.0f}s, "
          f"score='{args.score}'")
    results = []
    for f in files:
        r = process_file(f, args.span, args.step, analysis_args, args.score)
        if r.ok:
            print(f"[ok]   {r.name:32s} veh={r.n_vehicles} "
                  f"win={r.t0:.0f}-{r.t1:.0f}s counts={r.per_pair_counts} "
                  f"sum={r.summed} guard={r.guard_pass}")
        else:
            print(f"[skip] {r.name:32s} {r.error}")
        results.append(r)

    ok, failed = rank_results(results)
    df = build_dataframe(ok, failed, args.span, args.score)
    csv_path = os.path.join(args.outdir, "astazero_survey_ranking.csv")
    df.to_csv(csv_path, index=False)

    print("\n[ranking]")
    show = [c for c in ("rank", "file", "n_vehicles", "t0_s", "t1_s",
                        "per_pair_counts", "summed_connectors", "min_pair",
                        "guard_pass", "score") if c in df.columns]
    print(df[df["ok"] == True][show].to_string(index=False)
          if (df["ok"] == True).any() else "  (no files ranked)")

    outputs = [csv_path]
    if ok:
        win = ok[0]
        stem = os.path.splitext(win.name)[0]
        v_ref = acb._platoon_vref(win.analyses, win.t, win.t0, win.t1)
        png = os.path.join(args.outdir, f"platoon_propagation_{stem}.png")
        acb.fig_platoon_spacetime(win.analyses, png, win.t0, win.t1, v_ref,
                                  tags=args.overlay_tags,
                                  tag_min_dt=args.overlay_tag_spacing)
        outputs.append(png)
        print(f"\n[winner] {win.name}  window {win.t0:.0f}-{win.t1:.0f}s  "
              f"v_ref={v_ref:.1f} m/s  (tier {win.tier}, score {win.score:.3f})")
        print(f"[winner] platoon order: {' > '.join(win.order)}")

    print(f"\n[done] wrote {len(outputs)} output(s):")
    for o in outputs:
        print(f"        {o}")


if __name__ == "__main__":
    main()
