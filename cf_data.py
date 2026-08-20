#!/usr/bin/env python3
"""
cf_data.py
==========
Loading and unit handling for extracted car-following (CF) pairs.

A "pair" is one CSV produced by ``extract_cf_pairs.py`` with columns

    t, x_follower, v_follower, a_follower,
    x_leader, v_leader, a_leader, leader_length,
    spacing, dv

where ``spacing`` = net gap s = x_leader - x_follower - leader_length and
``dv`` = v_follower - v_leader (positive = closing). This module wraps such a
CSV in a small :class:`PairData` record, converts it to SI (decision D1), and
provides folder / manifest loading for train/test sets (decision D10).

Unit auto-detection (decision D1)
---------------------------------
NGSIM pairs are imperial (feet, ft/s) unless the extractor was run with
``--to-si``; TGSIM pairs are already metric. Detection uses magnitude
heuristics on speed, spacing and leader length. These are reliable for typical
freeway pairs but *can* misclassify a very slow, tight, short-vehicle imperial
pair as metric. Pass ``units='feet'`` or ``units='si'`` to be certain; for raw
NGSIM, ``units='feet'`` is always safe.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FT_TO_M = 0.3048

# Columns the calibrator consumes; ``a_*`` are optional (diagnostics only).
REQUIRED_COLS = ("t", "x_follower", "v_follower", "x_leader", "v_leader",
                 "leader_length", "spacing", "dv")


# --------------------------------------------------------------------------- #
# Unit detection
# --------------------------------------------------------------------------- #
def detect_units(df: pd.DataFrame) -> str:
    """Best-effort 'feet' | 'si' guess from column magnitudes (decision D1).

    Any one of these firing votes 'feet':
        max |v_follower|  > 45     (m/s is implausible for a road follower)
        max  spacing      > 110    (extractor caps metric gaps near 100 m)
        max  leader_length > 25    (a 25 m road vehicle is a rare outlier)
    """
    v_max = float(np.nanmax(np.abs(df["v_follower"].to_numpy())))
    s_max = float(np.nanmax(df["spacing"].to_numpy()))
    L_max = float(np.nanmax(df["leader_length"].to_numpy())) \
        if "leader_length" in df.columns else 0.0
    if v_max > 45.0 or s_max > 110.0 or L_max > 25.0:
        return "feet"
    return "si"


@dataclass
class PairData:
    """One CF pair in SI units, with the empirical leader as exogenous input."""
    name: str
    t: np.ndarray
    x_follower: np.ndarray
    v_follower: np.ndarray
    a_follower: np.ndarray
    x_leader: np.ndarray
    v_leader: np.ndarray
    leader_length: float
    spacing: np.ndarray            # net gap s (m)
    dv: np.ndarray                 # v_follower - v_leader (m/s), + = closing
    units_source: str              # 'feet -> m' | 'm (native)'
    dt: float                      # median sampling step (s)
    split: Optional[str] = None    # 'train' | 'test' | None (from a manifest)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def n(self) -> int:
        return len(self.t)


def load_pair(path: str, units: str = "auto",
              split: Optional[str] = None) -> PairData:
    """Load one pair CSV and return it converted to SI.

    Args:
        path  : path to a per-pair CSV.
        units : 'auto' (default), 'si', or 'feet'. 'auto' calls :func:`detect_units`.
        split : optional split tag to attach ('train' | 'test').
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing columns {missing}; "
                       f"found {list(df.columns)}")

    u = str(units).strip().lower()
    if u == "auto":
        u = detect_units(df)
    if u not in ("si", "feet"):
        raise ValueError(f"units must be auto|si|feet, got {units!r}")

    k = FT_TO_M if u == "feet" else 1.0
    unit_src = "feet -> m" if u == "feet" else "m (native)"

    t = df["t"].to_numpy(dtype=float)
    dts = np.diff(t)
    dt = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 0.1

    a_foll = (df["a_follower"].to_numpy(dtype=float) * k
              if "a_follower" in df.columns
              else np.full(len(t), np.nan))

    # leader_length is a per-pair constant; take the first finite value
    Lcol = df["leader_length"].to_numpy(dtype=float)
    Lval = float(np.nanmedian(Lcol)) * k

    return PairData(
        name=os.path.splitext(os.path.basename(path))[0],
        t=t,
        x_follower=df["x_follower"].to_numpy(dtype=float) * k,
        v_follower=df["v_follower"].to_numpy(dtype=float) * k,
        a_follower=a_foll,
        x_leader=df["x_leader"].to_numpy(dtype=float) * k,
        v_leader=df["v_leader"].to_numpy(dtype=float) * k,
        leader_length=Lval,
        spacing=df["spacing"].to_numpy(dtype=float) * k,
        dv=df["dv"].to_numpy(dtype=float) * k,
        units_source=unit_src,
        dt=dt,
        split=split,
    )


def load_folder(root: str, split: Optional[str] = None,
                units: str = "auto", limit: Optional[int] = None) -> List[PairData]:
    """Load pairs from an ``extract_cf_pairs.py`` output folder.

    If ``root/manifest.csv`` exists it is used to resolve paths and the split
    column; otherwise every CSV under ``root`` (or ``root/<split>``) is loaded.
    """
    manifest = os.path.join(root, "manifest.csv")
    pairs: List[PairData] = []

    if os.path.exists(manifest):
        man = pd.read_csv(manifest)
        path_col = next((c for c in ("path", "file", "csv", "filepath")
                         if c in man.columns), None)
        split_col = next((c for c in ("split", "set") if c in man.columns), None)
        for _, row in man.iterrows():
            if split is not None and split_col is not None \
                    and str(row[split_col]).lower() != split.lower():
                continue
            rel = str(row[path_col]) if path_col else None
            p = rel if (rel and os.path.isabs(rel)) else os.path.join(root, rel) \
                if rel else None
            if p is None or not os.path.exists(p):
                continue
            tag = str(row[split_col]) if split_col else split
            pairs.append(load_pair(p, units=units, split=tag))
            if limit and len(pairs) >= limit:
                break
        return pairs

    # no manifest: glob CSVs, optionally under a split subfolder
    search_root = os.path.join(root, split) if split and \
        os.path.isdir(os.path.join(root, split)) else root
    for p in sorted(glob.glob(os.path.join(search_root, "**", "*.csv"),
                              recursive=True)):
        if os.path.basename(p).lower() == "manifest.csv":
            continue
        pairs.append(load_pair(p, units=units, split=split))
        if limit and len(pairs) >= limit:
            break
    return pairs


def discover_pair(explicit: Optional[str] = None) -> Optional[str]:
    """Find a single pair CSV: an explicit path, else uploads, else CWD."""
    if explicit and os.path.exists(explicit):
        return explicit
    for pat in ("/mnt/user-data/uploads/*.csv",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.csv"),
                "*.csv"):
        hits = [h for h in glob.glob(pat)
                if os.path.basename(h).lower() != "manifest.csv"]
        if hits:
            return sorted(hits)[0]
    return None


if __name__ == "__main__":
    p = discover_pair()
    if p is None:
        print("no CF-pair CSV found (uploads/CWD empty)")
        raise SystemExit
    pair = load_pair(p)
    print(f"file        : {pair.name}")
    print(f"units       : {pair.units_source}")
    print(f"samples     : {pair.n}   dt={pair.dt:.3f}s   "
          f"duration={(pair.t[-1] - pair.t[0]):.1f}s")
    print(f"leader L    : {pair.leader_length:.2f} m")
    print(f"v_follower  : {pair.v_follower.min():.2f}..{pair.v_follower.max():.2f} m/s")
    print(f"spacing     : {pair.spacing.min():.2f}..{pair.spacing.max():.2f} m")
