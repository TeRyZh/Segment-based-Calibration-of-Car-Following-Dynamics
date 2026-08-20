#!/usr/bin/env python3
"""
simulate.py
===========
Step 2 of the phase-transition calibration framework: continuous forward
integration of the follower given the *exogenous* observed leader.

The follower is integrated across the ENTIRE car-following pair with a single
global parameter set and is NOT reset at phase boundaries -- this is exactly
what tests the true stability / transferability of the parameters (framework
Step 2, and the manuscript's stability claim). The observed leader trajectory
(position and speed) is fed in verbatim from the data at each step.

Integration scheme (decision D5): ballistic / Treiber update.

    a_n     = model.accel(v_n, dv_n, s_n, theta)      (then clipped, see below)
    v_{n+1} = max(0, v_n + a_n * dt)
    x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt^2             (ballistic position step)

with exact within-step stop handling when v would cross zero inside a step
(distance to stop = -0.5 v^2 / a). At dt = tau this reproduces standard Gipps
exactly for the Gipps model (see cf_models.Gipps), so the same integrator serves
all three models.

Physical envelope owned by the SIMULATOR, not the models
--------------------------------------------------------
The models intentionally leave acceleration unbounded (IDM's collision-avoidance
term blows up as s -> 0; OVM has no acceleration cap). The simulator therefore
owns:
  * an acceleration clip [a_min, a_max]  (keeps a badly-parameterised run finite
    instead of producing +/-1e2 m/s^2 or NaN), and
  * a no-overtake barrier at s_min_hard  (the follower cannot pass through the
    leader's rear bumper).
When either fires it is surfaced via SimResult.collided / n_barrier so the
objective can add a soft penalty rather than silently accept a crash.

Conventions (match cf_models and the extracted CF-pair CSVs)
------------------------------------------------------------
    dv = v_follower - v_leader   (positive = closing / approaching)
    s  = net bumper-to-bumper gap = x_leader - x_follower - leader_length (> 0)
    everything SI (convert NGSIM feet -> m upstream, decision D1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cf_models import CFModel

# Physical acceleration envelope owned by the simulator (SI, m/s^2).
A_MIN_DEFAULT = -9.0    # ~0.9 g emergency-braking floor
A_MAX_DEFAULT = 6.0     # generous accel cap (guards OVM's unbounded a)
S_MIN_HARD = 0.1        # no-overtake barrier: minimum net gap [m]


@dataclass
class SimResult:
    """Result of a continuous forward simulation over one pair (SI units)."""
    t: np.ndarray            # time grid (copied from the data)
    x: np.ndarray            # simulated follower position
    v: np.ndarray            # simulated follower speed (>= 0)
    a: np.ndarray            # applied (post-clip) acceleration
    s: np.ndarray            # simulated net spacing = x_leader - x - L
    collided: bool           # follower hit the no-overtake barrier >= once
    n_barrier: int           # number of steps clamped by the barrier

    def __len__(self) -> int:
        return len(self.t)


def simulate(model: CFModel,
             theta: Sequence[float],
             t: np.ndarray,
             x_leader: np.ndarray,
             v_leader: np.ndarray,
             leader_length: float,
             x0: float,
             v0: float,
             a_min: float = A_MIN_DEFAULT,
             a_max: float = A_MAX_DEFAULT,
             s_min_hard: float = S_MIN_HARD) -> SimResult:
    """Integrate the follower forward over the whole pair.

    Args:
        model         : a CFModel (idm / gipps / ovm).
        theta         : parameter vector aligned to ``model.param_names`` (SI).
        t             : time grid of the pair (s), length n, need not be uniform.
        x_leader      : leader position series (m), length n, increasing in travel.
        v_leader      : leader speed series (m/s), length n.
        leader_length : leader length L (m) for the net-gap definition.
        x0, v0        : initial follower position (m) and speed (m/s).
        a_min, a_max  : physical acceleration clip (m/s^2).
        s_min_hard    : no-overtake barrier / minimum net gap (m).

    Returns:
        SimResult with per-sample x, v, a, s and barrier bookkeeping.
    """
    t = np.asarray(t, dtype=float)
    x_leader = np.asarray(x_leader, dtype=float)
    v_leader = np.asarray(v_leader, dtype=float)
    n = len(t)

    L = float(leader_length)
    x = np.empty(n)
    v = np.empty(n)
    a = np.empty(n)
    s = np.empty(n)

    x[0] = float(x0)
    v[0] = max(0.0, float(v0))
    n_barrier = 0

    accel = model.accel                      # bind once (hot loop)
    th = np.asarray(theta, dtype=float)

    for k in range(n - 1):
        dt = t[k + 1] - t[k]

        s_k = x_leader[k] - x[k] - L         # current net gap
        s[k] = s_k
        dv_k = v[k] - v_leader[k]            # closing speed (positive = closing)

        a_raw = accel(v[k], dv_k, max(s_min_hard, s_k), th)
        a_k = a_raw if a_min <= a_raw <= a_max else (a_min if a_raw < a_min else a_max)
        a[k] = a_k

        # ballistic update with exact within-step stop handling
        v_next = v[k] + a_k * dt
        if v_next < 0.0:                      # follower stops inside [t_k, t_{k+1}]
            x_next = x[k] - 0.5 * v[k] * v[k] / a_k   # a_k < 0 here
            v_next = 0.0
        else:
            x_next = x[k] + v[k] * dt + 0.5 * a_k * dt * dt

        # no-overtake barrier against the leader's rear bumper at t_{k+1}
        x_barrier = x_leader[k + 1] - L - s_min_hard
        if x_next > x_barrier:
            x_next = x_barrier
            v_next = min(v_next, max(0.0, v_leader[k + 1]))
            n_barrier += 1

        x[k + 1] = x_next
        v[k + 1] = v_next

    # trailing-sample bookkeeping (no step is taken from the final index)
    s[n - 1] = x_leader[n - 1] - x[n - 1] - L
    a[n - 1] = a[n - 2] if n > 1 else 0.0

    return SimResult(t=t, x=x, v=v, a=a, s=s,
                     collided=n_barrier > 0, n_barrier=n_barrier)


# --------------------------------------------------------------------------- #
# Self-test: internal consistency + Gipps-at-dt=tau equivalence sanity
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    from cf_models import get_model

    print("=" * 70)
    print("simulate.py self-test")
    print("=" * 70)

    # A leader that cruises, brakes to a crawl, holds, then accelerates back.
    dt = 0.1
    t = np.arange(0.0, 60.0 + dt / 2, dt)
    n = len(t)
    vL = np.piecewise(
        t,
        [t < 15, (t >= 15) & (t < 25), (t >= 25) & (t < 35), t >= 35],
        [18.0,
         lambda tt: 18.0 - 1.4 * (tt - 15),      # decel to ~4 m/s
         4.0,
         lambda tt: np.minimum(18.0, 4.0 + 1.2 * (tt - 35))],
    )
    vL = np.clip(vL, 0.0, None)
    xL = np.concatenate([[200.0], 200.0 + np.cumsum(0.5 * (vL[1:] + vL[:-1]) * dt)])

    idm = get_model("idm")
    theta = idm.default_params()
    res = simulate(idm, theta, t, xL, vL, leader_length=5.0,
                   x0=xL[0] - 25.0, v0=18.0)

    print(f"\n[1] shapes & finiteness (n={n})")
    for name, arr in (("x", res.x), ("v", res.v), ("a", res.a), ("s", res.s)):
        assert arr.shape == (n,), f"{name} wrong shape"
        assert np.all(np.isfinite(arr)), f"{name} has non-finite values"
    print("    all arrays length n and finite -> PASS")

    print("\n[2] physical invariants")
    assert np.all(res.v >= -1e-9), "speed went negative"
    assert np.all(res.s > 0.0), "net gap went non-positive (barrier failed)"
    assert res.a.min() >= A_MIN_DEFAULT - 1e-9, "accel below floor"
    assert res.a.max() <= A_MAX_DEFAULT + 1e-9, "accel above cap"
    print(f"    v>=0, s>0, a in [{A_MIN_DEFAULT},{A_MAX_DEFAULT}]  "
          f"(gap min={res.s.min():.2f} m, v range {res.v.min():.2f}"
          f"..{res.v.max():.2f} m/s) -> PASS")

    print("\n[3] follower tracks the leader's slow-down (qualitative)")
    # around the leader's crawl (t in [26,34]) the follower should be slow too
    mask = (t >= 26) & (t <= 34)
    assert res.v[mask].mean() < 8.0, "follower did not slow behind a crawling leader"
    print(f"    mean follower speed during leader crawl = "
          f"{res.v[mask].mean():.2f} m/s (< 8) -> PASS")

    print("\n[4] Gipps at dt = tau reproduces the model's own speed update")
    gip = get_model("gipps")
    p = gip.default_params()                      # a,V,tau,b,b_hat,s0
    tau = float(p[2])
    tg = np.arange(0.0, 20.0 + tau / 2, tau)      # step exactly at tau
    vLg = np.full_like(tg, 12.0)
    xLg = np.concatenate([[100.0],
                          100.0 + np.cumsum(0.5 * (vLg[1:] + vLg[:-1]) * tau)])
    rg = simulate(gip, p, tg, xLg, vLg, leader_length=5.0,
                  x0=xLg[0] - 30.0, v0=8.0, a_min=-50, a_max=50)
    # reconstruct one manual Gipps step and compare to the integrator
    v0g, s0g = rg.v[0], rg.s[0]
    a_eq = gip.accel(v0g, v0g - vLg[0], s0g, p)
    v1_manual = max(0.0, v0g + a_eq * tau)
    assert abs(v1_manual - rg.v[1]) < 1e-6, "Gipps dt=tau speed update mismatch"
    print(f"    manual v1={v1_manual:.4f}  integrator v1={rg.v[1]:.4f} -> PASS")

    print("\nAll simulator checks passed.")


if __name__ == "__main__":
    _selftest()
