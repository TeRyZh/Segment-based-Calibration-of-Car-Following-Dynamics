#!/usr/bin/env python3
"""
cf_models.py
============
Car-following (CF) acceleration models for phase-transition calibration.
 
Five models are provided behind a single interface so the simulator, the
objective functions, and the optimizer can treat them interchangeably:
 
    IDM   - Intelligent Driver Model     (Treiber, Hennecke & Helbing, 2000)
    Gipps - Gipps safe-distance model    (Gipps, 1981)
    OVM   - Optimal Velocity Model       (Bando et al., 1995)
    FVDM  - Full Velocity Difference Model (Jiang, Wu & Zhu, 2001)  (OVM + dv term)
    d-IDM - Delayed IDM = IDM + explicit response delay tau  (this file)
 
Common interface
----------------
Every model exposes
 
    accel(v, dv, s, theta) -> a
 
returning the follower's longitudinal acceleration given
    v      follower speed               (>= 0)
    dv     v_follower - v_leader        (POSITIVE = closing / approaching)
    s      net bumper-to-bumper gap     (> 0)
    theta  1-D parameter array aligned to `param_names`
 
`dv` and `s` follow the sign / definition of the extracted CF-pair CSVs (their
`dv` and `spacing` columns), so model inputs map straight onto the data with no
re-derivation. `accel` is scalar-per-call (a CF simulation steps sequentially).
 
Units
-----
The models are unit-agnostic: pass a self-consistent unit system and the output
is in those units. All DEFAULT parameter values and bounds below are SI
(metres, seconds, m/s, m/s^2). NGSIM extraction CSVs are in feet; convert to SI
upstream (x 0.3048) before calibrating so these bounds apply.
 
Sign conventions
----------------
Deceleration parameters (IDM `b`, Gipps `b`, `b_hat`) are stored as POSITIVE
magnitudes; the negative sign is applied internally. This keeps every bound
positive, which is friendlier for box-constrained global optimizers
(differential evolution / PSO).
 
Notes on the Gipps interface
----------------------------
Gipps is natively a speed-update model evaluated over a reaction time `tau`. It
is exposed here as an *equivalent acceleration* a_eq = (v_next(tau) - v) / tau.
With a ballistic integrator at dt = tau this reproduces standard Gipps exactly
(both the speed update and the trapezoidal position update); dt < tau gives a
consistent sub-stepping. See Gipps.accel for the algebra.

Notes on the Delayed IDM (d-IDM)
--------------------------------
d-IDM (class ``DelayedIDM``) appends a single response delay ``tau`` to IDM's
five parameters. Because ``accel`` is history-free by the interface contract
above, the delay is applied by a *delay-aware simulator* -- it feeds ``accel``
the stimulus (v, dv, s) sampled at t - tau -- NOT by ``accel`` itself. Running
d-IDM through the ordinary memoryless simulator therefore reproduces standard
IDM with tau ignored (tau = 0). See the DelayedIDM docstring.
"""
 
from __future__ import annotations
 
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Tuple
 
import numpy as np
 
Number = float
_EPS = 1e-9
 
 
# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class CFModel(ABC):
    """Abstract car-following model with a positional parameter vector."""
 
    name: str = "base"
    param_names: List[str] = []
    param_bounds: List[Tuple[float, float]] = []   # SI, aligned to param_names
    fixed_params: Dict[str, float] = {}            # not calibrated (e.g. IDM delta)
    _defaults: List[float] = []                    # SI literature-typical, aligned
 
    # ---- introspection helpers ----
    @property
    def n_params(self) -> int:
        return len(self.param_names)
 
    def default_params(self) -> np.ndarray:
        """Literature-typical SI parameter vector (starting point / smoke tests)."""
        return np.asarray(self._defaults, dtype=float)
 
    def bounds(self) -> List[Tuple[float, float]]:
        return list(self.param_bounds)
 
    def param_dict(self, theta: Sequence[float]) -> Dict[str, float]:
        """Map a parameter vector to a named dict, including any fixed params."""
        d = {n: float(v) for n, v in zip(self.param_names, theta)}
        d.update(self.fixed_params)
        return d
 
    def within_bounds(self, theta: Sequence[float]) -> bool:
        return all(lo <= v <= hi for v, (lo, hi) in zip(theta, self.param_bounds))
 
    def clip(self, theta: Sequence[float]) -> np.ndarray:
        lo = np.array([b[0] for b in self.param_bounds])
        hi = np.array([b[1] for b in self.param_bounds])
        return np.clip(np.asarray(theta, float), lo, hi)
 
    # ---- the model ----
    @abstractmethod
    def accel(self, v: Number, dv: Number, s: Number,
              theta: Sequence[float]) -> Number:
        """Follower acceleration from (v, dv=v-v_lead, s) and parameters theta."""
        ...
 
    def __repr__(self) -> str:
        return f"<CFModel {self.name}: {self.param_names}>"
 
 
# --------------------------------------------------------------------------- #
# IDM - Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000)
# --------------------------------------------------------------------------- #
class IDM(CFModel):
    r"""Intelligent Driver Model.
 
        a = a_max * [ 1 - (v/v0)^delta - (s*(v, dv) / s)^2 ]
        s*(v, dv) = s0 + max(0, v*T + v*dv / (2*sqrt(a_max*b)))
 
    with dv = v_follower - v_leader (closing positive). `delta` is fixed (= 4).
    Braking is intentionally unbounded as s -> 0 (collision avoidance); the
    simulator is responsible for any physical acceleration floor.
    """
 
    name = "idm"
    param_names = ["v0", "T", "a_max", "b", "s0"]

    # Adjusted for standard highway/arterial traffic
    param_bounds = [(15.0, 35.0),  # v0    desired speed          [m/s] (approx 54 - 126 km/h)
                    (1.2, 3.0),    # T     safe time headway      [s]
                    (1.0, 2.5),    # a_max max acceleration       [m/s^2]
                    (1.5, 3.0),    # b     comfortable decel      [m/s^2] (magnitude)
                    (1.5, 5.0)]    # s0    minimum jam gap        [m]
                    
    # Defaults based on typical Treiber/Kesting recommendations
    _defaults    = [30.0, 1.5, 1.4, 2.0, 2.0]
    
    # Delta of 4.0 is standard and correct
    fixed_params = {"delta": 4.0}
 
    def accel(self, v, dv, s, theta):
        v0, T, a_max, b, s0 = theta
        delta = self.fixed_params["delta"]
 
        v = max(0.0, float(v))
        s = max(_EPS, float(s))                     # guard zero / negative gap

        s_dyn = v * T + (v * float(dv)) / (2.0 * np.sqrt(a_max * b))
        s_star = s0 + max(0.0, s_dyn)               # canonical floor at s0
 
        free = 1.0 - (v / v0) ** delta
        interaction = (s_star / s) ** 2
        return float(a_max * (free - interaction))
 
 
# --------------------------------------------------------------------------- #
# d-IDM - Delayed Intelligent Driver Model (IDM + explicit response delay tau)
# --------------------------------------------------------------------------- #
class DelayedIDM(IDM):
    r"""Delayed Intelligent Driver Model (d-IDM).

    Standard IDM augmented with a single explicit response delay ``tau``: the
    follower reacts to the stimulus it perceived ``tau`` seconds ago, not the
    instantaneous one. Writing the delayed stimulus as (v, dv, s) evaluated at
    t - tau,

        a_f(t) = a_max * [ 1 - (v(t-tau)/v0)^delta
                             - ( s*(v(t-tau), dv(t-tau)) / s(t-tau) )^2 ]
        s*(v, dv) = s0 + max(0, v*T + v*dv / (2*sqrt(a_max*b)))

    Parameter vector -- tau appended to the five IDM parameters (index 5):

        theta = [v0, T, a_max, b, s0, tau]

    ``delta`` stays fixed at 4 (inherited). The bounds are deliberately wider
    than the human-highway IDM parent to suit commercial ACC at the minimum-gap
    setting (shorter headways, harder braking, plus the tau axis) so calibration
    is free to expose the accel/decel asymmetry this model exists to measure --
    clipping those bounds would hide the effect.

    Where the delay lives (READ THIS)
    ---------------------------------
    By the project-wide contract, ``accel(v, dv, s, theta)`` is an *instantaneous*
    scalar map with no access to trajectory history, so it CANNOT apply the delay
    itself. It uses only the IDM parameters (theta[:5]) and returns the ordinary
    IDM acceleration for whatever (v, dv, s) it is handed; ``tau`` (theta[5]) is
    consumed *upstream, by a delay-aware simulator* that feeds ``accel`` the
    stimulus from t - tau. Consequences:

      * Running d-IDM through the ordinary memoryless ``simulate.simulate``
        reproduces standard IDM with tau IGNORED (equivalently tau = 0). To
        exercise tau you must use the delayed integrator, which reads
        ``tau_frames(theta, dt)`` to index t - tau.
      * Calibrating d-IDM with any objective built on the memoryless simulator
        leaves tau unidentifiable (a flat direction). Always calibrate d-IDM
        through the delayed simulator.

    The accessors below (``tau``, ``tau_frames``, ``idm_theta``) are the interface
    a delay-aware simulator uses to split the vector and discretise the delay to
    whole samples (nearest-frame; d-IDM design decision D1).
    """

    name = "didm"
    param_names = ["v0", "T", "a_max", "b", "s0", "tau"]

    # ACC minimum-gap envelope: intentionally wider than the human-highway IDM
    # parent (shorter T, harder b, plus the tau axis). See the class docstring.
    param_bounds = [(10.0, 40.0),  # v0    desired speed          [m/s]
                    (0.3, 3.0),    # T     desired time headway   [s]
                    (0.2, 3.0),    # a_max max acceleration       [m/s^2]
                    (0.2, 5.0),    # b     comfortable decel       [m/s^2] (magnitude)
                    (0.5, 8.0),    # s0    minimum standstill gap  [m]
                    (0.0, 3.0)]    # tau   response delay          [s]

    _defaults = [30.0, 1.2, 1.5, 2.0, 2.0, 1.0]

    # fixed_params (delta = 4.0) inherited from IDM.

    # ---- delay accessors (used by a delay-aware simulator) ----
    @property
    def tau_index(self) -> int:
        """Position of tau in the parameter vector."""
        return self.param_names.index("tau")

    def tau(self, theta: Sequence[float]) -> float:
        """Response delay tau (seconds), read from the parameter vector."""
        return float(theta[self.tau_index])

    def tau_frames(self, theta: Sequence[float], dt: float) -> int:
        """Delay as a whole number of samples at step ``dt`` (nearest-frame).

        Design decision D1: the continuous tau is discretised by rounding to the
        nearest sample so a delay-aware simulator can index t - tau_frames. ``dt``
        is the pair's sampling step (0.1 s at 10 Hz for OpenACC / NGSIM).
        """
        return max(0, int(round(self.tau(theta) / float(dt))))

    def idm_theta(self, theta: Sequence[float]) -> np.ndarray:
        """The five IDM parameters (tau dropped), as a float array."""
        return np.asarray(theta[:5], dtype=float)

    # ---- the model (instantaneous IDM on theta[:5]; see class docstring) ----
    def accel(self, v, dv, s, theta):
        # Delegate to the IDM map using only the five IDM parameters. The delay
        # is NOT applied here (accel has no history); a delay-aware simulator
        # applies it by choosing which (v, dv, s) to pass. Accepts a 5- or
        # 6-element theta.
        return super().accel(v, dv, s, theta[:5])


# --------------------------------------------------------------------------- #
# Gipps - safe-distance model (Gipps, 1981)
# --------------------------------------------------------------------------- #
class Gipps(CFModel):
    r"""Gipps safe-distance model, exposed as an equivalent acceleration.
 
    Free-flow (desired-speed) branch:
        v_acc  = v + 2.5*a*tau*(1 - v/V)*sqrt(0.025 + v/V)
 
    Safe (following) branch, derived so the follower can stop behind the leader.
    Gipps' own decel constants are negative; here b, b_hat are positive
    magnitudes, so the standard formula
        v_safe = b_n*tau + sqrt( b_n^2 tau^2 - b_n[ 2(s - s0) - v*tau - v_lead^2/b_hat_n ] )
    with b_n = -b, b_hat_n = -b_hat becomes
        disc   = b^2 tau^2 + b*( 2(s - s0) - v*tau + v_lead^2 / b_hat )
        v_safe = -b*tau + sqrt(disc)          (disc < 0 -> emergency stop)
 
    Reported acceleration:  a_eq = (max(0, min(v_acc, v_safe)) - v) / tau
    where v_lead = v - dv. At dt = tau a ballistic integrator reproduces standard
    Gipps exactly (velocity, and the 0.5*(v + v_next)*tau position update).
    """
 
    name = "gipps"
    param_names = ["a", "V", "tau", "b", "b_hat", "s0"]
    
    # Adjusted for realistic human physical limits and standard traffic
    param_bounds = [(1, 2.5),    # a      max acceleration           [m/s^2]
                    (15.0, 35.0),  # V      desired speed              [m/s]
                    (0.5, 2.0),    # tau    reaction time              [s]
                    (1.5, 4.0),    # b      max decel (magnitude)      [m/s^2]
                    (2.0, 4.0),    # b_hat  estimated leader decel     [m/s^2]
                    (1.5, 4.0)]    # s0     effective jam gap          [m]
                    
    # Defaults based on original Gipps (1981) and subsequent empirical studies
    _defaults    = [1.7, 30.0, 0.67, 3.0, 3.0, 2.0] 
 
 
    def accel(self, v, dv, s, theta):
        a, V, tau, b, b_hat, s0 = theta
 
        v = max(0.0, float(v))
        v_lead = max(0.0, v - float(dv))
        s = max(_EPS, float(s))
 
        # free-flow branch (valid formula; v > V gives a gentle relaxation down)
        ratio = v / V
        v_acc = v + 2.5 * a * tau * (1.0 - ratio) * np.sqrt(0.025 + ratio)
 
        # safe following branch (b, b_hat as positive magnitudes)
        disc = b * b * tau * tau + b * (2.0 * (s - s0) - v * tau
                                        + (v_lead * v_lead) / b_hat)
        v_safe = (-b * tau + np.sqrt(disc)) if disc >= 0.0 else 0.0
 
        v_next = max(0.0, min(v_acc, v_safe))
        return float((v_next - v) / tau)
 
 
# --------------------------------------------------------------------------- #
# OVM - Optimal Velocity Model (Bando et al., 1995)
# --------------------------------------------------------------------------- #
class OVM(CFModel):
    r"""Optimal Velocity Model.
 
        a = kappa * (V_opt(s) - v)
        V_opt(s) = v_max * [tanh((s - s_c)/w) + tanh(s_c/w)] / (1 + tanh(s_c/w))
 
    A normalized optimal-velocity function with V_opt(0) = 0 and
    V_opt(inf) = v_max, monotonically increasing in the gap s. Pure OVM depends
    only on (s, v): it has NO relative-speed (dv) term by construction, so `dv`
    is accepted for interface uniformity but unused. Adding a c*dv term would
    recover the Full Velocity Difference Model (Jiang et al., 2001) if wanted.
    """
 
    name = "ovm"
    param_names = ["kappa", "v_max", "s_c", "w"]
    
    # Adjusted to prevent extreme, physically impossible accelerations
    param_bounds = [(0.4, 2.0),    # kappa  sensitivity                [1/s] (0.4 to 2.0 sec relaxation)
                    (15.0, 35.0),  # v_max  desired / free speed       [m/s]
                    (15.0, 30.0),  # s_c    inflection gap             [m]
                    (5.0, 20.0)]   # w      transition width           [m]
                    
    # Defaults based on standard Bando OVM literature and stable traffic flow
    _defaults    = [1.5, 30.0, 20.0, 10.0]

    @staticmethod
    def _v_opt(v_max, s_c, w, s):
        """Normalized optimal-velocity V_opt(s): 0 at s=0, -> v_max as s -> inf."""
        s = max(0.0, float(s))
        t_sc = np.tanh(s_c / w)
        return v_max * (np.tanh((s - s_c) / w) + t_sc) / (1.0 + t_sc)

    def accel(self, v, dv, s, theta, a_max=2.5, d_max=-4.0):
        # dv unused (pure OVM)
        # theta unpacks: sensitivity, desired speed, inflection gap, transition width
        kappa, v_max, s_c, w = theta
 
        # Prevent negative speeds and gaps (physics check)
        v = max(0.0, float(v))
        s = max(0.0, float(s))
 
        # Calculate the optimal velocity based on the current gap
        v_opt = self._v_opt(v_max, s_c, w, s)
        
        # Calculate the raw, unbounded OVM acceleration
        raw_accel = kappa * (v_opt - v)
        
        # Apply physical kinematic constraints (clipping)
        # Prevents rocket-launch accelerations and physically impossible braking
        capped_accel = max(d_max, min(a_max, raw_accel))
        
        return float(capped_accel)
 
# --------------------------------------------------------------------------- #
# FVDM - Full Velocity Difference Model (Jiang, Wu & Zhu, 2001) -- OVM variant
# --------------------------------------------------------------------------- #
class FVDM(OVM):
    r"""Full Velocity Difference Model (Jiang, Wu & Zhu, 2001).

    OVM augmented with a relative-velocity (velocity-difference) term. Pure OVM
    responds only to the gap size and is blind to how fast that gap is changing,
    which lets it produce unrealistically large accelerations during a strong
    approach; the extra term fixes this:

        a = kappa * (V_opt(s) - v) + lambda * (v_leader - v_follower)
          = kappa * (V_opt(s) - v) - lambda * dv        (dv = v_follower - v_leader)

    with the same normalized optimal-velocity function V_opt(s) as OVM (inherited
    via ``_v_opt``). The sensitivity ``lambda`` (>= 0, units 1/s) makes the
    follower brake when closing (dv > 0) and accelerate when the leader pulls away
    (dv < 0). ``lambda = 0`` recovers pure OVM exactly.

    Parameter vector -- lambda appended to OVM's four parameters:

        theta = [kappa, v_max, s_c, w, lambda]
    """

    name = "fvdm"
    param_names = ["kappa", "v_max", "s_c", "w", "lambda"]

    param_bounds = [(0.4, 2.0),    # kappa   OV sensitivity              [1/s]
                    (15.0, 35.0),  # v_max   desired / free speed        [m/s]
                    (15.0, 30.0),  # s_c     inflection gap              [m]
                    (5.0, 20.0),   # w       transition width            [m]
                    (0.0, 3.0)]    # lambda  relative-speed sensitivity  [1/s]

    _defaults = [1.5, 30.0, 20.0, 10.0, 0.5]

    def accel(self, v, dv, s, theta, a_max=2.5, d_max=-4.0):
        # theta = [kappa, v_max, s_c, w, lambda]
        kappa, v_max, s_c, w, lam = theta
        v = max(0.0, float(v))
        v_opt = self._v_opt(v_max, s_c, w, s)             # shared OVM optimal velocity
        # OV relaxation + relative-velocity term. dv = v_follower - v_leader
        # (closing positive), so (v_leader - v_follower) = -dv: closing -> brake.
        raw_accel = kappa * (v_opt - v) - lam * float(dv)
        # physical kinematic clip (as in OVM: prevents impossible accel/braking)
        return float(max(d_max, min(a_max, raw_accel)))


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
MODELS: Dict[str, type] = {"idm": IDM, "gipps": Gipps, "ovm": OVM,
                           "didm": DelayedIDM, "fvdm": FVDM}
 
 
def get_model(name: str) -> CFModel:
    """Instantiate a model by name ('idm' | 'gipps' | 'ovm' | 'didm' | 'fvdm')."""
    key = str(name).strip().lower()
    if key not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[key]()
 
 
def available_models() -> List[str]:
    return list(MODELS)
 
 
# --------------------------------------------------------------------------- #
# Self-test: bounds, limiting behaviour, and a real-data plausibility smoke test
# --------------------------------------------------------------------------- #
def _load_si_pair():
    """Best-effort load of an extracted CF-pair CSV, converted to SI (feet->m).
 
    Returns (v, dv, s, a_obs) in SI, or None if no CSV / pandas is available.
    """
    import glob
    import os
 
    paths = (glob.glob("/mnt/user-data/uploads/*.csv")
             + glob.glob(os.path.join(os.path.dirname(__file__), "*.csv"))
             + glob.glob("*.csv"))
    if not paths:
        return None
    try:
        import pandas as pd
        df = pd.read_csv(paths[0])
    except Exception:
        return None
    need = {"v_follower", "dv", "spacing", "a_follower"}
    if not need.issubset(df.columns):
        return None
 
    FT = 0.3048  # NGSIM pairs are imperial; assume feet unless clearly metric
    metric = df["spacing"].max() < 60.0            # crude unit sniff
    k = 1.0 if metric else FT
    return (df["v_follower"].to_numpy() * k,
            df["dv"].to_numpy() * k,
            df["spacing"].to_numpy() * k,
            df["a_follower"].to_numpy() * k,
            os.path.basename(paths[0]),
            "m (native)" if metric else "ft -> m")
 
 
def _selftest() -> None:
    print("=" * 70)
    print("cf_models.py self-test")
    print("=" * 70)
 
    # 1) defaults lie within bounds
    print("\n[1] default parameters within bounds")
    for name in available_models():
        m = get_model(name)
        ok = m.within_bounds(m.default_params())
        print(f"    {name:6s} {dict(zip(m.param_names, m.default_params()))}  "
              f"-> {'PASS' if ok else 'FAIL'}")
        assert ok, f"{name} defaults out of bounds"
 
    # 2) qualitative limiting behaviour
    print("\n[2] limiting-behaviour checks")
 
    idm = get_model("idm"); p = idm.default_params()          # v0,T,a_max,b,s0
    a_rest = idm.accel(0.0, 0.0, 1000.0, p)                    # from rest, open road
    a_cruise = idm.accel(p[0], 0.0, 1000.0, p)                # at v0, open road
    a_close = idm.accel(15.0, 8.0, 3.0, p)                    # fast approach, tiny gap
    assert a_rest > 0.5,  "IDM should accelerate from rest on an open road"
    assert abs(a_cruise) < 0.05, "IDM should ~coast at v0 with no leader ahead"
    assert a_close < -1.0, "IDM should brake hard when closing into a small gap"
    print(f"    idm   a(rest,open)={a_rest:+.3f}  a(v0,open)={a_cruise:+.3f}  "
          f"a(close)={a_close:+.3f}  -> PASS")
 
    gip = get_model("gipps"); p = gip.default_params()        # a,V,tau,b,b_hat,s0
    g_rest = gip.accel(0.0, 0.0, 1000.0, p)
    g_close = gip.accel(20.0, 12.0, 3.0, p)
    assert g_rest > 0.1,  "Gipps should accelerate from rest on an open road"
    assert g_close < 0.0, "Gipps should decelerate when closing into a small gap"
    print(f"    gipps a(rest,open)={g_rest:+.3f}  a(close)={g_close:+.3f}  -> PASS")
 
    ovm = get_model("ovm"); p = ovm.default_params()          # kappa,v_max,s_c,w
    o_zero = ovm.accel(10.0, 0.0, 0.0, p)                     # zero gap -> V_opt=0
    o_open = ovm.accel(0.0, 0.0, 1000.0, p)                   # huge gap from rest
    assert o_zero < 0.0, "OVM V_opt(0)=0 -> must decelerate at zero gap"
    _ov_open = min(2.5, p[0] * p[1])              # OVM accel clips at a_max = 2.5
    assert abs(o_open - _ov_open) < 1e-3, "OVM open-road accel = clip(kappa*v_max)"
    print(f"    ovm   a(s=0,v=10)={o_zero:+.3f}  a(open,rest)={o_open:+.3f}  -> PASS")
 
    # 2a) Full Velocity Difference Model: lambda=0 recovers OVM; closing brakes more
    print("\n[2a] full velocity difference model (FVDM) checks")
    fvdm = get_model("fvdm"); pf = fvdm.default_params()      # kappa,v_max,s_c,w,lambda
    ovm_m = get_model("ovm")
    pf_l0 = pf.copy(); pf_l0[-1] = 0.0                        # lambda -> 0
    for (v, dv, s) in [(10.0, 0.0, 25.0), (12.0, 6.0, 15.0), (5.0, -4.0, 40.0)]:
        assert abs(fvdm.accel(v, dv, s, pf_l0) - ovm_m.accel(v, dv, s, pf_l0[:4])) < 1e-9, \
            "FVDM(lambda=0) must equal OVM"
    # dv>0 (closing): the relative-speed term can only lower accel vs OVM (monotone clip)
    assert fvdm.accel(12.0, 6.0, 15.0, pf) <= ovm_m.accel(12.0, 6.0, 15.0, pf[:4]) + 1e-9
    # and strictly lower where clipping does not mask it
    assert fvdm.accel(21.0, 3.0, 25.0, pf) < ovm_m.accel(21.0, 3.0, 25.0, pf[:4]) - 1e-6, \
        "FVDM should brake harder than OVM when closing"
    print("    lambda=0 == OVM; closing (dv>0) response <= OVM, strictly < unclipped -> PASS")

    # 2b) Delayed IDM: parameter plumbing + delay-agnostic accel equivalence
    print("\n[2b] delayed IDM (d-IDM) structural checks")
    didm = DelayedIDM(); pd6 = didm.default_params()          # v0,T,a_max,b,s0,tau
    assert didm.n_params == 6 and didm.param_names[-1] == "tau", "d-IDM needs a tau param"
    assert didm.within_bounds(pd6), "d-IDM defaults out of bounds"
    assert "didm" in available_models(), "d-IDM should be registered in MODELS"
    _idm5 = IDM()
    for (v, dv, s) in [(0.0, 0.0, 1000.0), (15.0, 8.0, 3.0), (25.0, -3.0, 40.0)]:
        a_didm = didm.accel(v, dv, s, pd6)                    # 6-vector (tau present)
        a_idm5 = _idm5.accel(v, dv, s, pd6[:5])               # same five parameters
        assert abs(a_didm - a_idm5) < 1e-12, "d-IDM.accel must equal IDM on theta[:5]"
    assert didm.tau_frames([30, 1.2, 1.5, 2.0, 2.0, 1.94], 0.1) == 19, "tau_frames rounding"
    assert didm.tau_frames([30, 1.2, 1.5, 2.0, 2.0, 0.0], 0.1) == 0, "tau=0 -> 0 frames"
    print("    n_params=6 (tau last); accel==IDM on theta[:5]; "
          "tau_frames(1.94s@10Hz)=19 -> PASS")

    # 3) real-data plausibility smoke test (NOT calibration)
    print("\n[3] real-data plausibility (default params, open-loop on observed states)")
    loaded = _load_si_pair()
    if loaded is None:
        print("    no CF-pair CSV found - skipping (synthetic checks above suffice)")
    else:
        v, dv, s, a_obs, fname, unit = loaded
        print(f"    file: {fname}   units: {unit}   n={len(v)}")
        print(f"    {'model':6s} {'pred a mean':>12s} {'pred a min/max':>18s} "
              f"{'RMSE vs obs a':>14s}")
        for name in available_models():
            m = get_model(name); p = m.default_params()
            a_pred = np.array([m.accel(v[i], dv[i], s[i], p) for i in range(len(v))])
            rmse = float(np.sqrt(np.nanmean((a_pred - a_obs) ** 2)))
            print(f"    {name:6s} {np.nanmean(a_pred):>12.3f} "
                  f"{np.nanmin(a_pred):>8.2f}/{np.nanmax(a_pred):<8.2f} "
                  f"{rmse:>14.3f}")
        print("    (default params are generic; RMSE here is a sanity check on\n"
              "     magnitude/sign only, not a calibrated fit.)")
 
    print("\nAll structural checks passed.")
 
 
if __name__ == "__main__":
    _selftest()