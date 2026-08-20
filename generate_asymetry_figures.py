#!/usr/bin/env python3
"""
Generate the two NEW Discussion figures from the segment-anchored OVM calibration.

  Fig 5-5  Controller heterogeneity learned by segment-anchored OVM:
           (a) learned optimal-velocity policy V_opt(s) per make;
           (b) sensitivity kappa vs aggregate response lag tau* (Fig 5-4, n=10),
               showing higher kappa (faster relaxation) <-> shorter lag.

  Fig 5-2b Spacing hysteresis per make (2x2): observed vs segment-anchored OVM,
           open-loop on the held-out tail against the observed leader.

Single run (040719_platoon7); positions/makes confounded -> descriptive.
"""
import types
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import acc_controller_behavior as acb
import acc_string_stability as vap
import simulate
from cf_models import get_model

CSV = "ASta_040719_platoon7.csv"
OVM = get_model("ovm")

# ---- consistent per-make aesthetics ---------------------------------------- #
MAKE_COL = {
    "Tesla Model 3": "#0072B2", "BMW X5": "#009E73",
    "Audi A6": "#E69F00", "Mercedes A-Class": "#CC3311",
}
# position -> canonical display make (align calib rows to pair makes)
POS_MAKE = {2: "Tesla Model 3", 3: "BMW X5", 4: "Audi A6", 5: "Mercedes A-Class"}
# aggregate response lag from Figure 5-4 (n=10 means, published)
TAU_STAR_FIG54 = {"Tesla Model 3": 1.70, "BMW X5": 2.60,
                  "Audi A6": 2.30, "Mercedes A-Class": 2.10}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def v_opt(s, kappa, v_max, s_c, w):
    t_sc = np.tanh(s_c / w)
    return v_max * (np.tanh((s - s_c) / w) + t_sc) / (1.0 + t_sc)


def load_calib():
    cal = pd.read_csv("acc_calib_ovm_run7/calib_summary.csv")
    cal["make"] = cal["follower_position"].map(POS_MAKE)
    return cal


def load_pairs():
    pairs, info = acb.load_openacc_asta(CSV)
    by_pos = {p.follower_veh: p for p in pairs}
    return by_pos, info


# =========================================================================== #
# Figure 5-5
# =========================================================================== #
def figure_5_5(cal, by_pos):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # ---- (a) learned optimal-velocity policy V_opt(s) --------------------- #
    s_grid = np.linspace(5.0, 55.0, 400)
    order = [2, 3, 4, 5]
    for pos in order:
        row = cal[cal["follower_position"] == pos].iloc[0]
        make = POS_MAKE[pos]
        col = MAKE_COL[make]
        vo = v_opt(s_grid, row.kappa_phase, row.v_max_phase, row.s_c_phase, row.w_phase)
        axA.plot(s_grid, vo, color=col, lw=2.2, label=make)
        # median operating gap marker
        p = by_pos[pos]
        s_med = float(np.nanmedian(p.s))
        vo_med = v_opt(s_med, row.kappa_phase, row.v_max_phase, row.s_c_phase, row.w_phase)
        axA.scatter([s_med], [vo_med], color=col, s=55, zorder=6,
                    edgecolor="white", linewidth=0.8)
    # observed operating band (p5-p95 of all gaps)
    all_s = np.concatenate([by_pos[pos].s for pos in order])
    lo, hi = np.nanpercentile(all_s, 5), np.nanpercentile(all_s, 95)
    axA.axvspan(lo, hi, color="0.5", alpha=0.08, lw=0)
    axA.text(0.5 * (lo + hi), axA.get_ylim()[1] * 0.06,
             "observed gap band (p5–p95)", ha="center", va="bottom",
             fontsize=8.5, color="0.35")
    axA.set_xlabel("net spacing $s$ (m)")
    axA.set_ylabel(r"optimal velocity $V_{\mathrm{opt}}(s)$ (m/s)")
    axA.set_title("(a) Learned steady-state speed–gap policy")
    axA.legend(title="follower (segment-anchored OVM)", loc="upper left")
    axA.set_xlim(5, 55)
    axA.set_ylim(bottom=0)

    # ---- (b) kappa vs aggregate response lag tau* ------------------------- #
    ks, taus = [], []
    for pos in order:
        row = cal[cal["follower_position"] == pos].iloc[0]
        make = POS_MAKE[pos]
        col = MAKE_COL[make]
        k = float(row.kappa_phase)
        tau = TAU_STAR_FIG54[make]
        ks.append(k); taus.append(tau)
        axB.scatter([k], [tau], color=col, s=140, zorder=6,
                    edgecolor="white", linewidth=1.0)
        dx = 0.004 if make != "Tesla Model 3" else -0.004
        ha = "left" if make != "Tesla Model 3" else "right"
        axB.annotate(make, (k, tau), xytext=(k + dx, tau + 0.03),
                     ha=ha, va="bottom", fontsize=9.5, color=col)
    # light guide line (rank trend, not a regression claim)
    ks = np.array(ks); taus = np.array(taus)
    xline = np.linspace(ks.min() - 0.01, ks.max() + 0.01, 50)
    b1, b0 = np.polyfit(ks, taus, 1)
    axB.plot(xline, b0 + b1 * xline, color="0.55", ls="--", lw=1.2, zorder=2)
    from scipy.stats import spearmanr
    rho = spearmanr(ks, taus).correlation
    axB.text(0.03, 0.05, f"rank inverse:  Spearman $\\rho$ = {rho:+.0f}",
             transform=axB.transAxes, fontsize=9.5, color="0.25",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    axB.set_xlabel(r"segment-anchored sensitivity $\kappa$  (1/s)")
    axB.set_ylabel(r"aggregate response lag $\tau^*$ (s)  [Fig. 5-4, $n{=}10$]")
    axB.set_title("(b) Faster relaxation ↔ shorter lag")
    axB.set_xlim(ks.min() - 0.03, ks.max() + 0.035)
    axB.set_ylim(taus.min() - 0.25, taus.max() + 0.35)

    fig.tight_layout()
    out = "fig5_5_controller_heterogeneity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] wrote {out}  (kappa vs tau* Spearman rho = {rho:+.2f})")
    return out


# =========================================================================== #
# Hysteresis 2x2 (open-loop OVM on held-out tail)
# =========================================================================== #
def _heldout_arrays(p, split):
    """Reconstruct the sim inputs for one pair on the held-out tail [split:]."""
    t = p.t[split:]
    vL = p.v_leader[split:]
    vF = p.v_follower[split:]
    s = p.s[split:]
    xL = acb.cumdist(t, vL)                # leader position (integrated speed)
    x_f0 = float(xL[0] - s[0])             # L = 0 (IVS is net gap)
    return t, xL, vL, vF, s, x_f0


def figure_hysteresis(cal, by_pos, split):
    order = [2, 3, 4, 5]
    edges_bin = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
    axes = axes.ravel()
    widths = {}
    for ax, pos in zip(axes, order):
        make = POS_MAKE[pos]
        col = MAKE_COL[make]
        row = cal[cal["follower_position"] == pos].iloc[0]
        theta = [row.kappa_phase, row.v_max_phase, row.s_c_phase, row.w_phase]
        p = by_pos[pos]
        t, xL, vL, vF, s, x_f0 = _heldout_arrays(p, split)
        r = simulate.simulate(OVM, theta, t, xL, vL, 0.0, x_f0, float(vF[0]))

        # restrict edges to the well-populated speed band (trim sparse extremes)
        vlo = float(np.nanpercentile(vF, 3)); vhi = float(np.nanpercentile(vF, 97))
        vmin = np.floor(vlo); vmax = np.ceil(vhi)
        edges = np.arange(vmin, vmax + edges_bin, edges_bin)
        c = 0.5 * (edges[:-1] + edges[1:])

        wo, sco, soo = vap.hyst_width(vF, s, vL, edges, mc=8, sw=5)
        ws, scs, sos = vap.hyst_width(r.v, r.s, vL, edges, mc=8, sw=5)
        widths[make] = (wo, ws, float(np.min(r.a)))

        # observed loop (grey band + black lines) — only where both branches exist
        vo = ~(np.isnan(sco) | np.isnan(soo))
        ax.fill_between(c[vo], sco[vo], soo[vo], color="0.55", alpha=0.30, lw=0)
        ax.plot(c[vo], sco[vo], color="k", lw=1.6, label="observed – closing")
        ax.plot(c[vo], soo[vo], color="k", lw=1.6, ls="--", label="observed – opening")
        # simulated loop (make colour)
        vs = ~(np.isnan(scs) | np.isnan(sos))
        ax.fill_between(c[vs], scs[vs], sos[vs], color=col, alpha=0.20, lw=0)
        ax.plot(c[vs], scs[vs], color=col, lw=1.9, label="segment-OVM – closing")
        ax.plot(c[vs], sos[vs], color=col, lw=1.9, ls="--", label="segment-OVM – opening")

        ax.set_xlim(vlo - 1, vhi + 1)
        _yv = np.concatenate([sco[vo], soo[vo], scs[vs], sos[vs]])
        if _yv.size:
            ax.set_ylim(max(0, np.nanmin(_yv) - 2), np.nanmax(_yv) + 3)
        ax.set_title(f"{make}\nobs |width| = {abs(wo):.2f} m   "
                     f"sim |width| = {abs(ws):.2f} m", fontsize=10.5)
        ax.set_xlabel("follower speed (m/s)")
        ax.set_ylabel("net spacing (m)")
        if pos == 2:
            ax.legend(loc="upper left", fontsize=7.8, ncol=1, framealpha=0.9)

    fig.suptitle("Spacing hysteresis: observed vs segment-anchored OVM",
                 y=0.995, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = "fig5_2b_hysteresis.png"
    fig.savefig(out)
    plt.close(fig)
    print("[fig] wrote", out)
    print("[hysteresis] observed vs sim |width| (m), sim peak decel (m/s^2):")
    for mk, (wo, ws, pdc) in widths.items():
        print(f"    {mk:18s}  obs={abs(wo):5.2f}  sim={abs(ws):5.2f}  peak_decel={pdc:6.2f}")
    return out, widths


if __name__ == "__main__":
    cal = load_calib()
    by_pos, info = load_pairs()
    split = int(cal["n_train"].iloc[0])   # 6459
    print(f"[setup] split_index={split}  held-out frames={info['n_samples']-split}")
    f55 = figure_5_5(cal, by_pos)
    fhy, widths = figure_hysteresis(cal, by_pos, split)
    print("\n[done]", f55, fhy)
