# Segment-based versus Sample-based Calibration of Car-Following Dynamics

Replication package for the manuscript. The framework anchors calibration at
**behavioral segment boundaries** (accel↔decel critical points detected by
PELT+) instead of at every trajectory sample, and evaluates the consequences on
two datasets:

* **Track 1 — human driving (NGSIM I-80).** Pooled, transferability-style
  calibration of IDM / Gipps / OVM under a segment-based objective and two
  sample-based baselines → **Table 1, Table 2, Fig 3**.
* **Track 2 — automated driving (OpenACC AstaZero).** Per-make ACC
  characterization: controller asymmetry and response lag, closed-loop string
  instability, and controller heterogeneity → **Figs 4–7**.

> **Naming.** User-facing text says *segment-based*. For output stability, the
> code and CLI keep the legacy token `phase` (objective key `phase`, flag
> `--features`, and `*_phase` columns in `calib_summary.csv`). `phase` in the
> code == *segment-based* in the paper. `acc_controller_heterogeneity.py` accepts
> either `*_phase` or `*_segment` columns.

---

## 1. Repository layout

**Core library (shared by both tracks)**

| Module | Role |
|---|---|
| `phase_segmentation.py` | PELT+ (CUSUM-screened, pruned exact) segmentation → critical points + per-segment features. *Read-only; shared.* |
| `pelt_plus_class.py` | PELT+ change-point engine used by the segmenter. |
| `cf_models.py` | IDM / Gipps / OVM / FVDM (`get_model`), parameter names and bounds. |
| `simulate.py` | Ballistic open-loop follower integration under a leader trajectory. |
| `cf_data.py` | NGSIM/TGSIM pair loading (`load_folder`), feet↔SI handling. |
| `objectives.py` | `PhaseAnchoredObjective` (segment-based) + sample RMSE objectives. |
| `calibrate.py` | Pooled Differential Evolution over training pairs. |
| `evaluate.py` | Held-out free-simulation metrics, paired tests, non-inferiority. |

**Track 1 drivers**

| Script | Produces |
|---|---|
| `extract_cf_pairs.py` | Builds the NGSIM pair set (`cf_ngsim_I80/`). |
| `run_experiment.py` | Main calibration (Table 1, Fig 3). |
| `run_ablation.py` + `ablation_anchors.py` | Feature / anchor ablation (Table 2). |

**Track 2 drivers** (OpenACC AstaZero)

| Script | Produces |
|---|---|
| `acc_controller_behavior.py` | Asymmetry, response lag, perturbation propagation (Fig 4, §5.1). |
| `calibrate_acc.py` | Per-make OVM calibration on one run. |
| `validate_acc_platoon.py` | Closed-loop string instability (Fig 5, §5.2). |
| `acc_controller_heterogeneity.py` | V_opt state-space overlay + radar (Figs 6–7, §5.3). |

**Data**

* `cf_ngsim_I80/` — extracted NGSIM I-80 pairs (`manifest.csv` + `train/` + `test/`). Built in §4.
* `oa_cf_pairs/` — extracted OpenACC pairs (`manifest.csv` + per-pair `oa_pair_*_f{pos}_{make}.csv`).
* `ASta_040719_platoon7.csv` — raw OpenACC AstaZero 5-vehicle platoon, read directly by the Track-2 behavior/heterogeneity scripts.

---

## 2. Environment

Python ≥ 3.9 with a standard scientific stack:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib
# optional, only if you export parquet pair sets: pip install pyarrow
```

NumPy 2.x is fine — the scripts alias `np.trapz → np.trapezoid` internally.
All figures render headless (`matplotlib Agg`).

**Sanity check without any data** (each supports a self-test):

```bash
python extract_cf_pairs.py --self-test -o /tmp/cf_selftest
python calibrate_acc.py --self-test
python validate_acc_platoon.py --self-test
```

---

## 3. Quick start (TL;DR)

```bash
# --- Track 1: NGSIM I-80 ---------------------------------------------------
# python extract_cf_pairs.py ngsim -i <I80_trajectory_csvs...> \
#     --n-train 100 --n-test 20 --min-duration 30 -o cf_ngsim_I80

python run_experiment.py --root cf_ngsim_I80 --model idm   --out experiment_out_idm
python run_experiment.py --root cf_ngsim_I80 --model ovm   --out experiment_out_ovm
python run_experiment.py --root cf_ngsim_I80 --model gipps --out experiment_out_gipps

python run_ablation.py  --root cf_ngsim_I80 --model idm   --out ablation_out

# --- Track 2: OpenACC AstaZero (oa_cf_pairs/ already prepared) --------------
python acc_controller_behavior.py --csv ASta_040719_platoon7.csv \
    --common-window --outdir acc_behavior

python calibrate_acc.py oa_cf_pairs --model ovm --objective both \
    --train-frac 0.7 -o acc_calib_ovm_run7

python validate_acc_platoon.py --pairs-dir oa_cf_pairs \
    --calib-dir acc_calib_ovm_run7 --which both -o acc_validation_ovm

python acc_controller_heterogeneity.py --csv ASta_040719_platoon7.csv \
    --calib acc_calib_ovm_run7/calib_summary.csv --all-makes --outdir acc_heterogeneity_ovm
```

Details, outputs, and paper mapping follow.

---

## 4. Data preparation

### 4.1 NGSIM I-80 pairs → `cf_ngsim_I80/`

Extract stable, single-lane, single-leader car-following pairs, split at the
**follower** level (no follower appears in both splits), keeping **native feet**:

```bash
python extract_cf_pairs.py ngsim \
    -i <I80_trajectory_csvs...> \
    --n-train 100 --n-test 20 --min-duration 30 \
    -o cf_ngsim_I80
```

* `-i` takes one or more I-80 trajectory CSVs (a `Preceding` leader column is
  required). Do **not** pass `--to-si`: segmentation is done on native ft/s and
  the loader converts features to SI (see §8).
* Split is follower-level by default (`--split-unit follower`), seed `42`.
* The manuscript's headline numbers use the reconstructed ("precise") I-80
  trajectories; feeding the raw CSVs instead gives the raw-data condition. Both
  are handled by the same extractor (raw↔precise is compared in §7).

Output: `cf_ngsim_I80/{manifest.csv, train/, test/, extraction_config.json}`.
Per-pair columns: `t, x_follower, v_follower, a_follower, x_leader, v_leader,
a_leader, leader_length, spacing, dv`.

### 4.2 OpenACC AstaZero pairs → `oa_cf_pairs/`

`oa_cf_pairs/` is the prepared OpenACC pair set: a `manifest.csv`
(`run, follower_position, follower_make, leader_make, path, …`) plus one
`oa_pair_*_f{pos}_{make}.csv` per consecutive predecessor→follower link
(1→2, 2→3, 3→4, 4→5). Same per-pair column schema as §4.1, all SI. The raw
platoon file `ASta_040719_platoon7.csv` is used directly by the Track-2
behavior and heterogeneity scripts.

---

## 5. Track 1 — segment-based calibration (Table 1, Fig 3)

One global parameter vector is fit per objective by **pooled** DE over all
training pairs, then graded on the held-out test pairs by free (open-loop)
simulation. Three objectives are compared: `phase` (segment-based, features
`{s_end, dist, phase_speed_ols}`), `sample/spacing` (strong baseline, RMSE on
the gap), and `sample/speed` (naive baseline, RMSE on speed).

```bash
python run_experiment.py --root cf_ngsim_I80 --model idm   --out experiment_out_idm
python run_experiment.py --root cf_ngsim_I80 --model ovm   --out experiment_out_ovm
python run_experiment.py --root cf_ngsim_I80 --model gipps --out experiment_out_gipps
```

**Defaults that match the paper** (no extra flags needed): features
`s_end,dist,phase_speed_ols`; DE `popsize 15`, `maxiter 60`; units `feet`;
non-inferiority margins `--delta-abs 0.5`, `--delta-rel 0.05`, `--gate and`;
seed `42`. To reproduce a specific ablation cell, pass e.g.
`--features s_end,dist`.

**Outputs per run** (`experiment_out_<model>/`): `theta.json`,
`test_per_pair.csv`, `summary.json`, `fig_test_spacing.png`,
`fig_paired_diff.png`, `fig_examples.png` (300 dpi).

**→ Table 1** is assembled across the three models from each run's
`summary.json → test_aggregate` (mean/median spacing RMSE, RMSPE,
collision-free fraction) and `test_per_pair.csv` (MAE, speed RMSE). The
segment-based objective attains the lowest mean spacing RMSE for every model
(≈14–15% below the sample/spacing baseline) despite never training on the gap.

**→ Fig 3** (observed-gap overlays, segment vs both sample baselines) is
`fig_examples.png`. Each run emits its own model panel; the manuscript
composites the congested stop-and-go pair across IDM / OVM / Gipps.

---

## 6. Track 1 — feature ablation (Table 2)

One-factor-at-a-time from the reference `{s_end, dist, phase_speed_ols}` on IDM,
sharing the protocol of §5. Reports held-out spacing RMSE per cell, a
railed-parameter count, per-feature diagonal weights and effective contribution,
and paired non-inferiority vs the sample/spacing baseline.

```bash
python run_ablation.py --root cf_ngsim_I80 --model idm --out ablation_out
```

**Outputs** (`ablation_out/`): `ablation_table.csv`, `feature_weights.csv`,
`ablation_summary.json`, `fig_ablation.png`, `fig_feature_weights.png`.

**→ Table 2** is `ablation_table.csv` (`RMSE_mean / RMSE_median / RMSE_pooled`
per feature set). The findings replicated: position-level features dominate;
`{s_end}` alone is within ~0.003 m of the full set; the velocity-only cell
`{v_end}` degrades sharply (re-imports differentiation noise); the
`+duration` negative control reproduces the reference exactly (zero residual
under fixed observed boundaries). The anchor-placement axis
(PELT+ ≈ uniform-K ≈ random-K at matched count) is emitted in the same table.

---

## 7. Track 2 — OpenACC AstaZero (Figs 4–7)

### 7.1 Controller asymmetry, response lag, perturbation propagation (Fig 4, §5.1)

Operates directly on the raw platoon CSV. Places PELT+ critical points on the
leader and each follower, matches leader→follower same-kind events to estimate
the response lag, and extracts peak accel/decel per regime.

```bash
python acc_controller_behavior.py --csv ASta_040719_platoon7.csv \
    --common-window --outdir acc_behavior
```

* `--common-window` is what produces the platoon overlay; use `--t0/--t1` to fix
  a manual window. `--smoke` runs a fast single-pair sanity pass.
* Follower acceleration is a Savitzky–Golay derivative (kept strictly
  descriptive; sensitivity across smoothing windows via `--sg-windows`).

**Outputs** (`acc_behavior/`): **`platoon_spacetime.png`** (oblique
space–time perturbation propagation with lag-tagged connectors = **Fig 4**),
`lag_connectors_*`, `lag_distance_*`, `phases_*`, `xcorr_*`,
`peaks_by_follower.png`, per-follower summary/event CSVs, `acc_peak_sensitivity.csv`.

### 7.2 Per-make OVM calibration

One OVM parameter set per make on the single run, train/held-out temporal split,
under both objectives (`sample` and segment-based `phase`).

```bash
python calibrate_acc.py oa_cf_pairs \
    --model ovm --objective both \
    --train-frac 0.7 \
    -o acc_calib_ovm_run7
```

* Segmentation and DE use the script defaults (§8). Add `--run <substr>` only if
  `oa_cf_pairs/` holds more than one run.
* Output dir named `acc_calib_ovm_run7/` so it matches the heterogeneity
  script's default `--calib` path in §7.4.

**Outputs** (`acc_calib_ovm_run7/`): `calib_f{pos}_{make}.json` (holds
`theta_sample` and `theta_phase`, i.e. the segment-based fit), `calib_summary.csv`
(learned parameters under legacy `*_phase` suffixes), `calib_config.json`.

### 7.3 Closed-loop string instability (Fig 5, §5.2)

Chains the per-make OVM controllers closed-loop on the held-out tail
(vehicle 1 = observed leader; each follower follows the *simulated* predecessor,
warm-started 15 s before the split). The model (OVM) is read from the
calibration JSONs, so no `--model` flag is needed.

```bash
python validate_acc_platoon.py \
    --pairs-dir oa_cf_pairs \
    --calib-dir acc_calib_ovm_run7 \
    --which both \
    -o acc_validation_ovm
```

Add `--hysteresis` for per-make spacing-hysteresis panels.

**Outputs** (`acc_validation_ovm/`): **`platoon_<run>.png`** — observed vs
sample-sim vs segment-sim, with cumulative disturbance amplitude Aᵢ and per-link
amplification Γᵢ = Aᵢ/Aᵢ₋₁ (= **Fig 5**); `platoon_metrics.csv`,
`phase_regime_metrics.csv`, `validation_metrics.csv`, plus `trace_*` and
`phase_decomp_*` figures.

### 7.4 Controller heterogeneity: V_opt curves + radar (Figs 6–7, §5.3)

Reuses the §7.1 segmentation and the §7.2 OVM parameters.

```bash
python acc_controller_heterogeneity.py \
    --csv ASta_040719_platoon7.csv \
    --calib acc_calib_ovm_run7/calib_summary.csv \
    --all-makes \
    --outdir acc_heterogeneity_ovm
```

(`--calib` defaults to `acc_calib_ovm_run7/calib_summary.csv`, so it can be
omitted if you kept that output name.)

**Outputs** (`acc_heterogeneity_ovm/`):

* **`fig5_6_speed_spacing_overlay.png`** — 2×2 speed–spacing state space per
  make, PELT+ regime-colored cloud with the learned V_opt(s) locus and critical
  spacing S_c overlaid (= **Fig 6**).
* **`fig5_7_controller_radar.png`** — five-axis heterogeneity radar
  (relaxation time 1/κ, critical spacing S_c, transition width w, median
  operating gap, peak deceleration) (= **Fig 7**).
* `fig5_8_stimulus_response_<make>.png` — stimulus–response temporal alignment
  (an extra diagnostic, not in the manuscript).

---

## 8. Reproducibility notes

* **Native-unit segmentation (Track 1).** NGSIM pairs are feet; PELT+ CUSUM
  thresholds are tuned for ft/s. `run_experiment.py`/`run_ablation.py` detect
  boundaries on native ft/s (indices are unit-invariant) and recompute features
  on SI arrays. Segmenting in SI under-detects boundaries.
* **Track-2 segmentation flags.** OpenACC channels are SI, so the Track-2
  scripts use SI CUSUM defaults. Defaults differ between scripts
  (`calibrate_acc.py`: `--cusum-threshold 2 --cusum-drift 0.3 --penalty 50`;
  `validate_acc_platoon.py`: `--cusum-threshold 2.1 --penalty 75`). If you
  override any segmentation flag, keep it **identical** across calibration,
  validation, and heterogeneity so the critical points line up.
* **Optimizer.** Differential Evolution, seed `42`. The segment-based objective
  runs with `polish=False` (the loss is piecewise-constant in the boundary
  indices, so L-BFGS-B polishing is harmful).
* **Pre-registered non-inferiority (Track 1).** Margins `δ_abs = 0.5 m`,
  `δ_rel = 0.05`, AND gate — set on the CLI before touching the test split; do
  not relax them retroactively.
* **Parameter identifiability.** Reported as-is: IDM `b` floors under
  position-level objectives; OVM `κ` and `s_c` rail structurally from
  spacing-level data. **Tesla `v_max`** is unidentified in-run (the follower
  never reaches free speed) and is pinned to an observed proxy
  (max speed + 2 m/s) for the Fig 6/7 OVM fit.
* **ACC inference scope.** One run, one vehicle per make, with
  make/position/predecessor confounded. Cross-make numbers are **descriptive
  per-position** characterizations, not manufacturer-level inference; windowed
  sub-segments are not independent samples. Follower acceleration in Track 2 is
  a Savitzky–Golay derivative, kept descriptive.
* **Determinism.** Re-running with the same seeds reproduces `theta.json` /
  `calib_*.json` and every derived table and figure.

---

## 9. Paper artifact → command → output

| Paper artifact | Command | Key output |
|---|---|---|
| **Table 1** (Track-1 accuracy, 3 models) | `run_experiment.py` ×3 (idm/ovm/gipps) | `summary.json` + `test_per_pair.csv` |
| **Fig 3** (gap overlays, segment vs sample) | `run_experiment.py` | `fig_examples.png` (per model) |
| **Table 2** (feature ablation) | `run_ablation.py --model idm` | `ablation_table.csv` |
| **Fig 4** (§5.1 propagation + asymmetry) | `acc_controller_behavior.py --common-window` | `platoon_spacetime.png` |
| **Fig 5** (§5.2 closed-loop string instability) | `validate_acc_platoon.py --which both` | `platoon_<run>.png` |
| **Fig 6** (§5.3 V_opt state space) | `acc_controller_heterogeneity.py` | `fig5_6_speed_spacing_overlay.png` |
| **Fig 7** (§5.3 controller radar) | `acc_controller_heterogeneity.py` | `fig5_7_controller_radar.png` |

Fig 1 (PELT+ segmentation illustration) is a demonstration of
`phase_segmentation.py` (see `run_demo.py`) and is not part of the quantitative
pipeline.

---

