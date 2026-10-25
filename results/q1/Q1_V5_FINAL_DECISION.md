# Q1 V5 final decision record

## Decisions

**Phase B — `Q1_FIXED_POLICY`: FAIL.** A single regime-blind `old_plus_eb` rule was selected using only the 40 training and 8 inner-validation backgrounds, then evaluated on 16 fresh ID-like and 16 fresh background-OOD backgrounds. Five of six pre-existing acceptance gates pass. The U4 gate fails because V5 does not significantly beat the population-mean U4 baseline in ID-like backgrounds. Its paired V5-minus-population interval is `-0.019 [-0.039, 0.004]`; OOD is `+0.277 [0.246, 0.307]`. The active results below do not change this fixed-policy verdict.

**Phase D — `Q1_ACTIVE_EXCITATION`: FAIL.** The Fisher objective increased nuisance-residualized information for `log(b)` over sinusoidal and random exploration in both regimes. The preregistered success gate also required positive U4 skill in both regimes and superiority to sinusoidal U4 skill in both regimes. The U4 conditions fail: confidence intervals include zero in both regimes, and OOD U4 skill is significantly below sinusoidal. This is a separate active-excitation outcome, not a PASS for the fixed-policy benchmark.

Neither protocol accessed protected test splits.

## Phase B — fixed-policy results

All cells below average the 21 programs within each background; intervals bootstrap the 16 development backgrounds. Independent-query MSE uses 256 fresh query transitions per background/program. The complete five-budget outputs, per-program rows, U1–U4 effects, baselines, intervals, and resolved configuration are in [`q1_v5_fixed_confirmatory_dev`](q1_v5_fixed_confirmatory_dev/).

| Regime | Estimator | Functional-class accuracy at N=100 (95% CI) | Independent-query MSE (95% CI) |
|---|---|---:|---:|
| ID-like | V5 (`old_plus_eb`) | 0.812 [0.777, 0.848] | 2.621e-7 [2.599e-7, 2.645e-7] |
| ID-like | V2 retrained | 0.440 [0.405, 0.476] | 6.957e-6 [6.482e-6, 7.411e-6] |
| ID-like | V3 | 0.048 [0.048, 0.048] | 8.633e-5 [8.432e-5, 8.856e-5] |
| ID-like | M4 | 0.348 [0.318, 0.381] | 8.733e-6 [7.834e-6, 9.673e-6] |
| OOD | V5 (`old_plus_eb`) | 0.949 [0.929, 0.970] | 2.659e-7 [2.633e-7, 2.684e-7] |
| OOD | V2 retrained | 0.482 [0.467, 0.497] | 2.037e-5 [1.925e-5, 2.147e-5] |
| OOD | V3 | 0.048 [0.048, 0.048] | 1.443e-4 [1.416e-4, 1.468e-4] |
| OOD | M4 | 0.283 [0.271, 0.295] | 5.602e-5 [5.262e-5, 5.931e-5] |

The paired V5-minus-V2 functional-class accuracy differences are `+0.372 [0.312, 0.435]` ID-like and `+0.467 [0.443, 0.491]` OOD. Both ordered action-pair orders have V5 functional-class accuracy `1.000` in each regime at N=100. The population-mean, identity-operator, and direct-context results are in the detailed summary; direct-context is a prediction-only baseline and does not output program classes or ordered compositions.

| Regime | V5 U4 effect skill (95% CI) | Population-mean U4 skill (95% CI) | Paired skill difference (95% CI) |
|---|---:|---:|---:|
| ID-like | 0.953 [0.937, 0.970] | 0.972 [0.957, 0.986] | -0.019 [-0.039, 0.004] |
| OOD | 0.944 [0.917, 0.968] | 0.667 [0.650, 0.683] | +0.277 [+0.246, +0.307] |

All six fixed gates are recorded in `q1_v5_fixed_confirmatory_dev/summary.json`: U4 beats population in both regimes **FAIL**; U3/U4 skill positive in both **PASS**; U1–U3 noninferior to V3 at margin 0.02 **PASS**; both ordered-pair accuracies at least 0.80 **PASS**; ordered-pair class accuracy noninferior to V3 at margin 0.02 **PASS**; V5 class accuracy beats V2 in both regimes **PASS**.

## Phase D — active excitation results

The policy chose from `{-1,0,1}²` using a two-step frozen-model rollout and maximized `log(1 + I_eff(log b))` after nuisance residualization over six other task parameters and five learned-response coordinates. The matched uncertainty baseline maximized the twelve-coordinate information log-determinant. Four fresh backgrounds per regime, all 21 programs, four policies, five adaptation budgets, 100 adaptation transitions, and 256 independent query transitions were used. No action-bound violations occurred. Each active policy evaluated 151,200 candidate-action scores; active-action selection and trajectory-generation CPU time was 105.5 s for the target-information policy and 95.0 s for D-optimal, versus 9.3 s each for sinusoidal and random policy generation. Program ranking/refitting was run on the same cases and budgets for all policies. Full action paths and accounting are in [`q1_v5_active_excitation_dev`](q1_v5_active_excitation_dev/).

| Regime | Policy | N=100 nuisance-residualized Fisher for log(b), mean (95% CI) | N=100 functional-class accuracy | N=100 U4 effect skill (95% CI) |
|---|---|---:|---:|---:|
| ID-like | Sinusoidal | 12.35 [9.17, 16.08] | 0.869 | 0.210 [-0.044, 0.674] |
| ID-like | Random | 11.96 [8.43, 15.60] | 0.869 | -0.039 [-0.295, 0.136] |
| ID-like | Active `I_eff(log b)` | 55.18 [35.71, 80.74] | 0.845 | 0.598 [-0.032, 0.986] |
| ID-like | Uncertainty D-optimal | 34.01 [28.50, 40.76] | 0.929 | 0.744 [0.248, 0.995] |
| OOD | Sinusoidal | 42.19 [29.22, 57.15] | 0.964 | 0.958 [0.936, 0.979] |
| OOD | Random | 41.32 [28.92, 55.22] | 0.952 | 0.791 [0.437, 0.983] |
| OOD | Active `I_eff(log b)` | 59.12 [44.32, 73.92] | 0.833 | 0.432 [-0.081, 0.944] |
| OOD | Uncertainty D-optimal | 115.96 [89.17, 148.70] | 0.952 | 0.987 [0.979, 0.993] |

At N=100, active Fisher exceeds sinusoidal by `+42.82 [26.28, 64.00]` ID-like and `+16.92 [8.43, 25.41]` OOD; it exceeds random by `+43.22 [26.43, 64.18]` and `+17.80 [10.07, 25.53]`. It is also above D-optimal in ID-like backgrounds by `+21.17 [3.14, 41.34]`, but below it in OOD by `-56.84 [-84.34, -24.55]`. The target-information policy therefore increased its target information against the two exploration controls but did not dominate the uncertainty-driven baseline across regimes. Its U4 skill does not meet the active success gate.

## Mathematical status

`docs/q1_v5_identifiability_and_recovery.md` now records exact algebraic properties and local Gaussian Fisher identities as proved; estimation, operator recovery, finite-candidate recovery, and adaptive concentration as conditional on explicit regularity, separation, independence, and excitation assumptions; and uniform finite-sample guarantees for the implemented learned-response pipeline as open. The active Fisher objective is a local model-based surrogate, not a guarantee of realized U4 or program-recovery improvement.

## Provenance and verification

- Branch: `codex/q1-v5-identifiability-active`; requested starting commit: `43245aee373ddd1061406a8c22f17f7ba686e30d`.
- Phase B: seed `20261031`; 40 training, 8 inner-validation, and 16 development backgrounds per regime. Phase D: seed `20261033`; 4 fresh development backgrounds per regime. Split manifests record environment parameters and collision audits.
- Both runs used CPU, `dt=0.05`, `sigma=0.0005`, 96 old-support transitions, 100 adaptation transitions, and 256 independent-query transitions. Protected test data were not used.
- All Q1 tests: **95 passed, 1 failed**. The failure is the existing stochastic V2 test `tests/test_q1_v2_root_repair.py::test_effect_aware_training_learns_small_in_basis_effects_and_selects_on_inner_only`; it failed again when run alone (`after[3].skill=-1.9731` vs baseline `0`). The V5/active focused set passed **11/11**, and the Q1 protocol/model subset passed **42/42**. No V2 root-repair source or acceptance criterion was changed. `py_compile` passed for the new runners and active-design module; `git diff --check` passed.
- Phase B elapsed time: 983.5 seconds. Phase D elapsed time and source/checkpoint hashes are recorded in each run's `provenance.json`.
