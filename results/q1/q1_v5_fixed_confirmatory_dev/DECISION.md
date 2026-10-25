# Q1 V5 fixed-policy confirmatory decision

**Verdict: FAIL.** The single deployable rule selected from training/inner-validation only is `old_plus_eb`. It is regime-blind and was frozen before fresh development evaluation. Five of six acceptance gates pass; U4 skill does not significantly exceed the population-mean baseline in both regimes.

All values below average the 21 programs within each background, with 95% bootstrap confidence intervals over 16 backgrounds. Query MSE is measured on 256 independent transitions per background/program.

| Regime | Method | N=100 functional-class accuracy (95% CI) | N=100 independent-query MSE (95% CI) |
|---|---|---:|---:|
| ID-like | V5 `old_plus_eb` | 0.812 [0.777, 0.848] | 2.621e-7 [2.599e-7, 2.645e-7] |
| ID-like | V2 retrained | 0.440 [0.405, 0.476] | 6.957e-6 [6.482e-6, 7.411e-6] |
| ID-like | V3 | 0.048 [0.048, 0.048] | 8.633e-5 [8.432e-5, 8.856e-5] |
| ID-like | M4 | 0.348 [0.318, 0.381] | 8.733e-6 [7.834e-6, 9.673e-6] |
| OOD | V5 `old_plus_eb` | 0.949 [0.929, 0.970] | 2.659e-7 [2.633e-7, 2.684e-7] |
| OOD | V2 retrained | 0.482 [0.467, 0.497] | 2.037e-5 [1.925e-5, 2.147e-5] |
| OOD | V3 | 0.048 [0.048, 0.048] | 1.443e-4 [1.416e-4, 1.468e-4] |
| OOD | M4 | 0.283 [0.271, 0.295] | 5.602e-5 [5.262e-5, 5.931e-5] |

Population-mean and identity-operator rows are included at every N in `summary.json` and the per-program table. The direct-context baseline has independent-query MSE at each N but does not output program classes or ordered compositions. At N=100 both ordered action-pair orders have V5 functional-class accuracy 1.000 in both regimes.

| Regime | V5 U4 skill (95% CI) | Population-mean U4 skill (95% CI) | Paired V5-minus-population (95% CI) |
|---|---:|---:|---:|
| ID-like | 0.953 [0.937, 0.970] | 0.972 [0.957, 0.986] | -0.019 [-0.039, 0.004] |
| OOD | 0.944 [0.917, 0.968] | 0.667 [0.650, 0.683] | +0.277 [+0.246, +0.307] |

The U4 comparison includes zero in ID-like backgrounds, which fails the required both-regimes test. U1–U4 effect rows, all five budgets, per-program accuracy/MSE and the complete gate JSON are in `effect_per_background.csv`, `per_background_per_program_metrics.csv`, `per_program_summary.csv`, and `summary.json`. Direct-context query predictions are in `direct_context_per_background.csv`.

The runner used 40 training backgrounds, 8 inner-validation backgrounds, and 16 fresh development backgrounds per regime. It preserved `dt=0.05`, `sigma=0.0005`, 96 old-support transitions, 100 adaptation transitions, 256 independent-query transitions, and the fixed sinusoidal-with-jitter action policy. V2/V3/M4 and all V5-aligned controls used the same generated observations for a case. Protected test data were not used. See `config.resolved.json`, `split_manifest.json`, `training_records.json`, and `provenance.json` for run details.
