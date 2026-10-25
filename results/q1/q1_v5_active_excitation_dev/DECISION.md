# Q1_ACTIVE_EXCITATION development decision

**Verdict: FAIL under the predeclared active-success gate.** The Fisher gate passed: nuisance-residualized `I_eff(log b)` at N=100 exceeded both sinusoidal and random controls in both regimes. The two U4 gates failed: the active U4-skill intervals include zero in both regimes, and OOD active U4 skill is significantly below sinusoidal. This is a separate active protocol and does not alter Phase B's fixed-policy verdict.

| Regime | Policy | N=100 `I_eff(log b)` (95% background CI) | Functional-class accuracy | U4 effect skill (95% background CI) |
|---|---|---:|---:|---:|
| ID-like | Sinusoidal | 12.35 [9.17, 16.08] | 0.869 | 0.210 [-0.044, 0.674] |
| ID-like | Random | 11.96 [8.43, 15.60] | 0.869 | -0.039 [-0.295, 0.136] |
| ID-like | Active `I_eff(log b)` | 55.18 [35.71, 80.74] | 0.845 | 0.598 [-0.032, 0.986] |
| ID-like | Uncertainty D-optimal | 34.01 [28.50, 40.76] | 0.929 | 0.744 [0.248, 0.995] |
| OOD | Sinusoidal | 42.19 [29.22, 57.15] | 0.964 | 0.958 [0.936, 0.979] |
| OOD | Random | 41.32 [28.92, 55.22] | 0.952 | 0.791 [0.437, 0.983] |
| OOD | Active `I_eff(log b)` | 59.12 [44.32, 73.92] | 0.833 | 0.432 [-0.081, 0.944] |
| OOD | Uncertainty D-optimal | 115.96 [89.17, 148.70] | 0.952 | 0.987 [0.979, 0.993] |

At N=100 the active-minus-sinusoidal Fisher differences are `+42.82 [26.28, 64.00]` ID-like and `+16.92 [8.43, 25.41]` OOD. Active-minus-random differences are `+43.22 [26.43, 64.18]` and `+17.80 [10.07, 25.53]`. Active-minus-sinusoidal U4 skill is `+0.388 [-0.110, 0.885]` ID-like and `-0.526 [-1.037, -0.015]` OOD.

The run used four fresh development backgrounds per regime, all 21 programs, budgets `{5,10,20,50,100}`, 96 old-support transitions, 100 adaptation transitions, and 256 independent-query transitions per program. All policies share the same per-case old support, simulator, query seed and query budget; only the adaptation action policy changes. No action-bound violations occurred. Each active selector evaluated 151,200 candidate-action scores; trajectory/acquisition time was 105.5 s for target-information selection and 95.0 s for D-optimal, versus 9.3 s each for sinusoidal and random policy generation. All policies performed the same candidate ranking and selected-program refit work.

Full per-background results, action paths, Fisher matrices, U4 effects, resolved configuration, seed/split manifest, frozen training records, and provenance are in this directory. See `summary.json` for the five-budget paired bootstrap comparisons. Protected test data were not used.
