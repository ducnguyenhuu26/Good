# Q1 V3 repair ablations — development decision

Fresh development only; 16 ID-like and 16 background-OOD backgrounds, all 21 programs, original test split untouched.

## N=100 functional-class accuracy

| Regime | Original V3 | Action only | Operator only | Combined | V2 original | V2 retrained | M4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | 0.725 | 0.725 | 0.738 | 0.737 | 0.429 | 0.626 | 0.369 |
| BACKGROUND_OOD_DEVELOPMENT | 0.626 | 0.626 | 0.619 | 0.619 | 0.491 | 0.582 | 0.315 |

## Functional-class learning curve (marginal scorer)

| Regime | N | Original V3 | Combined | V2 retrained |
|---|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | 5 | 0.601 | 0.580 | 0.432 |
| IDLIKE_DEVELOPMENT | 10 | 0.673 | 0.685 | 0.525 |
| IDLIKE_DEVELOPMENT | 20 | 0.725 | 0.726 | 0.589 |
| IDLIKE_DEVELOPMENT | 50 | 0.713 | 0.723 | 0.649 |
| IDLIKE_DEVELOPMENT | 100 | 0.725 | 0.737 | 0.626 |
| BACKGROUND_OOD_DEVELOPMENT | 5 | 0.506 | 0.469 | 0.360 |
| BACKGROUND_OOD_DEVELOPMENT | 10 | 0.677 | 0.619 | 0.464 |
| BACKGROUND_OOD_DEVELOPMENT | 20 | 0.674 | 0.628 | 0.518 |
| BACKGROUND_OOD_DEVELOPMENT | 50 | 0.628 | 0.626 | 0.604 |
| BACKGROUND_OOD_DEVELOPMENT | 100 | 0.626 | 0.619 | 0.582 |

## U1–U4 independent-query effect skill

| Regime | Code | Original V3 | Action only | Operator only | Combined | Combined population mean |
|---|---|---:|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | U1 | 0.975 | 0.973 | 0.974 | 0.972 | 0.963 |
| IDLIKE_DEVELOPMENT | U2 | 0.936 | 0.932 | 0.937 | 0.934 | 0.937 |
| IDLIKE_DEVELOPMENT | U3 | 0.963 | 0.963 | 0.918 | 0.916 | 0.867 |
| IDLIKE_DEVELOPMENT | U4 | -4.612 | -4.756 | -3.234 | -3.319 | -3.599 |
| BACKGROUND_OOD_DEVELOPMENT | U1 | 0.942 | 0.939 | 0.941 | 0.938 | 0.934 |
| BACKGROUND_OOD_DEVELOPMENT | U2 | 0.857 | 0.851 | 0.855 | 0.849 | 0.872 |
| BACKGROUND_OOD_DEVELOPMENT | U3 | 0.982 | 0.981 | 0.944 | 0.942 | 0.723 |
| BACKGROUND_OOD_DEVELOPMENT | U4 | -2.397 | -2.518 | -1.520 | -1.590 | -1.016 |

## Ordered compositions at N=100

| Regime | Model | U1→U2 | U2→U1 |
|---|---|---:|---:|
| IDLIKE_DEVELOPMENT | original_v3 | 1.000 | 1.000 |
| IDLIKE_DEVELOPMENT | action_response_only | 1.000 | 1.000 |
| IDLIKE_DEVELOPMENT | operator_selectivity_only | 1.000 | 1.000 |
| IDLIKE_DEVELOPMENT | combined_action_and_operator | 1.000 | 1.000 |
| BACKGROUND_OOD_DEVELOPMENT | original_v3 | 1.000 | 1.000 |
| BACKGROUND_OOD_DEVELOPMENT | action_response_only | 1.000 | 1.000 |
| BACKGROUND_OOD_DEVELOPMENT | operator_selectivity_only | 1.000 | 1.000 |
| BACKGROUND_OOD_DEVELOPMENT | combined_action_and_operator | 1.000 | 1.000 |

## U4 and ablation gates

| Ablation | ID U4 skill | OOD U4 skill | ID U4 minus prior CI | OOD U4 minus prior CI | Verdict |
|---|---:|---:|---:|---:|---|
| action_response_only | -4.756 [-8.339,-1.817] | -2.518 [-3.596,-1.552] | 0.045 [-0.191,0.292] | -1.000 [-1.422,-0.646] | FAIL |
| operator_selectivity_only | -3.234 [-5.799,-1.125] | -1.520 [-2.315,-0.808] | 0.292 [0.078,0.589] | -0.534 [-0.830,-0.285] | FAIL |
| combined_action_and_operator | -3.319 [-5.933,-1.167] | -1.590 [-2.408,-0.855] | 0.281 [0.064,0.577] | -0.574 [-0.885,-0.311] | FAIL |

## Diagnosis

U1–U3 effect non-inferiority and both ordered-composition criteria use paired background bootstrap intervals with the predeclared 0.02 margin. U4 requires positive skill in each regime and a positive 95% paired interval over the population-mean baseline.
The flexible response spline did not improve training-inner old-law validation (NLL/observation 3.037) over original V3 (2.923). Learned assignment probabilities are recorded per opaque code and seed in `repair_metrics.json`; U4's assignment remained diffuse (max group probability below 0.29), so the selective constraint did not isolate its mechanism.
Selectivity-only and combined reduce U4 error but leave skill negative; selectivity variants also regress U3 beyond the predeclared margin. The learned assignment map is data-driven and contains no hard-coded physical code identities.
The combined ablation verdict is **FAIL**. The repair did not satisfy every gate. Stop the current Q1 formulation; do not continue architectural tweaks or open another confirmation on these development backgrounds.
