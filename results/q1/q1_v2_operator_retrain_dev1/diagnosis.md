# Q1 V2 operator retraining diagnosis

This is a fresh development diagnostic. The original V2 basis/prior checkpoint and architecture were reused unchanged; the operator banks were trained on all 480 training episodes in each epoch, with balanced opaque-code ordering. Operator checkpoint epochs were selected by predictive NLL on inner-training backgrounds. No test split was used.

## Retrained operator results

| Regime | N | Bayesian marginal | Point estimate | Rows |
|---|---:|---:|---:|---:|
| BACKGROUND_OOD_DEVELOPMENT | 5 | 0.411 | 0.324 | 672 |
| BACKGROUND_OOD_DEVELOPMENT | 10 | 0.460 | 0.378 | 672 |
| BACKGROUND_OOD_DEVELOPMENT | 20 | 0.519 | 0.257 | 672 |
| BACKGROUND_OOD_DEVELOPMENT | 50 | 0.610 | 0.246 | 672 |
| BACKGROUND_OOD_DEVELOPMENT | 100 | 0.577 | 0.238 | 672 |
| IDLIKE_DEVELOPMENT | 5 | 0.457 | 0.427 | 672 |
| IDLIKE_DEVELOPMENT | 10 | 0.525 | 0.496 | 672 |
| IDLIKE_DEVELOPMENT | 20 | 0.580 | 0.430 | 672 |
| IDLIKE_DEVELOPMENT | 50 | 0.629 | 0.426 | 672 |
| IDLIKE_DEVELOPMENT | 100 | 0.619 | 0.397 | 672 |

## Single-intervention effect skill at N=100

| Intervention | Mean skill | Median skill | Rows |
|---|---:|---:|---:|
| U1 | 0.912 | 0.914 | 64 |
| U2 | 0.932 | 0.966 | 64 |
| U3 | -1.710 | -1.085 | 64 |
| U4 | -214.473 | -92.594 | 64 |

## Ordered compositions at N=100

| Program | Functional-class accuracy | Rows |
|---|---:|---:|
| (U1,U2) | 1.000 | 64 |
| (U2,U1) | 1.000 | 64 |

## Comparison on the same fresh development cases

| Regime / model / N | Functional-class accuracy | Mean query Δv MSE | Cases |
|---|---:|---:|---:|
| IDLIKE_DEVELOPMENT|V2_RETRAINED|N=5 | 0.457 | 5.81e-06 | 672 |
| IDLIKE_DEVELOPMENT|V2_RETRAINED|N=10 | 0.525 | 5.57e-06 | 672 |
| IDLIKE_DEVELOPMENT|V2_RETRAINED|N=20 | 0.580 | 5.68e-06 | 672 |
| IDLIKE_DEVELOPMENT|V2_RETRAINED|N=50 | 0.629 | 5.95e-06 | 672 |
| IDLIKE_DEVELOPMENT|V2_RETRAINED|N=100 | 0.619 | 5.86e-06 | 672 |
| BACKGROUND_OOD_DEVELOPMENT|V2_RETRAINED|N=5 | 0.411 | 2.63e-05 | 672 |
| BACKGROUND_OOD_DEVELOPMENT|V2_RETRAINED|N=10 | 0.460 | 2.48e-05 | 672 |
| BACKGROUND_OOD_DEVELOPMENT|V2_RETRAINED|N=20 | 0.519 | 2.57e-05 | 672 |
| BACKGROUND_OOD_DEVELOPMENT|V2_RETRAINED|N=50 | 0.610 | 2.95e-05 | 672 |
| BACKGROUND_OOD_DEVELOPMENT|V2_RETRAINED|N=100 | 0.577 | 2.86e-05 | 672 |
| development_background_ood|V2_ORIGINAL|N=5 | 0.443 | 1.29e-05 | 672 |
| development_background_ood|V2_ORIGINAL|N=10 | 0.449 | 1.29e-05 | 672 |
| development_background_ood|V2_ORIGINAL|N=20 | 0.527 | 1.22e-05 | 672 |
| development_background_ood|V2_ORIGINAL|N=50 | 0.524 | 1.21e-05 | 672 |
| development_background_ood|V2_ORIGINAL|N=100 | 0.515 | 1.22e-05 | 672 |
| development_background_ood|M4|N=5 | 0.321 | 3.29e-05 | 336 |
| development_background_ood|M4|N=10 | 0.396 | 3.14e-05 | 336 |
| development_background_ood|M4|N=20 | 0.336 | 2.98e-05 | 336 |
| development_background_ood|M4|N=50 | 0.310 | 2.9e-05 | 336 |
| development_background_ood|M4|N=100 | 0.310 | 2.85e-05 | 336 |
| development_idlike|V2_ORIGINAL|N=5 | 0.408 | 7.07e-06 | 672 |
| development_idlike|V2_ORIGINAL|N=10 | 0.411 | 6.94e-06 | 672 |
| development_idlike|V2_ORIGINAL|N=20 | 0.432 | 6.57e-06 | 672 |
| development_idlike|V2_ORIGINAL|N=50 | 0.432 | 6.39e-06 | 672 |
| development_idlike|V2_ORIGINAL|N=100 | 0.429 | 6.41e-06 | 672 |
| development_idlike|M4|N=5 | 0.372 | 1.03e-05 | 336 |
| development_idlike|M4|N=10 | 0.375 | 9.54e-06 | 336 |
| development_idlike|M4|N=20 | 0.405 | 9.8e-06 | 336 |
| development_idlike|M4|N=50 | 0.336 | 8.96e-06 | 336 |
| development_idlike|M4|N=100 | 0.327 | 8.92e-06 | 336 |

## Diagnosis

At N=100 the fresh ID-like accuracy rose from 0.429 for the original banks to 0.619 after full-coverage retraining. On background-OOD it changed from 0.515 to 0.577. Each retrained bank received 1,920 updates versus 160 in the old loop, so this experiment tests better coverage together with a larger training budget and cannot attribute the gain to coverage alone. The extreme negative U4 effect skill (-214.5) and negative U3 skill (-1.71) show that additional training did not resolve all operator failures. Ordered (U1,U2) and (U2,U1) classes score 1.000 and 1.000. This is development evidence only, not acceptance or test-set confirmation.
