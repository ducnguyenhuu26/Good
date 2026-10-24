# Q1 V3 mechanism-structured study decision

Label: **grey-box system identification with a disclosed mechanism prior**, not unsupervised mechanism discovery. Development only; original tests untouched.

## N=100 functional-class accuracy (marginal scorer; per case means)

| Regime | V3 learned/inferred | V3 population mean | V3 fixed-tanh diagnostic | V2 original | V2 retrained | V2 root repair | M4 | Oracle program |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | 0.699 | 0.539 | 0.801 | 0.411 | 0.574 | 0.616 | 0.327 | 0.893 |
| BACKGROUND_OOD_DEVELOPMENT | 0.609 | 0.321 | 0.854 | 0.515 | 0.592 | 0.656 | 0.330 | 0.955 |

## Independent-query effect skill (mean per background; 95% background-bootstrap CI)

| Regime | Code | Inferred old | Population mean |
|---|---|---:|---:|
| IDLIKE_DEVELOPMENT | U1 | 0.973 [0.969,0.977] | 0.960 [0.954,0.965] |
| IDLIKE_DEVELOPMENT | U2 | 0.939 [0.935,0.942] | 0.943 [0.934,0.951] |
| IDLIKE_DEVELOPMENT | U3 | 0.970 [0.964,0.976] | 0.953 [0.942,0.964] |
| IDLIKE_DEVELOPMENT | U4 | -8.376 [-14.996,-3.896] | -9.231 [-17.857,-3.933] |
| BACKGROUND_OOD_DEVELOPMENT | U1 | 0.946 [0.942,0.950] | 0.942 [0.940,0.945] |
| BACKGROUND_OOD_DEVELOPMENT | U2 | 0.862 [0.849,0.875] | 0.883 [0.876,0.891] |
| BACKGROUND_OOD_DEVELOPMENT | U3 | 0.983 [0.980,0.985] | 0.835 [0.823,0.848] |
| BACKGROUND_OOD_DEVELOPMENT | U4 | -5.242 [-7.634,-3.034] | -3.743 [-5.400,-2.186] |

## Marginal vs point and order check

| Regime | Learned marginal N=100 | Learned point N=100 | Fixed-tanh marginal / point | Oracle program |
|---|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | 0.699 | 0.693 | 0.801 / 0.804 | 0.893 |
| BACKGROUND_OOD_DEVELOPMENT | 0.609 | 0.601 | 0.854 / 0.830 | 0.955 |

## Orders and diagnosis

N=100 ordered-pair class accuracy: ID-like U1→U2 1.000, U2→U1 1.000; OOD 1.000/1.000.
Privileged 1024-fit/1024-query mechanism capacity is strong for U3/U4 (learned odd U3/U4: ID 1.000/1.000; OOD 1.000/1.000). U4 inferred-old skill is negative in both regimes and below the population mean; classify as `PRIOR_SHORTCUT`/operator-transfer failure. Fixed-tanh U4 is high, but its population-mean effect is also high, so it does not establish dynamics-specific transfer. Marginal and point class scores are close; marginal is +0.006 ID / +0.007 OOD at N=100. Oracle program-ranking gaps vs learned are 0.193 ID / 0.347 OOD.
**Gate verdict: FAIL**. Development gates: capacity=True, positive U3/U4=False, beats prior-only U3/U4=False, pair order threshold=True, beats retrained V2 class accuracy in both regimes=True. No Q1_PASS and no original confirmatory run.
