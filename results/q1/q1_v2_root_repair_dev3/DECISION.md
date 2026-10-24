# Q1 V2 comprehensive repair decision

Selected basis widths: `[6, 4, 2]`. This is a fresh development engineering study; original Q1 thresholds and test splits are unchanged.

## Frozen-basis projection and selected repair

| Effect | Direct paired projection at 512 / 1024 probes, ID-like | Direct paired projection at 512 / 1024 probes, OOD |
|---|---:|---:|
| U3 | 0.995/0.995 | 0.996/0.996 |
| U4 | 0.594/0.611 | 0.750/0.749 |

## Chosen-model effect skill (N=100; mean [background-bootstrap 95% CI])

| Regime | U1 | U2 | U3 | U4 |
|---|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | 0.979 [0.976, 0.983] | 0.873 [0.851, 0.894] | -0.528 [-0.951, -0.148] | -132.880 [-229.772, -63.600] |
| BACKGROUND_OOD_DEVELOPMENT | 0.970 [0.967, 0.973] | 0.769 [0.738, 0.798] | 0.357 [0.239, 0.474] | -40.034 [-54.973, -26.052] |

## N=100 class accuracy and ordered pairs (background means)

| Regime | Effect-aware | NLL-only | Retrained V2 | Original V2 | M4 | U1→U2 | U2→U1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | 0.595 | 0.557 | 0.626 | 0.423 | 0.336 | 0.906 | 0.906 |
| BACKGROUND_OOD_DEVELOPMENT | 0.649 | 0.637 | 0.610 | 0.506 | 0.301 | 0.906 | 0.969 |

## Diagnosis

Classification: **MIXED, primarily coefficient inference / operator objective; basis ceiling is not ruled out for U4.** Paired frozen-basis U4 projection skill is 0.594/0.611 ID-like and 0.750/0.749 OOD at 512/1024 probes: positive but below the 0.8 engineering trigger. At N=100, learned U4 improves over retrained V2 (−133 vs −312 ID; −40 vs −243 OOD) but remains below zero-effect skill 0; U3 is −0.53 ID and +0.36 OOD. Inner train coefficient audit (seed 4201) shows U4 operator displacement at 0.76× the separately fitted coefficient delta with effect skill −24.9; U3 skill is −1.08. Old-posterior versus cross-fit coefficient relative error is 4.7–6.1%. Full 480/480 episode coverage and 80 balanced effect tasks/code/epoch were achieved, so incomplete coverage alone is insufficient. Class accuracy improves OOD but declines ID versus retrained V2; ordered-pair accuracy remains high.


## Decision

**Weak-label U3/U4 effect gate: FAIL.** Required positive effect skill and improvement over retrained V2 in both development regimes. Measured: IDLIKE_DEVELOPMENT U3=-0.528 vs retrained V2=-1.334; BACKGROUND_OOD_DEVELOPMENT U3=0.357 vs retrained V2=-1.914; IDLIKE_DEVELOPMENT U4=-132.880 vs retrained V2=-312.255; BACKGROUND_OOD_DEVELOPMENT U4=-40.034 vs retrained V2=-242.844.
The paired conditional-mean effect projection and the cross-fitted noisy-observation learner are separate information conditions. Width and checkpoint selection use inner-train data only. This development repair does not establish Q1_PASS.
