# Q1 v2 development gate

**Decision: OPERATOR_TRANSFER_FAILURE. Acceptance failed; not Q1_PASS.** Stop this v2 acceptance attempt. Do not retune from these development results, inspect test splits, or proceed to Q2.

## Structural gates

- **PASS:** model/training import boundary; no simulator interventions, hidden physical parameters, physical ranges, or simulator response targets in the learned modules.
- **PASS:** Gaussian posterior algebra, finite/symmetric PSD covariance, noise monotonicity, jitter reporting, direct/Woodbury agreement, affine identity/associativity/repetition, noncommuting operator support, and full 21-program enumeration. Focused v2 suite: **8 passed**.
- **PASS, with failure evidence:** predictions depend on old support. On primary N=100 single-code cases, swapping old support to another background sharply increased candidate NLL; a separately trained context-ablated bank reached only 0.077 ID-like / 0.074 OOD functional-class accuracy.
- **PASS:** split and seed collision audit completed before training; original test splits were unused. Assessment queries were used after inference only.

## Frozen scientific results at N=100

| Primary low-excitation metric | ID-like | Background-OOD |
|---|---:|---:|
| V2 marginal functional-class accuracy | 0.449 [0.415, 0.484] | 0.512 [0.501, 0.523] |
| Legacy M4 accuracy on same cases | 0.330 [0.293, 0.368] | 0.301 [0.278, 0.323] |
| Paired V2 minus M4 | +0.119 [0.090, 0.148] | +0.211 [0.185, 0.237] |
| V2 point-estimate accuracy | 0.440 | 0.542 |
| V2 singles-only accuracy | 0.155 | 0.190 |
| V2 context-ablated operator accuracy | 0.077 | 0.074 |
| Paired marginal minus point estimate | +0.009 [−0.030, +0.047] | −0.030 [−0.059, −0.001] |

Intervals cluster at background level. V2 composition beats legacy M4 and the singles-only control in the primary protocol. Covariance propagation itself has no measured gain.

**The U4 gate fails:** independent-query effect skill is −121.77 [−184.90, −58.63] ID-like and −47.70 [−63.21, −32.19] OOD; zero-effect skill is 0. The privileged constant-drag analytic diagnostic is 0.983 / 0.840. Under the separately selected informative schedule, V2 class accuracy rises to 0.548 / 0.655 and beats M4 by +0.220 [+0.174, +0.266] / +0.363 [+0.326, +0.400], but U4 effect skill remains −139.13 [−229.87, −48.38] / −42.73 [−72.12, −13.33].

**Ordered-pair sensitivity is asymmetric:** `(U1,U2)` versus `(U2,U1)` class accuracy is 0.125 / 1.000 ID-like and 0.000 / 1.000 OOD on primary exploration; informative exploration raises the first order to 0.375 / 0.250 and the reverse to 0.875 / 0.938.

Privileged estimated-old physical ID reaches 0.818 / 0.935 at N=100 and known-old reaches 0.890 / 0.952, compared with learned V2 at 0.449 / 0.512. These are offline references only. The analytical constant-drag result and informative schedule suggest insufficient excitation alone does not explain the learned U4 failure.

## Classification and limits

The strongest supported classification is **operator transfer failure**: old support affects predictions, synthetic weak-label operators transfer on an unseen background/composition, and informative exploration improves overall program ranking, yet the real learned U4 effect fails even against zero effect. Basis-family inadequacy remains possible because the unrestricted basis ablation was not configured. The informative protocol did not include point-estimate, singles-only, context-ablated, or privileged-oracle controls; it is secondary development evidence only. The v1 direct-effect G was excluded because its training used privileged paired simulator effects.

Raw rows, split manifests, checkpoint/source hashes, environment details, failed attempts, and reproduction commands are in this result directory and `docs/q1_v2_architecture_and_results.md`.
