# Q1 v2 Bayesian function-space operators

## Decision

**`OPERATOR_TRANSFER_FAILURE` — development acceptance failed. This is not `Q1_PASS`.** The learned system uses old support and its full candidate scorer outperforms legacy M4 on the paired low-excitation development cases. However, U4 effect prediction is far below the zero-effect baseline on fresh queries, remains poor under informative exploration, and U1/U2 order sensitivity is strongly asymmetric. The Bayesian marginal score does not improve on the same model with `S_old=0`. Stop this v2 acceptance attempt: no retuning, test-set evaluation, or Q2.

## Architecture and information boundary

`q1_v2/basis.py` learns an additive K=12 transition-function basis from action, velocity, and position partitions. `q1_v2/bayes.py` computes a Gaussian posterior over task coefficients with Cholesky solves and provides direct and Woodbury marginal likelihoods. `q1_v2/operators.py` learns four same-parameterization block-diagonal affine coefficient transformations; `q1_v2/inference.py` composes them exactly and scores the 21 candidate programs using old support and adaptation transitions.

The learned modules and training API do not import simulator interventions, hidden dynamics parameters, physical ranges, or response functions. The pipeline imports simulator code only to generate noisy observed transition episodes and to evaluate offline oracles. Training targets are observed `(s,a,delta_v)` data with opaque intervention tags; there are no paired simulator conditional-mean training labels. Assessment queries are independent and only used after ranking. All test splits remain untouched.

The composition algebra and Gaussian covariance propagation are exact conditional on the learned finite basis and affine approximation. They do not guarantee that the basis captures the transition family or that trained operators recover physical interventions.

## Frozen runs

The primary run used 40 fresh train-range backgrounds for optimization, 8 distinct train-range backgrounds for inner selection, and 16 new backgrounds in each of ID-like and background-OOD development regions. Seeds were collision-checked against saved split/protocol JSON files. Two optimizer seeds trained the support-conditioned and context-ablated operator banks. The length penalty was selected from `{0, 0.01, 0.1}` using only the 8 inner backgrounds and permitted compositions; all candidates tied at 0.5288 class accuracy and 0.0 was frozen.

The primary protocol used `dt=0.05`, `sigma=0.0005`, 96 old-support transitions, one uninterrupted 100-step adaptation trajectory, independent assessment queries, all 21 programs, and budgets `N={5,10,20,50,100}`. Every candidate received the same old support and adaptation observations. Fresh development cases were paired with the available legacy M2/M3/M4 checkpoints.

A separate informative-exploration protocol selected a persistent diagonal action schedule using only state-action coverage, velocity spread, and bounded-state checks on 8 training backgrounds. It used the same schedule for every compared program and new development background. Legacy M3/M4 were also evaluated on these same informative trajectories. This is secondary development evidence, not confirmation.

## Main results at N=100

| Protocol and region | V2 marginal | V2 point estimate | V2 single-only | V2 context-ablated | Legacy M4 | Paired V2 minus M4 |
|---|---:|---:|---:|---:|---:|---:|
| Low-excitation, ID-like | 0.449 [0.415, 0.484] | 0.440 | 0.155 | 0.077 | 0.330 [0.293, 0.368] | +0.119 [0.090, 0.148] |
| Low-excitation, background-OOD | 0.512 [0.501, 0.523] | 0.542 | 0.190 | 0.074 | 0.301 [0.278, 0.323] | +0.211 [0.185, 0.237] |
| Informative, ID-like | 0.548 | — | — | — | 0.327 | +0.220 [0.174, 0.266] |
| Informative, background-OOD | 0.655 | — | — | — | 0.292 | +0.363 [0.326, 0.400] |

Values are functional-class accuracy. Confidence intervals and paired differences use background as the unit; optimizer seeds and programs are averaged within background. The informative table reports the main v2 scorer and paired legacy M4 only; its point-estimate and context controls were not run.

Marginal uncertainty did not help: paired marginal-minus-point accuracy was +0.009 [−0.030, +0.047] on ID-like backgrounds and −0.030 [−0.059, −0.001] on background-OOD. The low-excitation context-ablated operator scored only 0.077 and 0.074; paired old-context swaps increased true-program NLL substantially. Thus old support mattered, but covariance propagation did not yield a measured gain.

## Failure diagnosis

- **U4 effect transfer fails.** On independent query probes, support-conditioned U4 effect skill was −121.77 [−184.90, −58.63] for low-excitation ID-like and −47.70 [−63.21, −32.19] for background-OOD. Zero-effect skill is 0. The context-ablated U4 effect was worse still. Under informative exploration, U4 skill remained −139.13 [−229.87, −48.38] and −42.73 [−72.12, −13.33].
- **The low-excitation result is not explained solely by unobservable U4 effects.** A privileged analytic constant-drag calculation using the train-background mean achieved skill 0.983 ID-like and 0.840 OOD on the primary independent probes. On the old-support system-ID reference, estimated-old physics achieved 0.818 / 0.935 class accuracy at N=100, compared with known-old oracle 0.890 / 0.952. V2 reached 0.449 / 0.512. These equation-aware results are offline diagnostics, never learned-model inputs.
- **Ordered compositions remain asymmetric.** For `(U1,U2)` versus `(U2,U1)` at N=100, low-excitation class accuracy was 0.125 versus 1.000 ID-like and 0.000 versus 1.000 OOD. Informative exploration raised the first order to 0.375 / 0.250 while the reverse order was 0.875 / 0.938. The learned action-block commutator is nonzero but small (about 0.0096 in both seeds); that alone does not establish useful order inference.
- **Composition helps overall, but not enough to pass.** On the low-excitation primary run, V2 exceeds the singles-only and legacy M4 controls, especially under OOD. The U4 effect and ordered-pair results still fail the frozen scientific checks.

The best-supported failure class is **operator transfer failure**: the model is measurably conditioned on old support, its synthetic weak-label transfer test passes, and informative exploration improves broad class accuracy, yet the learned U4 operator does not transfer its small velocity-dependent effect. A basis-family limitation remains possible; the unrestricted basis ablation was not configured or run. The prior covariance trace fell from about 9.96 to about `2e-5`, while point-estimate scoring performed as well or better, so the model's learned posterior spread was not a useful ranking signal in this study.

## Verification and limitations

Eight focused v2 tests pass, covering analytic Gaussian posterior recovery, predictive Monte Carlo NLL, direct/Woodbury agreement, covariance and jitter, exact affine algebra, noncommuting matrices, synthetic weak-label operator transfer to an unseen background/composition, basis shapes/gradients, full 21-program scoring, and import boundaries. The v1 direct-effect G checkpoints were not included because they were trained from privileged paired simulator effects, not the v2 weak-label protocol. The informative secondary run lacks the full set of point/no-composition/context-ablated controls and privileged system-ID references. Neither omission is treated as a pass.

Three early partial primary runs are retained and explicitly marked failed under `results/q1/q1_v2_bayesian_operators_dev1_failed*`; none has interpretable evaluation metrics. The accepted development run was trained from scratch after those code defects were fixed.

## Reproduction commands

From the repository root with Python 3.11+ and CPU PyTorch:

```powershell
uv sync --extra q1 --extra test
uv run --extra q1 --extra test python -m pytest -q tests/test_q1_v2_bayesian_operators.py
uv run --extra q1 python scripts/q1_v2_pipeline.py --config configs/q1_v2.yaml --run-dir results/q1/q1_v2_bayesian_operators_dev1
uv run --extra q1 python scripts/q1_v2_legacy_compare.py --run-dir results/q1/q1_v2_bayesian_operators_dev1
uv run --extra q1 python scripts/q1_v2_privileged_oracles.py --run-dir results/q1/q1_v2_bayesian_operators_dev1
uv run --extra q1 python scripts/q1_v2_finalize_report.py --run-dir results/q1/q1_v2_bayesian_operators_dev1
uv run --extra q1 python scripts/q1_v2_informative.py --run-dir results/q1/q1_v2_bayesian_operators_dev1 --output results/q1/q1_v2_informative_dev1
uv run --extra q1 python scripts/q1_v2_informative_legacy.py --run-dir results/q1/q1_v2_bayesian_operators_dev1 --output results/q1/q1_v2_informative_dev1
uv run --extra q1 python scripts/q1_v2_effect_controls.py --run-dir results/q1/q1_v2_bayesian_operators_dev1 --informative results/q1/q1_v2_informative_dev1
uv run --extra q1 python scripts/q1_v2_summarize.py --run-dir results/q1/q1_v2_bayesian_operators_dev1
```

Raw per-background rows, split IDs, RNG seeds, parameter ranges, checkpoint/source hashes, runtimes, and machine-readable reports are saved in `results/q1/q1_v2_bayesian_operators_dev1/` and `results/q1/q1_v2_informative_dev1/`.
