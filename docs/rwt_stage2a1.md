# RWT Stage 2A.1 — Structural identifiability

This stage diagnoses why the Stage 2A latent composition did not reliably
generalize to held-out regime C. It does not tune Stage 1.5 policies, implement
Stage 2B/model-based RL, or change upstream NEO.

## Questions and diagnostics

### Ground-truth composition closure

The simulator's deterministic expected transition was checked on matching
state/action probes sampled from both A/B/D training support and held-out C
support. For each point the diagnostic computes

\[
\Delta(s,a)=F_C(s,a)-F_B(s,a)-F_D(s,a)+F_A(s,a),\qquad
e_{rel}=\frac{\|\Delta\|_2}{\|F_B-F_A\|_2+\|F_D-F_A\|_2}.
\]

These calls use ground-truth dynamics only in this diagnostic. The resulting
closure residuals are at double-precision roundoff on both supports, so
additive composition is correctly specified by the environment. This rules out
non-additive environment structure as the explanation for the Stage 2A result.

### Why three observed codes do not identify the fourth

Training sees three regimes mapped to codes 00, 10, and 01, while code 11 is
never observed. The test constructs two bijections from A/B/D to those codes.
Both reconstruct every training observation exactly, yet their code-11
predictions differ (4 versus 2 in the scalar witness). Thus training likelihood
alone cannot determine the semantic anchor of the two latent bits. Optimization
cannot resolve this missing information without additional assumptions or
observations.

### Temporal and structural ablations

`composition_iid` is the Stage 2A per-transition marginal likelihood. The HMM
variant replaces it with the exact forward-algorithm sequence evidence under a
uniform-start persistent-code prior with hazard 0.04. Its filter predicts with
the prior belief before observing a transition, then updates from that
transition's likelihood. No regime IDs or schedule are given to the learner.

The structured linear ablation gives the base theory full normalized
state/action inputs, one primitive velocity-only inputs, and one primitive
action-only inputs. Means are composed additively. Normalization statistics
come from the current training split only. The weakly anchored variant fixes
the first 8 transitions of each training episode in A to reference code 00;
this is reported separately and is not fully unsupervised. The oracle-assignment
diagnostic receives A/B/D regime labels and is an upper-bound check only.

All principal models train on the same A/B/D transition episodes. Validation
also contains only A/B/D. C is absent from training, validation, checkpoint
selection and all ablations' hyperparameter decisions. Evaluation streams use
the fixed seeds in `results/rwt/stage2a1/config.json`. There are three training
replications and six fixed evaluation seeds. Confidence intervals resample
training replication and evaluation seed; paired comparisons match both.

## Final CPU experiment

The finalized run uses episode budgets 1/4/12, three training replications,
28 epochs, hidden width 32 for neural models, and fixed validation/evaluation
seeds. The training manifest, per-episode values, paired bootstrap results,
learning curves, posterior occupancy, primitive diversity and full per-step
traces are in `results/rwt/stage2a1/`. The summary table below is populated from
the final post-normalization run. MSE and rollout MSE average over four state
coordinates; rollout MSE uses a five-step autoregressive rollout and holds the
causal theory belief fixed within each rollout.

Reproduce the CPU sweep with `python experiments/rwt/run_stage2a1.py --output-dir results/rwt/stage2a1 --budgets 1,4,12 --replications 3 --epochs 28 --hidden-dim 32 --batch-size 256 --device cpu`. At the 12-episode budget, held-out C results are mean ± SD with 95% two-way bootstrap CI (training replication × fixed evaluation seed):

| Model | C one-step MSE | C one-step NLL | C 5-step rollout MSE |
|---|---:|---:|---:|
| composition_iid | 0.080 ± 0.030 [0.060, 0.101] | 6.201 ± 3.677 [3.539, 8.746] | 1.306 ± 0.625 [0.954, 1.730] |
| composition_hmm | 0.132 ± 0.063 [0.090, 0.183] | 21.267 ± 12.898 [12.911, 32.397] | 1.822 ± 1.038 [1.183, 2.665] |
| monolithic | 0.076 ± 0.021 [0.062, 0.089] | 12.081 ± 4.866 [8.777, 15.309] | 1.160 ± 0.391 [0.884, 1.432] |
| independent_experts | 0.107 ± 0.032 [0.086, 0.130] | 11.600 ± 4.399 [8.973, 14.610] | 1.720 ± 0.590 [1.272, 2.168] |
| linear_structured_iid | 0.173 ± 0.057 [0.129, 0.231] | 12.582 ± 7.774 [6.066, 21.425] | 31.773 ± 54.715 [2.008, 90.883] |
| linear_structured_hmm | 0.576 ± 0.250 [0.323, 0.831] | 100.039 ± 46.972 [57.018, 144.107] | 377.676 ± 650.699 [6.533, 1069.390] |
| weak_calibration_hmm | 0.638 ± 0.461 [0.404, 1.090] | 114.324 ± 108.955 [62.201, 221.422] | 202.373 ± 465.823 [3.659, 686.544] |
| oracle_assignment | 0.044 ± 0.025 [0.029, 0.065] | 2.714 ± 4.759 [-0.118, 6.810] | 0.859 ± 0.441 [0.568, 1.246] |
| stage2a_reference | 0.073 ± 0.033 [0.049, 0.097] | 5.298 ± 5.016 [1.879, 9.122] | 1.230 ± 0.571 [0.817, 1.633] |

Per-regime one-step MSE at the same budget:

| Model | A | B | C (held out) | D |
|---|---:|---:|---:|---:|
| composition_iid | 0.0236 ± 0.0121 | 0.0304 ± 0.0175 | 0.0804 ± 0.0300 | 0.0757 ± 0.0248 |
| composition_hmm | 0.0716 ± 0.0161 | 0.0620 ± 0.0229 | 0.1319 ± 0.0629 | 0.1602 ± 0.0594 |
| monolithic | 0.0110 ± 0.0039 | 0.0217 ± 0.0069 | 0.0757 ± 0.0206 | 0.0815 ± 0.0211 |
| independent_experts | 0.0409 ± 0.0165 | 0.0342 ± 0.0099 | 0.1073 ± 0.0321 | 0.1103 ± 0.0258 |
| linear_structured_iid | 0.1494 ± 0.0488 | 0.1087 ± 0.0352 | 0.1735 ± 0.0575 | 0.1995 ± 0.0954 |
| linear_structured_hmm | 0.5907 ± 0.2077 | 0.5425 ± 0.2371 | 0.5761 ± 0.2501 | 0.6941 ± 0.2819 |
| weak_calibration_hmm | 0.6074 ± 0.2511 | 0.5147 ± 0.3031 | 0.6377 ± 0.4609 | 0.6663 ± 0.3645 |
| oracle_assignment | 0.0208 ± 0.0127 | 0.0243 ± 0.0123 | 0.0439 ± 0.0245 | 0.0318 ± 0.0114 |
| stage2a_reference | 0.0115 ± 0.0050 | 0.0194 ± 0.0097 | 0.0729 ± 0.0328 | 0.0596 ± 0.0197 |

Held-out C learning curve (one-step MSE, mean ± SD across the 18 matched
training-replication/evaluation-seed pairs):

| Model | 1 episode | 4 episodes | 12 episodes |
|---|---:|---:|---:|
| composition_iid | 0.163 ± 0.096 | 0.156 ± 0.102 | 0.080 ± 0.030 |
| composition_hmm | 0.168 ± 0.084 | 0.168 ± 0.106 | 0.132 ± 0.063 |
| monolithic | 0.152 ± 0.062 | 0.172 ± 0.086 | 0.076 ± 0.021 |
| independent_experts | 0.172 ± 0.056 | 0.180 ± 0.202 | 0.107 ± 0.032 |
| linear_structured_iid | 1.169 ± 0.908 | 0.435 ± 0.258 | 0.173 ± 0.057 |
| linear_structured_hmm | 1.878 ± 2.113 | 0.852 ± 0.596 | 0.576 ± 0.250 |
| weak_calibration_hmm | 2.104 ± 2.389 | 0.643 ± 0.330 | 0.638 ± 0.461 |
| oracle_assignment | 0.121 ± 0.032 | 0.134 ± 0.070 | 0.044 ± 0.025 |

All budgets and per-regime MSE/NLL/rollout values with confidence intervals are in `metrics_summary_ci.csv`; paired seed-matched contrasts are in `paired_comparisons.csv`. For C MSE, `composition_iid - monolithic` is +0.0048 [−0.0133, 0.0263], so the difference is not resolved by this experiment. `composition_iid - independent_experts` is −0.0268 [−0.0539, −0.0077], while `composition_iid - composition_hmm` is −0.0515 [−0.0938, −0.0165]. Negative values mean composition_iid has lower error. The Stage 2A reference C MSE is 0.0729 ± 0.0328 [0.0488, 0.0972]. Oracle assignment reaches 0.0439 ± 0.0245 [0.0286, 0.0649], but uses privileged labels.

At 12 episodes, closure has mean/max absolute error `7.90e-16 / 7.33e-15` on training support and `1.07e-15 / 7.16e-15` on held-out C support; relative means/maxima are `1.52e-15 / 6.71e-14` and `1.94e-15 / 2.26e-14`. The synthetic assignments both have zero A/B/D reconstruction error and predict 4 versus 2 on code 11.

The latent posterior does not collapse to one code: for composition_iid at 12 episodes it averages 3.3 active codes (>5% occupancy) on A/B/D and 4 on C, with normalized entropy 0.72–0.92. However, occupancy is not aligned into a stable semantic mapping, consistent with the algebraic ambiguity. Learned primitive mean norms are nonzero (0.338 and 0.419 for composition_iid; 0.700 and 0.657 for composition_hmm), so the observed failure is not a dead-primitive collapse. The HMM has significantly worse paired C MSE than iid, and weak calibration also fails to help (0.638 C MSE). The structured linear prior is too restrictive for these nonlinear dynamics (0.173 C MSE even for iid, and poor A/B/D reconstruction).

**Conclusion:** additive environment structure is verified, but unsupervised latent theory composition still does not reliably identify the held-out combination. The learned composition beats independent experts on C but is statistically indistinguishable from monolithic dynamics, and its five-step C rollout error is higher than monolithic. The Stage 2B gate is not met. Temporal persistence, the tested structural linear prior, and this weak calibration setting did not resolve the identifiability problem.

Ground-truth closure summary is in `closure_summary.csv`; assignment witnesses
are in `assignment_ambiguity.csv`. `stage2a1_data_scaling.png` plots A/B/D-only
validation learning curves and held-out C error versus training budget.

## Interpretation and limits

The tests establish that the environment admits exact additive composition and
that the three-code observation design is algebraically non-identifying. The
empirical comparison determines whether temporal persistence, structural
constraints, or weak calibration help under this small-data CPU protocol. The
unlabeled composition is only supported as a better zero-shot model if paired
held-out C results beat both monolithic and independent-expert baselines with
uncertainty accounted for. Privileged oracle assignment must not be described
as unsupervised learning.

No Stage 2B recommendation follows unless this criterion is met or a concrete
specification error in the earlier comparison is found and corrected.
