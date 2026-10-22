# CLSM-R3 final-attempt evidence

**Frozen source base:** `2818503bf0fe90aa79e31ecad795b80343424ba5`

**Branch:** `codex/clsm-r3-final`
**Gate order:** R0 completed; R1 evaluated; R2 and later gates halted by the registered R1 failure.

## Final decision

`FINAL_RESEARCH_VERDICT: ABANDON_CLSM_ARCHITECTURE`

The sole registered R3 decoder failed its decisive R1 test on newly generated mixed compositions. This is a locked scientific failure, not an implementation placeholder or an inconclusive run. The revision contract requires stopping after this failure. The known-code numerical reference (R2), age-conditioned posterior filter (R3), generalized EM, C2, and C3 were therefore not run. No claim is made about those unevaluated components.

The evidence rejects this final R3 candidate for the requested mixed-composition and singleton-to-composition claims. It does not establish that every possible transition model is impossible; the registered final-attempt contract does not authorize another architecture or threshold revision in this study.

## R0: comparison fairness and physical sufficiency

R0 audited the prior Gate A comparator confound without changing the earlier result. The old mixed-seen structured decoder had trained on mixed data while the old MLP comparator had trained on singleton data. A new MLP was fitted on the same mixed transitions, nominal replay, seeds, validation split and selection policy as the structured mixed-seen model.

On the previously scored development code-`101` matched probes, the R2 structured model still had higher normalized MSE than the matched mixed MLP by `0.03827` (paired-world 95% CI `[0.03028, 0.04643]`) and higher signature error by `0.05565` (CI `[0.04455, 0.06694]`). These are development results, not R1 confirmation.

The full-step force audit found `128/128` feasible pairs with matched initial generalized force (`qfrc_actuator + qfrc_passive` maximum relative error `1.14e-16`) at the same restored nominal integration state. Their next-observation normalized RMSE was `0.33422` (world bootstrap CI `[0.31787, 0.35033]`); deterministic replay error was zero. This supports preserving intervention-resolved channels in the tested physics, but does not prove that force compression was the sole cause of the earlier decoder deficit.

R0 status is `COMPLETE_WITH_LIMITATIONS`. It did not open the old confirmatory reserve. Published evidence is in [R0 report](clsm_r3_final_artifacts/r0/fairness_audit.md), [R0 decision JSON](clsm_r3_final_artifacts/r0/r0_decision.json), and [four-way per-world contrasts](clsm_r3_final_artifacts/r0/four_way_contrasts.csv).

## R1: registered R3 decoder falsification

### Frozen protocol

- Candidate: intervention-resolved, nominal-anchored decoder. The nominal R2 reference was frozen; the two-layer SiLU residual receives observation, action, nominal force, intervention strengths, and separate mechanism force changes.
- Same train/validation regimes as Gate A; oracle labels were confined to this diagnostic trainer/evaluator.
- Seeds: `4100`, `4101`, `4102`; AdamW `3e-4`, weight decay `1e-4`, batch `256`, up to `100` epochs, patience `12`; checkpoint selection used regime-specific validation standardized Gaussian NLL.
- Residual width `244`, yielding `76,406` trainable parameters versus `76,187` for the oracle-conditioned MLP comparator (difference `0.29%`). All six R3 fits selected epoch `100`; validation NLL per dimension was `0.66` for singleton fits and `0.49–0.50` for mixed-seen fits.
- Fresh held-out evaluation: 40 independent worlds per mixed code `{3,5,6,7}` (160 total, 50 transitions each); separate matched-state bank with 40 parameter worlds per code over 128 fixed snapshots (20,480 outcomes). Data manifests and hashes were frozen before fitting. The old reserved set of 32 worlds (8/code) was opened only after checkpoint freeze and used as supplementary evidence; it did not affect the registered decision.
- The R3 matched-probe noninferiority margin was `5%` of the matched-training MLP's per-code error. Confidence intervals resampled independent worlds after averaging training-seed checkpoint scores within each world; seed-level contrasts are saved separately.

The published [R1 decision JSON](clsm_r3_final_artifacts/r1/r1_decision.json), [R1 report](clsm_r3_final_artifacts/r1/decision.md), [per-world seed scores](clsm_r3_final_artifacts/r1/per_world_seed_scores.csv), [sanitized preregistration manifest](clsm_r3_final_artifacts/r1/preregistration.json), and [checkpoint training metadata](clsm_r3_final_artifacts/r1/training.json) contain the per-world scores, paired intervals, seed sensitivity, calibration summaries, checkpoint hashes, and frozen protocol. The bundle [manifest](clsm_r3_final_artifacts/manifest.json) lists SHA-256 hashes for every published result file. Raw evaluator-only arrays and model checkpoints are excluded from the bundle.

### Locked outcomes

| Criterion | Outcome | Evidence |
|---|---|---|
| Mixed-seen R3 noninferiority on matched probes, every code and both error metrics | **FAIL** | For code `101` (decimal 5), R3 minus matched MLP MSE was `+0.08113`, CI `[+0.07299,+0.08906]`; one-sided upper bound `+0.08783`, versus margin `0.00556`. Signature difference was `+0.10473`, CI `[+0.09460,+0.11442]`; upper `+0.11292`, versus margin `0.01666`. All four mixed codes failed at least one registered contrast; in fact both metrics failed for every code. |
| Singleton-trained R3 noninferiority to singleton-trained MLP on matched probes | **FAIL** | For code 5, MSE difference `+0.11076`, CI `[+0.09775,+0.12352]`, versus margin `0.00581`; signature difference `+0.13507`, CI `[+0.12010,+0.14954]`, versus margin `0.01699`. Both metrics failed for all four codes. |
| Singleton-trained R3 macro MSE improves over singleton MLP | **FAIL** | Paired stratified-world difference `+0.10642`, 95% CI `[+0.09814,+0.11418]`; the required upper bound below zero is not met. |
| No meaningful trajectory deterioration above 5% | **FAIL** | Versus singleton R2, R3-minus-R2 macro MSE was `+0.04478` (baseline `0.03570`); CI for difference minus the 5% margin `[+0.04007,+0.04592]`. Versus singleton MLP the corresponding CI was `[+0.01218,+0.02113]`. |
| Deterministic physical replay | **PASS** | Maximum absolute replay error `0`. |

Mixed-seen variance calibration was also poor: on trajectories, its standardized residual-variance ratio ranged from `0.041` to `0.132`, while nominal marginal 90% coverage ranged from `0.993` to `0.999`; on matched probes the ratio ranged from `0.436` to `0.516`, with coverage `0.956` to `0.964`. These descriptive checks show substantial over-dispersion, especially on trajectories. No post-hoc calibration cutoff was used to change the gate decision.

The result separates two claims. The R3 model had a valid nominal anchor, correct mechanism channels, matched parameter count, and completed fits; nevertheless, its mixed-seen representation lost to the matched MLP on unseen worlds and matched-state probes. Singleton-only R3 also failed to transfer to mixed interventions. Therefore fitting the proposed decoder correctly does not rescue the registered CLSM architecture under the final test.

## Implementation and checks

The new versioned model and strict config are [dynamics_r3.py](../src/rwt/clsm/dynamics_r3.py), [config_r3.py](../src/rwt/clsm/config_r3.py), and [clsm_r3_final.yaml](../experiments/rwt/clsm/configs/clsm_r3_final.yaml). The isolated diagnostic runner is [run_clsm_r3_gate_a.py](../scripts/run_clsm_r3_gate_a.py); R0 is [run_clsm_r3_r0_audit.py](../scripts/run_clsm_r3_r0_audit.py). R2 `ExecutableLaw` and earlier Gate A–C evidence files were not changed.

R3 tests cover intervention bit masking, zero inactive-strength gradients, exact nominal anchoring, distinct mechanism-channel identity under equal resultant force, strict config validation, and rejection of R2 or non-diagnostic checkpoints. The complete CLSM regression suite passed: **76 passed**.

## Reproducibility boundary

The worktree was branched from the pinned evidence commit on `codex/clsm-r3-final`. The registered source datasets and old model artifacts were hash-checked from that checkout; new R1 public/evaluator files and checkpoints have their own SHA-256 entries in the local manifests and training metadata. Exact per-seed training runtimes were not retained after a checkpoint-resume pass, so no per-seed training-time claim is made. The final CPU scoring pass took `17.06 s` wall time and `54.11 s` process CPU time, with a process peak working set of `349,147,136` bytes; amortized per-world CPU/wall timing and process peak RSS are recorded in each score row. The full six-model training and fresh data generation were also run on CPU. No push was performed as part of this R3 revision request.
