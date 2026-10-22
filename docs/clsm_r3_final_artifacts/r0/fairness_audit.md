# CLSM-R3 R0 fairness and physical sufficiency audit

**Status: COMPLETE_WITH_LIMITATIONS.** This completes R0 on the frozen Gate-A development data. It does not decide R1, and the reserved confirmatory split was not opened.

## Matched-data baseline correction

The original Gate-A mixed-seen structured decoder used mixed train/validation, while its `oracle_conditioned_unstructured_MLP` comparator used singleton train/validation. That contrast was not a matched-data capacity comparison. I trained a new oracle-conditioned MLP with the same 4,000 mixed transitions, 4,000 nominal replay transitions, three seeds `(4100, 4101, 4102)`, AdamW settings, 100-epoch ceiling, and mixed validation used by the mixed-seen R2 structured model. Selection was validation-only. The existing singleton R2/MLP and mixed R2 checkpoints were hash-verified and reused. The mixed MLP has 76,187 trainable parameters; the Gate-A structured reference has 76,479 (0.38% difference).

The code-`101` matched-probe deficit **persists** against the matched mixed-seen MLP on previously scored development worlds: structured-minus-MLP normalized MSE `0.03827` (paired-world bootstrap 95% CI `[0.030276228176177894, 0.04643360268599191]`); signature error `0.05565` (CI `[0.044549628925916185, 0.0669377017925192]`). The old contrast's fairness limitation is therefore narrowed, but the new comparison remains development evidence and cannot establish the final R1 claim. Full within-regime, per-code results are in `four_way_contrasts.csv` and `four_way_world_scores.csv`.

## Full-step generalized-force equivalence

Across `128/128` action-bounded pairs at identical restored nominal MuJoCo integration states, the motor-only (`code 001`, motor multiplier 0.65) and damping-only (`code 100`, damping multiplier 1.65) laws had equal initial `qfrc_actuator + qfrc_passive` within relative L2 tolerance `1e-07`. Maximum observed relative force error was `1.14e-16`; deterministic replay error was `0`. Despite this initial-force match, their full-frame-skip next observations differed by mean normalized RMSE `0.334` (bootstrap 95% CI `[0.3178674221864281, 0.3503338248629029]`). The scalar analytic fixture gives a next-velocity difference `0.004528` under equal initial force.

This is evidence that initial resultant generalized force is not a sufficient statistic for the tested finite MuJoCo step. It supports evaluating the single registered intervention-resolved decoder. It does not prove that this physical compression was the only cause of the old code-`101` model error.

## Joint input support and residuals

`joint_input_coverage.csv` reports per-world R2 `(normalized observation, effective initial force)` support and descriptive R3 `(observation, action, nominal force, eta, per-mechanism delta-force)` support, stratified by code, nominal contact, and train-defined velocity tertile. `residual_vs_distance_curve.csv` and `residual_vs_distance_by_world.csv` report singleton R2 error by nearest-neighbor distance with world-level uncertainty. Cutoffs are train-derived descriptive references only. For code `101` matched probes, about 65.9% were within the singleton R2 support cutoff and 10.8% within the mixed-train R2 cutoff; the corresponding trajectory fractions were 99.8% and 92.9%. This documents the matched-state distribution shift alongside the baseline comparison.

## Integrity and limits

- Oracle labels and full MuJoCo snapshots were used only in this evaluator/diagnostic code; they are not inputs to deployable training.
- The original confirmatory reserve was not opened.
- Old Gate-A evaluation worlds were reused only to audit the comparator confound; R1 must use newly generated, frozen evaluation worlds.
- The force-pair audit is bounded by feasible action solutions and coverage of the old development probe bank.

Checkpoint hashes, training logs, per-world scores, coverage strata, and the JSON decision are saved beside this report.
