# CLSM-R2 Gate A diagnostic protocol

**Protocol freeze:** commit in `codex/clsm-gate-a` before evaluating composition diagnostics.
**Question:** does the unchanged R2 force-conditioned `ExecutableLaw`, fitted on nominal and singleton configurations, predict held-out mixed mechanisms better than a condition-aware unstructured MLP and pooled MLP?

## Design

- Environment: HalfCheetah-v5; mechanism bits follow the R2 contract: code 1=first motor group, 2=second motor group, 4=damping. Composition codes are 3 (`011`), 5 (`101`), 6 (`110`), 7 (`111`). Physical multipliers are sampled independently per world from motor `[0.50,0.85]` and damping `[1.30,2.00]`.
- Singleton training: reuse the registered 4,000 public transitions / 40 training worlds in the R2 pilot and its evaluator-only train labels. The model-training program is isolated in this diagnostic script; the public loader still sees only public columns. Singleton validation uses the registered 1,000-transition / 10-world validation split. Each structured decoder starts from the nominal-reference law checkpoint.
- Mixed positive-control training: four thousand new transitions from codes 3/5/6/7, balanced at 1,000 per code and 20 independent worlds per code. Composition validation has 400 transitions per code from 8 fresh worlds. These data only train/select `oracle_G_z_xi_mixed_seen`.
- Held-out diagnostic trajectory split: 2,000 transitions per composition from 40 independent worlds/code (50 transitions/world). It is not opened for training or checkpoint selection. A separate 8-world/code confirmatory set is generated and retained without scoring in this Gate A run.
- Matched probes: a frozen bank of 128 state/action snapshots is collected in independent nominal worlds. For each composition/world, the same snapshots and actions are replayed under that world's fixed physical parameters. Evaluator simulator state and oracle labels remain in private diagnostic artifacts.
- Training seeds: 4100, 4101, 4102 for every model. Every model gets the same 4,000 changing-law transitions for its declared condition and the same 4,000 nominal replay transitions. Validation only selects epoch. The four models trained on singleton regimes use singleton validation; the mixed-seen positive control uses mixed validation.
- Models: oracle-G structured singleton, train-selected-G structured singleton, oracle-G structured mixed-seen positive control, oracle-conditioned unstructured MLP, pooled MLP. All use two SiLU hidden layers and Gaussian diagonal output. MLP widths are chosen before training to match the R2 parameter count as closely as integer widths permit. Oracle diagnostic checkpoints are tagged `role=oracle_diagnostic` and rejected by deployable CLSM loaders.
- Normalization: fixed from the registered nominal-reference fit; all models and splits use the same output mean/scale. The vector law signature is the concatenation of normalized predicted next-observation deltas over the same 128 probes.

## Metrics and uncertainty

For each code and independent world, report normalized predictive NLL/dimension, normalized delta MSE, normalized probe-vector signature RMSE, plus per-dimension residuals by train-defined joint-velocity tertile and evaluator contact/no-contact. Compare singleton structured vs oracle-conditioned MLP, singleton structured vs mixed-seen structured, learned-G vs oracle-G structured, and each conditional model vs pooled MLP. The 95% paired bootstrap resamples independent worlds within code (10,000 draws); never resample transitions as independent observations. Also report the per-world frequency that singleton structured MSE exceeds twice the oracle-conditioned MLP MSE; this is a descriptive severe-deterioration convention, not a PASS threshold.

The fitted-decoder Fisher diagnostic computes `J^T Sigma^-1 J` with autodiff on the same probes for each mixed code, includes all off-diagonal entries, and embeds inactive coordinates as zero in a 3×3 matrix. Report per-world and aggregate matrices, eigenvalues, minimum singular value, rank conditional on the fitted decoder, condition number, and sensitivity correlations. This is not a claim that the global support or the simulator is identifiable.

## Predeclared decision rule

- **FAIL—representation:** the oracle-G mixed-seen structured positive control is clearly worse than the oracle-conditioned unstructured baseline on normalized MSE/signature error with paired-world 95% CI excluding zero, despite valid matched coverage and finite/replayable simulator observations.
- **FAIL—zero-shot composition:** the mixed-seen structured control fits the mixed worlds, while singleton-trained structured is clearly worse than both that control and the oracle-conditioned MLP on the same held-out codes with paired-world 95% CIs excluding zero. Report this as failure of singleton-only composition generalization, not necessarily failure of representational capacity.
- **FAIL—support selection:** oracle-G singleton training succeeds but learned-G singleton training is clearly worse on the same compositions with paired-world 95% CI excluding zero.
- **PASS for the tested simulator family:** the mixed-seen positive control functions; singleton-trained structured is not detectably worse than the oracle-conditioned MLP on macro MSE and signature error, no individual code has a paired 95% CI showing deterioration, the learned-G comparison does not reveal a support penalty, NLL/calibration are interpretable, and the full active-set Fisher matrices are reported. This supports only zero-shot transfer on the declared HalfCheetah family and probe domain.
- **INCONCLUSIVE:** uncertainty overlaps both meaningful advantage and deterioration, matched probe coverage/replay is invalid, or training/validation does not establish that the models fit their declared regimes. The gate will not be upgraded based on point estimates alone.

No numerical NLL cutoff is selected from held-out results. All world IDs, seeds, sizes, contrasts, bootstrap procedure, model capacity, and decision logic are frozen in this file before creating or scoring the diagnostic composition worlds.
