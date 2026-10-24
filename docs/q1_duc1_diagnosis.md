# Q1 `duc1` repair diagnosis

## Scope and provenance

This is a validation-only development diagnosis of the saved `cpu_confirmatory_impl` M2, M3-before-composition, and M4-after-composition checkpoints. It used all 16 existing validation backgrounds, the declared 20 single/permitted/held-out programs, budgets `N={5,10,20,50,100}`, and independent stage-pair query trajectories. The ordered-pair identities had already been inspected, so these results are development data. No test backgrounds or training were used in P0/P1.

Reproduce the no-training checks from the repository root:

```bash
uv run --extra q1 python scripts/q1_repair_audit.py --config configs/q1_repair_diagnostic.yaml --run-dir results/q1/repair_v1_audit_full --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt --m4-checkpoint results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt --device cpu
uv run --extra q1 python scripts/q1_diagnose_feasibility.py --config configs/q1_repair_diagnostic.yaml --run-dir results/q1/repair_v1_diagnosis --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt --m4-checkpoint results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt --device cpu
```

The original CPU pilot directories remain unchanged. Checkpoint hashes and measured runtime are recorded in each run's `provenance.json`.

## P0 corrections verified

- M3 functional distances now normalize raw common probes with the train-fitted normalizer, then denormalize executor outputs. The audit reports physical delta-v RMSE on common probes and naturally visited validation state-action pairs, using all 64 validation single-intervention tasks.
- Physical-oracle rankings use the same uninterrupted `stage_pair` adaptation transitions as the scorer. Identity and simulator-defined functional-class accuracy are separate; independent-query simulator MSE is labeled as privileged.
- Opaque-code confusion matrices and physical-program confusion matrices are separately labeled. The persisted group-to-code map is checked for M3/M4 agreement and bijectivity.
- Noncommuting response RMSE is in physical delta-v units and is `null` for regimes without those cases. The simulator's order separation is saved as a reference.
- New M2 training evaluates every fixed validation query transition and saves the best validation-MSE checkpoint, while retaining final-epoch metrics. This change was unit-tested; the existing M2 checkpoint and old metrics were not regenerated.
- New M4 reports retain the original gate and threshold, explicitly label its single best aggregated cell, and add paired differences, relative gains, and signs by held-out regime and N. Historical M4 artifacts were not rewritten.

The corrected physical oracle reaches functional-class accuracy `0.844` at `N=5` and `0.961` at `N=100` over 128 validation stage-pair cases per budget. Raw identity accuracy is lower (`0.563` to `0.617`), as expected with equivalent programs. In the full fixed-probe audit, the closest M3 codes are U3/U4 at `1.34e-4` physical delta-v RMSE; these outputs are distinct, not collapsed.

## P1 findings

On fixed common validation probes, M3 effect skill versus a zero-effect prediction is `+0.847` for U1, `+0.562` for U2, `-0.436` for U3, and `-8.177` for U4. The U4 effect signal energy is only `7.43e-7`, so its very negative skill is sensitive to a small denominator; the per-background values are retained in `per_code_metrics.csv`.

The frozen-M2 persistence baseline is the M2 executor prediction from the old context on the same probe. M3 after-dynamics MSE versus that baseline is:

| Intervention | M3 after MSE | Frozen-M2 persistence MSE | Effect skill |
|---|---:|---:|---:|
| U1 | `4.56e-5` | `1.91e-4` | `+0.847` |
| U2 | `4.18e-5` | `6.33e-5` | `+0.562` |
| U3 | `8.12e-5` | `7.66e-5` | `-0.436` |
| U4 | `4.72e-5` | `4.27e-5` | `-8.177` |

M4 fine-tuning lowered common-probe single after-MSE for all four codes, so single-operator damage from M4 is not supported. It lowered true-code ordered-pair query MSE by `48.2%` on average versus M3. However, at `N=100`, M4 class accuracy over the validation development held-out pairs was `0.234`; median true-program versus best wrong-class score margin was `-0.1007`, and only `30.4%` of M4 program/budget cases had a positive margin. For held-out pairs, the searched code's query MSE (`3.20e-5`) was below the true-code candidate (`3.69e-5`). These results indicate model-ranking misspecification along with weak effect fidelity; they do not support a search-only fix.

Encoder-only after-context and direct-context optimization predict validation laws better than the old context. Direct-context fitting at 40 steps also beats M3 true-code query MSE for U3/U4. Representation is not proven fully solved, but a total M2 representation failure is not the first supported explanation.

**Selected first repair: Branch O, paired functional-effect training for M3.** This is a targeted development hypothesis, not a validated repair. Branch C is deferred because M4 improves single and composition prediction metrics; Branch S is not selected because the true-code candidate is often less predictive than the searched code. Branch R remains a limitation to monitor.

## Decision

Do not start M4 repair or confirmatory evaluation until a per-code M3 validation gate passes. The following paired-effect ablation was implemented and run once on training backgrounds with validation-only beta selection; all beta variants failed U4. The negative result and exact metrics are in [q1_duc1_repair_results.md](q1_duc1_repair_results.md) and `results/q1/repair_v1_m3_effect_ablation/`.
