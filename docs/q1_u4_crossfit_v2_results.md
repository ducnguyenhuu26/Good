# Q1 U4 cross-fit diagnostic v2

This development-only run separates effect fidelity from absolute after-dynamics error and fits one context displacement on fit labels before applying it to independent query inputs. M2 and M3 checkpoints stayed frozen. The runner evaluated U1/U3/U4 on the original 16 validation backgrounds with fit counts 128 and the predeclared 512 extension; query size remained 256. No test/OOD split or model training was used.

## Result

The overall classification is **INCONCLUSIVE**. At fit 128, U4 linear and nonlinear cross-fit effects showed positive but modest mean skill (0.190 and 0.194, respectively; background-clustered 95% intervals 0.126–0.255 and 0.131–0.258). The beta=0.25 M3 operator had mean U4 effect skill -3.600 (95% interval -5.736 to -1.463). This is a U4 operator-learning bottleneck candidate on inspected development data, not a final acceptance result. The controls differ: U1's nonlinear interval crosses zero, while U3 has positive linear, nonlinear, and M3 intervals. No minimum acceptable effect skill was preregistered, so positive skill intervals alone do not establish adequate accuracy.

The U4 query-refit upper-bound skill was 0.556, but it fits displacement using query labels and is explicitly leakage/nontransfer. It is excluded from transfer conclusions. At fit 512 the U4 linear/nonlinear cross-fit skills were approximately 0.319/0.324; this extension does not change the INCONCLUSIVE classification.

The virtual perfect-effect gain in absolute after-MSE averaged 1.98% across backgrounds (mean of per-background ratios; range 0.61%–4.25%), below the historical 5% after-MSE gate on every U4 background. Perfect-effect MSE equaled base MSE to numerical precision, validating the metric identity. The result demonstrates that the historical absolute-error threshold can reject a perfect transformation correction on these probes; it does not show that compensating context changes are impossible.

The model input to the learned operator uses the checkpoint's `group_to_code` bijection. Physical program identities, parameters, and conditional-mean counterfactuals are diagnostic-only. The protected v1 and M3 artifact trees have identical before/after hashes.

## Reproducibility and artifacts

The final run artifacts are under [`results/q1/u4_crossfit_v2_run5/`](../results/q1/u4_crossfit_v2_run5/): [configuration](../results/q1/u4_crossfit_v2_run5/config.resolved.yaml), [per-background CSV](../results/q1/u4_crossfit_v2_run5/crossfit_per_background.csv), [aggregates](../results/q1/u4_crossfit_v2_run5/crossfit_metrics.json), [P0 metric audit](../results/q1/u4_crossfit_v2_run5/p0_metric_audit.json), [diagnosis](../results/q1/u4_crossfit_v2_run5/diagnosis.md), [provenance](../results/q1/u4_crossfit_v2_run5/provenance.json), and [test statuses/logs](../results/q1/u4_crossfit_v2_run5/tests.json).

The focused legacy suite, 12 cross-fit tests, and `compileall` all passed. The tests manifest records earlier implementation failures and their corrections rather than hiding them.

## Next experiment

Do not change Q1's final gate from this result. If pursuing the candidate U4 M3 bottleneck, first preregister an effect-fidelity acceptance threshold, then run one isolated M3 development experiment with frozen M2 on a fresh development version. Confirmatory evaluation must use a separately frozen untouched protocol. No M3/M4 experiment was run here.