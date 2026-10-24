# Q1 feasibility A/B/C decision

**Overall status: DEVELOPMENT_DIAGNOSTIC — INCONCLUSIVE. This is not `Q1_PASS`.** The 16 existing validation backgrounds are `DEVELOPMENT_OOD`, not confirmatory. The new ID-like backgrounds are also development data.

## Evidence summary

| Experiment | Result | What it supports / does not establish |
|---|---|---|
| A: old-physics information | U4 class accuracy at N=100: privileged true-old A1 ID-like 0.812, estimated-old A2 0.438; `DEVELOPMENT_OOD` A1 0.875, A2 0.812. | Old-law estimation is a material ID-like bottleneck in this privileged equation-aware diagnostic; A2 nearly approaches A1 on the inspected OOD backgrounds. A1 is not a fair learned baseline. |
| B: M2 representation | Ridge probe log(d,b) MSE vs intercept: ID-like 0.0162 vs 0.0157; OOD 0.1698 vs 0.1783. U4 effect-energy / frozen-M2 old-error ratio: ID-like 0.00106, OOD 0.0006531. | The linear context probe adds no ID-like value and modest OOD value. Failure does not prove M2 lacks nonlinear information. U4 is tiny relative to executor old-law error. |
| C: direct function-space effect | U4 mean skill (background 95% CI): direct G(c) ID-like 0.515 [0.312, 0.719], context-free 0.598 [0.400, 0.796], M3 -21.797 [-48.443, 4.849]; OOD direct G(c) 0.832 [0.794, 0.869], context-free 0.875 [0.843, 0.906], M3 -5.581 [-7.719, -3.444]. | Direct G transfers better than M3 for U4, but context-free G matches or exceeds context-aware G. This does not show that G uses old-law information. Training used privileged paired simulator effects. |

## Interpretation and limitations

A fits the old law using only the same 96 noisy old-stage transitions used for context encoding. The solver knows the simulator equation and uses generic bounds; it receives no true parameter initialization or post-change observations. Its old-support residual is low, but estimated quadratic drag varies substantially from the reporting-only true value, and U4 ranking is weaker than known-old A1, especially on ID-like backgrounds. This supports an old-physics observability/estimation contribution, not a proof that ordinary few-shot inference can reach A1.

B fits only the original train backgrounds with a 36/12 background inner split and refits on all 48 training backgrounds. The transition-statistic baseline and intercept are included. The M2 context ridge is slightly worse than the intercept on ID-like log(d,b) error and somewhat better on OOD; an expressive probe might recover information the linear probe misses. The very small U4 effect energy compared with frozen-M2 old-law error also limits absolute after-MSE movement.

C trained only direct-effect heads on 40 original train backgrounds, selected best step on 8 inner train backgrounds, and used two fixed optimizer seeds with matched update counts and minibatch schedules. The paired simulator conditional-mean targets are privileged supervision stronger than opaque group labels. Evaluation used independent natural query trajectories common across all methods within each background/program. The head is single-intervention only; composition is untested.

The A2 physics plug-in and C direct head are diagnostic references, not ordinary deployment predictors. Simulator parameters/program labels are used only for target construction or offline scoring. No M2/M3/M4 retraining or test/OOD split evaluation occurred. Protected legacy trees retain their before/after hashes.

## Most supported next architecture decision

Keep a direct function-space effect head as a research baseline against latent M3, but do not yet make it context-conditioned by default: the context-ablated head is as good or better here, and the A2 ID-like fit reveals old-law estimation difficulty. Do not redesign M2 or retrain M3/M4 from this run. A fresh, preregistered development study should first test whether a nonlinear probe of frozen `c_old` adds paired predictive value over transition statistics and context-free G on broader samples from the same declared training ranges. Any later acceptance claim needs a separately frozen untouched protocol.

Evidence that would change this decision: (i) a context-aware direct head beats the context-free head on paired backgrounds in both regimes and improves prediction of held-out old-law parameters; (ii) a context-free head continues to match it, arguing that old-context conditioning is unnecessary on this task distribution; or (iii) the equation-aware A2 estimate closes the A1 gap on fresh development backgrounds, weakening the old-law observability explanation.

## Artifact index

See `a_system_id_metrics.json`, `a_per_background.csv`, `a_u4_identity_confusion.json`, `b_representation_metrics.json`, `b_per_background.csv`, `c_direct_effect_metrics.json`, `c_per_background.csv`, `c_training_curves.json`, `c_gradient_diagnostics.json`, `provenance.json`, and `tests.json` in this directory.
