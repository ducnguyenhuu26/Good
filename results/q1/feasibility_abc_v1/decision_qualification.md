# Decision qualifications

This addendum gives paired-background detail for the overall **INCONCLUSIVE** decision in `decision.md`.

- For true U4 at N=100, A1 versus A2 functional-class accuracy was 0.812 vs 0.438 on ID-like development backgrounds (a 37.5 percentage-point gap), and 0.875 vs 0.812 on `DEVELOPMENT_OOD`. A1 supplies true old physics and is privileged. A2's reporting-only quadratic-drag parameter RMSE was 0.0865 ID-like and 0.0551 OOD; its old-support residual RMSE was near the configured observation-noise scale. The row-level estimates and uncertainty are in `a_parameter_error_summary.json` and `a_u4_metrics.json`.
- The mean-of-background-ratios for U4 effect energy divided by frozen-M2 old-law MSE was 0.00106 ID-like and 0.000653 OOD. Thus U4's signal is much smaller than the executor's old-law error on these query trajectories.
- C's paired background skill difference, direct G(c) minus context-free G, was -0.083 (95% CI -0.155 to -0.011) ID-like and -0.043 (-0.062 to -0.024) OOD. Context-free G was better in both regimes here.
- C's paired direct G(c) minus M3 U4 difference was +22.312 (95% CI -4.239 to 48.864) ID-like, which is imprecise because of extreme per-background M3 skill values; the corresponding OOD difference was +6.413 (4.281 to 8.545). The OOD comparison favors direct G, but its privileged training supervision and lack of context benefit prevent a deployment claim.

All intervals use backgrounds as the uncertainty unit, with C optimizer seeds averaged within background. These diagnostics use development data and do not change the no-confirmatory-evidence conclusion.
