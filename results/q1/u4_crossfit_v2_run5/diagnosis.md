# U4 cross-fit diagnostic v2, run 5

**Decision: INCONCLUSIVE.** This is development-only evidence on the previously inspected 16 validation backgrounds. It is not a Q1 pass and does not establish final acceptance.

At fit 128/query 256, U4 linear cross-fit effect skill averaged 0.190 (background-clustered 95% CI 0.126–0.255); nonlinear executor skill averaged 0.194 (0.131–0.258). The existing beta=0.25 M3 operator scored -3.600 (-5.736 to -1.463). This is a U4 operator-learning bottleneck candidate, conditional on these inspected probes and the diagnostic metric. At fit 512, linear and nonlinear U4 skills were about 0.319 and 0.324; the M3 comparison remained poor. The larger fit count was predeclared and run regardless of the primary result.

Controls were mixed: U1 had positive linear skill but an inconclusive nonlinear interval; U3 had positive linear and nonlinear skill and a positive M3 interval. This mix, and the absence of a preregistered minimum effect-accuracy threshold, requires an overall INCONCLUSIVE label. The query-refit projection uses query labels by design and is only a leakage upper-bound diagnostic; its U4 mean skill was 0.556 and is excluded from transfer claims.

For U4, virtual perfect-effect absolute-MSE gain had a mean-of-background-ratios of 1.98%, with every background between 0.61% and 4.25%. Thus the historical 5% after-MSE threshold failed even under a perfect effect correction on these probes. The perfect-effect identity held exactly in saved float64 metrics, and the error decomposition maximum residual was 6.94e-18.

All physical effects use simulator conditional means at identical probes. M2 and M3 were frozen; no model training occurred. Test/OOD splits were excluded. Legacy result trees matched their before/after hashes. Physical program identities and simulator counterfactuals are diagnostic-only and are not deployed predictor inputs.
