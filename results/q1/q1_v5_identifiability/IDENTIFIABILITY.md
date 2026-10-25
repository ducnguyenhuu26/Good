# Q1 V5 fixed-policy identifiability audit

## Result

This is a diagnostic for local identification of `log(b)` under the supplied fixed action policy, not a revised Q1 test and not a test of the global learned model. The audit used two reproduction seeds, 16 ID-like and 16 background-OOD backgrounds per seed, 96-transition old-support trajectories, and prefixes of those same trajectories at N={5,10,20,50,96}. Shared positive-saturating action-response parameters were learned from the 40 training backgrounds per seed and included as nuisance columns in the local Jacobian. Simulator truth was used only in the explicitly offline `b_true_offline` / error columns.

At N=96, the median efficient information for `log(b)` was 3.69 ID-like and 8.23 OOD, corresponding to local standard-error bounds 0.52 and 0.35 in log units. Median absolute offline b estimation error was 0.050 ID-like and 0.031 OOD. Median cosine between stacked linear- and quadratic-drag velocity features was approximately 0.96 in both regions. At N=5 the efficient information was numerically zero because the nuisance columns span the short observation design; this is finite-design rank deficiency, not evidence of structural nonidentifiability.

The information increases overall with longer supports, while the N=20 to N=50 estimates are not required to be strictly monotone because the Jacobian is recomputed at a different fitted background parameter at each prefix. The stored per-background CSV exposes that variability. At N=96, the information is positive and finite, so the coefficient is locally identifiable in these fixed designs, but the moderate information and high feature collinearity leave substantial practical uncertainty. This supports a **practical weak-excitation / nuisance-collinearity** explanation, not a claim that `b` is structurally unidentifiable.

## Method and interpretation

The computational parameter is `log(b)`, so the reported information and CRLB standard error are for `log(b)`. The target derivative is the fifth coefficient of the observable response Jacobian. Nuisance columns include the other six task parameters and five shared-response coordinates (two mixture-weight logits and three log-slopes). An SVD projection removes their span; no inverse of an ill-conditioned nuisance Gram matrix is used. Noise is treated as independent Gaussian with `sigma=0.0005` per velocity coordinate.

This local CRLB is conditional on the fitted response family and local linearization. It is not a guarantee for a biased or misspecified estimator and it omits finite-sample posterior calibration. The offline truth comparison is a diagnostic of calibration, not an estimator input or model-selection signal. The audit does not establish a population theorem, nor distinguish prior-dominated estimation for the EB estimator.

The audit gives a concrete reason to investigate `Q1_ACTIVE_EXCITATION` separately. That new protocol has **not** been implemented or evaluated in this run. No active-policy benefit is claimed.

Files: `fisher_per_background.csv`, `fisher_summary.json`, `split_manifest.json`; implementation is in `q1_v5/identifiability.py` and `scripts/q1_v5_identifiability.py`.
