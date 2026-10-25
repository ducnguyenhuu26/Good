# Q1 V5 identifiability and recovery claims

This note separates exact properties of the implemented parameterization from conditional statistical statements and open questions. It does not turn the exploratory runner into a proven recovery algorithm.

## 1. Exact properties of the action response — PROVED

For positive weights `pi_j` summing to one and positive slopes `s_j`, the implemented response

`psi(z) = sum_j pi_j tanh(s_j z) / sum_j pi_j s_j`

is odd, equals zero at zero, and is bounded in absolute value by `1 / sum_j pi_j s_j`. Its derivative is `sum_j pi_j s_j sech^2(s_j z) / sum_j pi_j s_j`, which is strictly positive for every finite `z`; the derivative at zero is one. These follow directly from oddness, boundedness and strict monotonicity of `tanh`. The model's `softmax` weights and positive transformed slopes enforce the assumptions.

Each learned action intervention maps `B` to `L B`; composing codes in order multiplies the corresponding operators in that order. Scalar rate interventions add in log-rate coordinates and commute. The two learned action matrices need not commute. The row-major derivative of `vec(LB)` is `L ⊗ I_2`, as used by `ProgramModel.transform_jac`. These are algebraic code properties; operator accuracy is empirical and conditional on the learned grouping.

## 2. Local identification of quadratic drag — CONDITIONAL

At a fixed observed design and a fixed local response model, let `J=[z_b Z]` be the stacked Gaussian mean Jacobian for the target coefficient and nuisance parameters. If noise is iid `N(0,sigma^2 I)` and the model is locally linear and regular, the efficient information is `I_eff = sigma^-2 z_b^T(I-P_Z)z_b`. For unbiased regular estimators, the local Cramér–Rao inequality gives `Var(b_hat) >= 1/I_eff`. Full column rank after nuisance projection is required for finite variance. The audit computes this projection with an SVD, avoiding an inverse of `Z^T Z`.

The audit parameterizes the target as `log(b)`, so its information and standard error are in log units. It includes six other task parameters plus five coordinates of the learned shared response as nuisance columns. The reported bound is conditional on the learned response family and local linearization. It is not a finite-sample coverage guarantee and does not establish absence of model bias. Collinearity of `v` and `||v||v`, and small velocity magnitude, reduce the residual target sensitivity.

## 3. Operator estimation — CONDITIONAL

If exact matrices `B_i` and `B_i'` obey `B_i'=L B_i`, minimizing squared Frobenius error gives `L_hat=(sum_i B_i'B_i^T)(sum_i B_iB_i^T)^-1` when the Gram matrix is nonsingular. Under additive estimation error in `B_i'`, the perturbation is amplified in proportion to the inverse smallest singular value of the design operator. In this project, backgrounds' `B_i` are estimated from noisy transitions, not observed exactly; this formula alone gives no bound for the implemented joint fit.

## 4. Finite-candidate program recovery — CONDITIONAL

For a fixed nonadaptive design, known conditional means, iid Gaussian noise with variance `sigma^2`, and a finite set of candidates, a pairwise likelihood comparison has error probability `Phi(-||mu_P-mu_Q||/(2 sigma)) <= exp(-||mu_P-mu_Q||^2/(8 sigma^2))`. A union bound over candidates outside the true functional-equivalence class yields the sum of these terms. This is an oracle-mean statement only.

Extending it to V5 requires additional controls on old-law estimation, learned action response, operator estimation, adaptive-state dependence, posterior approximation, and a positive learned-response separation margin. Those controls are not proved here. The ZIP pilot does not supply such a guarantee. Candidate ranking must continue to keep the two noncommuting action orders distinct; functional canonicalization belongs only in evaluation.

## 5. Active design — OPEN

An active-design guarantee would need bounded state/features, valid concentration for the adaptive trajectory, a positive lower bound on nuisance-residualized information, and a positive separation margin between candidate programs. The trajectory is adaptive, so fixed-design iid concentration cannot be imported without a martingale or other dependence argument. No `O(N^-1/2)` recovery guarantee is claimed, and no active-excitation implementation was evaluated in this integration run.

## Evidence boundary

See [the fixed-policy information audit](../results/q1/q1_v5_identifiability/IDENTIFIABILITY.md) for measured information and its limits. These mathematical statements do not upgrade exploratory metrics into a scientific pass.
