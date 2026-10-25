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

## 5. Active design — conditional objective; guarantees OPEN

The implemented Phase D policy uses a finite action grid in `[-1,1]^2` and a two-step rollout from the current observed state under frozen fitted V5 parameters. For each candidate action `a`, it forms the local Jacobian `J_a` for seven task coordinates and five shared-response coordinates, and adds `sigma^-2 J_a^T J_a` to the information accumulated from observed old and adaptation histories. It chooses the action maximizing `log(1 + I_eff(log b))`, where `I_eff` is the Schur-complement / residual-projection information after nuisance adjustment. This is a mathematically defined greedy surrogate for one-step information gain about quadratic drag. The uncertainty-driven comparator instead maximizes the ridge log-determinant of the full twelve-coordinate information matrix. The policy implementation receives no intervention label, simulator parameter object, or conditional mean.

This objective is locally model-based. Its information gain is exact for the stated linearized Gaussian model and the selected finite candidate, but model misspecification, nonlinear rollout error, or an inaccurate nuisance span can make it a poor proxy for actual information. A theorem for closed-loop identification would additionally require predictable bounded actions/features, conditionally sub-Gaussian martingale-difference noise, uniform nuisance-adjusted persistent excitation, and a local curvature/remainder bound. Phase D empirical results and its independent decision are recorded in `results/q1/q1_v5_active_excitation_dev/`; they do not change the fixed-policy Phase B decision.

## 6. Expanded conditions for the completed experiments

### 6.1 Nuisance-adjusted Fisher information — PROVED for the local Gaussian model

Write the stacked local mean as `mu(theta)` with `theta=(beta,eta)`, where `beta=log(b)` and `eta` contains the other six task coordinates and the five shared-response coordinates. Let `J=[z Z]` be its Jacobian at a fixed design and let `W=sigma^-2 I`. The local Fisher matrix is

`I(theta) = J^T W J = [[I_bb,I_beta],[I_eta_b,I_eta_eta]]`.

When the nuisance block has full rank, completing the square in the Gaussian score gives the efficient information

`I_eff = I_bb - I_b_eta I_eta_eta^-1 I_eta_b = min_g ||W^(1/2)(z-Zg)||^2`.

With rank deficiency, use the Moore–Penrose inverse on the identifiable nuisance span; equivalently project `W^(1/2)z` onto the orthogonal complement of `W^(1/2)Z`. Thus `beta` is locally identifiable after nuisance adjustment iff `I_eff>0`. If `I_eff=0`, a first-order change in `beta` can be canceled by nuisance changes on this design. This is local/design-specific and does not imply global structural nonidentifiability.

For a sequence of designs, a sufficient fixed-trajectory persistent-excitation condition is `I_eff(N) >= N lambda_b` for some `lambda_b>0`, with a stable nuisance rank and a uniform lower bound on nonzero nuisance singular values. The efficient CRLB is `Var(beta_hat) >= 1/I_eff`; for `b=exp(beta)`, the delta-method local bound is `Var(b_hat) >= b^2/I_eff`. These bounds assume a correctly specified locally linear Gaussian mean and regular unbiased estimation; they are not finite-sample intervals for the V5 optimizer.

### 6.2 Local estimation error — CONDITIONAL proposition

Suppose (i) observations satisfy `y_t=mu_t(theta*)+epsilon_t`, (ii) the score noise is conditionally mean-zero and `sigma`-sub-Gaussian, (iii) predictable Jacobian rows are bounded by `L`, (iv) throughout a ball of radius `r`, `sum_t J_t^T J_t >= N lambda I` for the full locally identifiable parameter vector, and (v) the Jacobian is `H`-Lipschitz with `H r <= lambda/4` after consistent per-sample scaling. Standard score concentration plus local strong convexity then yields, with probability at least `1-delta`, a local least-squares/MAP error of order

`||theta_hat-theta*||_2 <= C sigma L sqrt((p+log(1/delta))/N) / lambda`,

provided the right side is at most `r`; `C` depends on the chosen concentration inequality and parameter scaling. This is a conditional rate statement, not a proof for the repository's bounded nonlinear optimizers, approximate priors, or learned operators. In particular, `lambda` is the nuisance-adjusted persistent-excitation constant; without it the rate is vacuous. A pointwise bound for `b` follows locally from `|exp(beta_hat)-exp(beta*)| <= b_max |beta_hat-beta*|` on a bounded beta interval.

### 6.3 Operator stability and composition — PROVED algebraically; estimation conditional

For an ordered code sequence `c_1,...,c_m`, the learned action operator is `L_{c_m}...L_{c_1}`. If `||L_c||_2 <= rho_c`, then its norm is at most `product_j rho_{c_j}`. If estimated operators obey `||Lhat_c-L_c||_2 <= eps_c`, telescoping the product gives

`||Lhat_{c_m}...Lhat_{c_1}-L_{c_m}...L_{c_1}||_2 <= sum_j (product_{k>j}||Lhat_{c_k}||_2) eps_{c_j} (product_{k<j}||L_{c_k}||_2)`.

This is a deterministic perturbation inequality, not a stability guarantee for the simulated state dynamics. Scalar log-rate offsets compose additively. Independent offsets commute; matrix products generally do not. Least-squares operator recovery needs a positive smallest eigenvalue of `G=sum_i B_i B_i^T`; an operator error bound scales as the response-estimation error divided by `lambda_min(G)`. Because V5 estimates each `B_i` from noisy transitions, using this formula for its fitted operators additionally needs uniform coefficient-estimation control and a perturbation bound for `G`.

### 6.4 Finite-candidate recovery with estimated means — CONDITIONAL

For an independent ranking sample `Y` whose fit data are separate, consider the specific rule that chooses the nearest estimated candidate mean in squared Euclidean distance. Suppose an event `E` holds on which every candidate mean estimate differs from its true mean by at most `epsilon` in stacked Euclidean norm. If the minimum true mean separation from the correct functional-equivalence class is `Delta > 4 epsilon`, the signed distance from the true mean to any estimated pairwise decision hyperplane is at least `Delta/2-2 epsilon`: the two mean-estimation errors move the hyperplane normal and its offset, and the offset along its unit normal is bounded by `2 epsilon`. Gaussian projection tails and a union bound therefore give

`P(error) <= P(E^c) + sum_{Q not equivalent P} exp(-(Delta_PQ-2 epsilon)^2/(8 sigma^2))`.

This accounts for mean-estimation error only when the uniform event `E` is justified and the ranking observations are independent of the fitted candidate means. It is an oracle-style nearest-mean bound, not a direct theorem for V5's adaptation-data marginal-likelihood ranking. For V5, a relevant event would have to cover the old-law MAP, shared response, every ordered operator composition, posterior approximation, and all candidate means; the same-data adapt-and-rank dependence would also need a separate argument. None of those uniform errors or dependence controls has been bounded for the implementation, so this is a conditional proposition, not an unconditional V5 guarantee.

### 6.5 Adaptive-data concentration — CONDITIONAL requirements

For an adaptive rollout, require actions and predicted features to be measurable with respect to the pre-transition filtration, bounded, and selected by the same policy under every candidate; require observation noise to be a conditionally sub-Gaussian martingale difference with the declared scale. Under these assumptions a self-normalized martingale inequality can control the score in the random design ellipsoid, with radius of order `sigma sqrt(p log(1+N L^2/lambda_0)+log(1/delta))`. Converting that score bound to parameter error still requires a predictable-design restricted strong-convexity/persistent-excitation lower bound and a local nonlinear remainder bound. If actions use future observations, oracle means, or candidate-specific hidden state, the conditions fail. The active policy implemented for this study uses only the current observed state, past observed design, and frozen fitted parameters; whether its trajectories satisfy the required uniform excitation and concentration conditions remains an empirical/open question.

## 7. Claim labels

- **PROVED:** positivity, oddness, boundedness and unit local slope of the positive saturating mixture; exact ordered operator composition; the local Schur-complement identity; Gaussian fixed-design pairwise bound; and deterministic operator-product perturbation inequality.
- **CONDITIONAL:** CRLB and local estimation rate under correct specification, regularity and persistent excitation; operator recovery under a well-conditioned coefficient design; finite-candidate error bound with a uniform mean-estimation event; and self-normalized concentration for predictable bounded adaptive designs.
- **EMPIRICAL:** fixed-policy and active-policy outcomes, Fisher diagnostics, and bootstrap intervals, which apply only to the recorded sampled backgrounds and protocol.
- **OPEN:** theorem-level bounds for the actual V5 learned-response/operator pipeline, posterior calibration under misspecification, uniform candidate margins for estimated adaptive means, and coverage guarantees for the active information surrogate.

## Evidence boundary

The original weak-excitation measurements are in [the fixed-policy information audit](../results/q1/q1_v5_identifiability/IDENTIFIABILITY.md). Fresh Phase B and Phase D empirical results are in [the fixed-policy decision](../results/q1/q1_v5_fixed_confirmatory_dev/DECISION.md) and [the separate active decision](../results/q1/q1_v5_active_excitation_dev/DECISION.md). They apply to the recorded development backgrounds and bootstrap protocol only; they do not access protected test data or upgrade one protocol's verdict into the other's.
