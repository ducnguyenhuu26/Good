# Q1 V5 pre-change excitation: mathematical audit

This note records exact algebraic facts, results proved only under stated assumptions, and empirical claims separately. It does not assert an unconditional excitation or recovery theorem for the implemented simulator/policy. The mathematical model and complete derivations came from the architecture handoff supplied on 2026-10-25; verification fixtures are in `tests/test_q1_v5_prechange.py` and the stage-aware information code is in `q1_v5/prechange_design.py`.

## Model and information boundary

The observed innovation is two dimensional:

`Δv_t = h[ψφ(Ba_t) − d v_t − b||v_t||v_t − κq_t] + ε_t`,

where `h=.05`, `ε_t | F_t ~ N(0, σ² I₂)`, and `σ=.0005`. Position updates deterministically from the next velocity, so the likelihood and Fisher calculations use the two velocity innovations, not a nonsingular four-dimensional next-state Gaussian. The latent vector is `θ=(vec_r(B), log d, log b, log κ)∈R⁷`. The response is a learned shared odd saturating mixture; its parameters are model coordinates, not physical simulator parameters. This is a grey-box model: the force decomposition and response family are inductive biases, while coefficients and intervention maps are estimated from noisy observations.

Physical intervention names below are solely for mathematical statements, generation, and offline scoring. A learner receives opaque intervention labels and observed transitions. A pre-change action schedule receives only time and its predeclared random seed. Offline stage-correct Fisher calculations may condition on an actual program for auditing; no such label is supplied to action generation.

## Exact results — proved

### Theorem 1: response regularity

For `π_j>0`, `Σπ_j=1`, `s_j>0`, define

`ψ(z)=Σ_j π_j tanh(s_j z) / Σ_j π_j s_j`.

Oddness and `ψ(0)=0` follow termwise. Differentiation gives

`ψ'(z)=Σ_j π_j s_j sech²(s_j z) / Σ_j π_j s_j`.

Every term is positive at finite `z`, so `ψ'(z)>0`; because `sech²≤1`, `ψ'(z)≤1`, with equality at zero. The mean-value theorem makes `ψ` globally 1-Lipschitz. `|tanh|≤1` gives `|ψ(z)|≤(Σπ_j s_j)⁻¹`. This does not prove the physical response was recovered or identify mixture components beyond function equivalence.

### Theorem 2: ordered operator composition

For a left action operator `B↦LB`, row-major vectorization obeys `vec_r(LB)=(L⊗I₂)vec_r(B)`. Sequential codes `(i,j)` produce `L_j L_i B`. Scalar log-rate offsets commute with one another and with action-coordinate transforms. With `R=R(π/6)` and `D=diag(1.5,1)`,

`DR−RD = [[0,−1/4],[−1/4,0]]`,

so the two action orders remain distinct whenever `B` is nonsingular. Algebraic distinction alone does not imply finite-trajectory statistical separation.

### Theorem 3: post-stage observational equivalence of two joint hypotheses

For `r>1`, compare `H₁: b_old=b, P=(U4)` with `H₂: b_old=b/r, P=(U4,U4)`, keeping other coefficients equal. Both have `b_post=rb`. Conditional on a common change-point state, their post-stage transition kernels agree at every observed state/action. Under any common history-based post policy, the conditional post-innovation likelihood ratio is one. This is a counterexample in the joint unknown-background/program space; it does not say the two programs are equivalent at the same known old `b`, nor that both old coefficients occur in the sampled prior range. Old-stage observations and the boundary-state distribution can distinguish the full hypotheses.

### Corollary 3.1: old-stage KL and binary lower bound

For a common predictable old-stage policy and Gaussian innovation noise with variance `σ²I₂`, the chain rule for KL gives

`KL(H₁||H₂)= h²(b−b/r)²/(2σ²) · E_H₁[Σ_{t<N_old} ||v_t||⁴]`.

At a common history the mean difference is `−h(b−b/r)||v_t||v_t`; the equal-covariance Gaussian KL is its squared norm divided by `2σ²`. Post-stage conditional KL is zero by Theorem 3. Equal-prior testing error is at least `¼ exp(−KL)` by Bretagnolle–Huber. This supports the claim that old velocity excitation can matter; it does not prove the four-block schedule is persistently exciting.

### Theorem 4: exact U4 effect-error factorization

For U4 changing `b↦rb`, on a shared query state/action,

`Δ_U4 = −h b(r−1)||v||v`.

With estimates `b̂,r̂` and the same feature, the vector squared error under any query distribution `Q` is

`E_Q ||Δ̂_U4−Δ_U4||² = h²[b̂(r̂−1)−b(r−1)]² E_Q||v||⁴`.

The coefficient error is bounded by `|b̂−b||r−1|+|b̂||r̂−r|`. When the true effect energy is positive, normalized effect skill is exactly `1−([b̂(r̂−1)−b(r−1)]/[b(r−1)])²`. Positive skill and superiority to a population comparator are different claims; neither follows without estimation bounds.

### Theorem 5: two-coordinate information bottleneck

Let `γ=log b`, `β=log r`. After projecting other nuisances, suppose old-stage information for `γ` is `A≥0`, and post-stage information `C≥0` is on the common coordinate `γ+β`. Then

`I_(γ,β)=[[A+C,C],[C,C]]`, `det I=AC`, and `I_(β|γ)=C−C²/(A+C)=AC/(A+C)≤A`.

The old stage contributes `diag(A,0)` and the post stage contributes `C[[1,1],[1,1]]`; their sum and Schur complement give the result. If `A=0` the matrix is rank one; increasing post information alone cannot exceed the old-stage information limit for `β`. This is an exact two-coordinate local toy calculation, not the full V5 Fisher matrix or a discrete program-recovery theorem.

## Conditional results — assumptions are part of each claim

### Proposition 6: nuisance-residualized Fisher information

For a fixed locally linearized Gaussian design with Jacobian `J=[z_b,Z]`, where `z_b=∂μ/∂log b=−h b||v||v` and `Z` contains all nuisance columns,

`I_eff(log b)=σ⁻² z_bᵀ(I−P_Z)z_b`, `P_Z=ZZ†`.

This is the Schur complement when `ZᵀZ` is nonsingular and the SVD/pseudoinverse projection otherwise. It is zero exactly when the target column lies in the nuisance span. A local regular unbiased estimator has the CRLB `Var(log b̂)≥1/I_eff`; a lower information bound alone does not upper-bound the variance of an arbitrary biased nonlinear optimizer.

### Proposition 7: fixed-design estimation bound

In `y=zγ+Zη+ε`, with deterministic design and iid `N(0,σ²)` noise, let `z⊥=(I−P_Z)z`. OLS satisfies `γ̂−γ=z⊥ᵀε/||z⊥||²` and `Var(γ̂)=σ²/||z⊥||²`. If the verifiable excitation condition `||z⊥||²≥cN` holds, Gaussian tails give

`P(|γ̂−γ| > σ sqrt(2 log(2/δ)/(cN))) ≤ δ`.

The proof residualizes nuisance and applies a two-sided Gaussian tail. Actual Q1 states and actions are noise-dependent; this deterministic-design argument does not by itself apply to the controlled process, learned response, or misspecified MAP/Laplace estimator.

### Proposition 8: operator least squares

For exact inputs `B_i` and `B'_i=LB_i+E_i`, let `G=Σ_i B_i B_iᵀ` and assume `λ_min(G)>0`. Then `L̂=(Σ_i B'_i B_iᵀ)G⁻¹` and

`||L̂−L||_F ≤ ||Σ_i E_i B_iᵀ||_F / λ_min(G)`.

Differentiate the squared Frobenius loss, use its normal equation, substitute the error model, then use `||G⁻¹||₂=1/λ_min(G)`. V5 estimates `B_i` from noisy transitions, so errors-in-variables and opaque-group assignment require additional control.

### Proposition 9: stability of a finite composition of estimated operators

For true operators `L₁,…,L_k` and estimates `L̂_i`, define left-to-right program composition `P=L_k⋯L₁` and `P̂=L̂_k⋯L̂₁`. Suppose `||L̂_i−L_i||₂≤ε_i`, `||L_i||₂≤M_i`, and `||L̂_i||₂≤M̂_i`. A telescoping expansion gives

`||P̂−P||₂ ≤ Σ_(j=1)^k (Π_(i>j) M̂_i) ε_j (Π_(i<j) M_i)`.

Indeed, replace one factor at a time from right to left and apply submultiplicativity to each summand. If all factors are bounded by `ρ`, the error is at most `ρ^(k−1) Σ_j ε_j`; when `ρ≤1`, it grows at most linearly in accumulated operator error. For `ρ>1` the bound can grow geometrically, so short programs or contractive/norm-controlled maps matter for stability. This is only a propagation bound conditional on factorwise accuracy; it does not establish that V5 learns accurate maps or that its latent operator parameterization satisfies these norms.

### Theorem 10: oracle recovery under adaptive controls

Let `P*` be true and `K` candidate functional classes, with known conditional means. For each wrong `Q`, let `d_t(Q)=μ_P*(s_t,a_t)−μ_Q(s_t,a_t)` be predictable before Gaussian innovation `ε_t|F_t~N(0,σ²I₂)`. Suppose, jointly for all wrong classes except an event of probability `δ_sep`, `S_N(Q)=Σ||d_t(Q)||²≥s_N>0`. Then conditional-likelihood selection obeys

`P(P̂≁P*) ≤ δ_sep+(K−1) exp(−s_N/(8σ²))`.

For each Q, the wrong-minus-true squared residual is `S_N+2M_N`, with `M_N=Σ<ε_t,d_t>`. The exponential process `exp(−M_N/(2σ²)−S_N/(8σ²))` is a mean-one martingale because the conditional Gaussian MGF cancels the drift. If Q beats truth and `S_N≥s_N`, that process is at least `exp(s_N/(8σ²))`; Markov and a union bound prove the result. This accommodates predictable adaptive actions, but assumes oracle means, true candidate inclusion, and realized joint separation. It does not show the block policy achieves `s_N≈Nγ²` or cover approximate V5 marginal evidence.

### Proposition 11: frozen mean-model perturbation on an exogenous design

For deterministic design and independent `ε~N(0,σ²I)`, suppose `||m̂_P*−m*||≤e` and every wrong-class candidate obeys `||m̂_Q−m̂_P*||≥d>2e`. Nearest estimated mean selection has error probability at most `(K−1)exp(−(d−2e)²/(8σ²))`. For `h=m̂_Q−m̂_P*`, a wrong decision requires `2<ε,h>≥||h||²−2e||h||`; normalize by `σ||h||`, apply a Gaussian tail, then union-bound. This result assumes frozen candidate means independent of scoring noise and exogenous design; it is not a theorem for adaptive same-data V5 marginal ranking.

## Stage-correct information and policy boundary

For offline candidate `P`, pre-stage derivatives are evaluated at `θ_old`; post-stage derivatives are evaluated at `T_P(θ_old)` and chain-ruled into old coordinates:

`J_P = [[Jθ(x₀;θ,c), Jc(x₀;θ,c)], [Jθ(x₁;T_P(θ),c) D T_P(θ), Jc(x₁;T_P(θ),c)]]`.

`stage_corrected_jacobian` implements this local derivative for `(θ_old,c)`. Its test compares the post block against a numerical derivative. The pre-change schedule itself uses the simpler old-stage target information only, so it does not require an intervention hypothesis online. If a future online selector uses total two-stage utility, it must average over a learned program posterior from observable history and include uncertainty in learned operator parameters; it must not condition on true `P`.

## Claim ledger

- **Proved:** mixture regularity; ordered operator algebra and noncommutation; conditional post-stage equivalence counterexample; exact old-stage KL under stated policy/noise conditions; exact U4 effect identity; two-coordinate Fisher bottleneck; local Schur complement; fixed-design OLS tail; finite-composition perturbation bound; oracle adaptive selector bound; fixed-design frozen-mean bound.
- **Conditional:** population recovery rates, operator error, finite-candidate recovery with learned means, adaptive-design estimation concentration, and any end-to-end V5 finite-sample result. Each requires the regularity, excitation, separation, independence, and nuisance controls stated above.
- **Empirical only:** action-energy matching, information ranks/values, EB calibration, program/effect metrics, and background bootstrap intervals. These apply to the sampled development backgrounds and frozen protocol.
- **Open:** uniform persistent excitation for the block schedule, globally consistent learned-response/operator estimation, Laplace/EB calibration under adaptive-state misspecification, and an end-to-end non-oracle program-recovery theorem.
