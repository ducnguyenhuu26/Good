# CLSM methods and implementation boundary

CLSM is the specified restricted model for *Theories in Flux*. The public history is a sequence of observations, commanded actions, public rewards, and next observations. Its hidden law is `omega_t=(z_t,theta_t,u_t)` with three binary mechanism bits, three continuous coordinates, and structural age. Physical transition `t` is emitted under `omega_t`; posterior correction uses that transition; only then does `K_eta` generate `omega_(t+1)`. Rewards are stationary functions of the public observation and commanded action and do not enter the model likelihood.

## Executable law

`MechanismForce` implements the specified force pathway: two learned nonnegative, unit-column-norm actuator supports `W` attenuate commanded motor action through `exp(W ell)`, with `ell_j=-z_j(delta_min+softplus(theta_j))`. Nominal actuator gear and joint topology give `B_e`; a positive normalized damping shape times nominal `d_ref` and observed joint velocities gives dissipative `tau_D`. Inactive coordinates are masked before nonlinear computation. `ExecutableLaw` sends only normalized public observation and the combined effective effort `tau_A+tau_D` through one shared SiLU MLP to produce the observation increment. Its Gaussian diagonal variance is separate from the mean model. Code, latent parameters, raw action, and reward have no direct MLP input.

The reference anchor uses code `000` and theta zero. It is initialized on separately collected nominal transitions. A trained reference anchor alone is **not** CLSM: joint nonstationary sequence training must optimize mechanism support, emission, semi-Markov evolution, and hierarchical priors together.

## Evolution, inference and memory

`LawEvolution` implements discrete-Weibull hazard `1-exp(-[((u+1)/kappa_z)^nu-(u/kappa_z)^nu])`, with `kappa_z=exp(eta_0+eta_1 popcount(z))`, a normalized sparse Hamming-edit kernel, Gaussian drift on active coordinates, and a same-structure parameter reset mixture. Same-structure reset increments age; structural switch sets age zero. `forward_log_evidence_discrete` is an exact finite-state timing reference and is checked against trajectory enumeration.

`LawFilter` maintains a float64 Gaussian mixture over code, age, active-parameter mean/covariance, and scratch/reuse hypothesis. `observe` computes pre-update predictive evidence and applies an EKF correction with Cholesky solves and Joseph covariance form. `evolve` branches through the semi-Markov law, moment-merges only equal `(code,age,hypothesis)` supports, reports discarded mass, and records proposal/pruning-corrected ancestry. `approximate_estep` runs this filter and an event-aware Gaussian backward pass on one independent world. This is an approximate inference path; sigma-point cross-checks, merge-error bounds and full generalized EM are still absent.

`TheoryMemory` defines Gaussian primitive and composition posteriors and an entry prior for scratch or reuse. The online filter updates prequential scratch/reuse evidence once per observed transition and contributes soft structural exit mass to prototype precision; same-code resets do not create a new visit. The expected completed-visit count sums responsibilities rather than counting update calls. Serial-dependence calibration and full hierarchical hyperparameter fitting are pending, so this is not a validated C3 result.

`LawForecaster` samples the current Gaussian mixture and propagates it through `K_eta^h` without new observations. It emits law draws and, on a fixed public probe bank, draws of normalized transition-law signatures. The identity-kernel comparator preserves the same current belief. `energy_score` implements the unbiased finite-sample pairwise estimator. Forecast correctness as a *trained* future-law model remains unverified until joint training and evaluator-only oracle targets are available.

## Scientific restrictions

CLSM uses Gaussian drift plus reset mixtures, not a general Levy process. It assumes exogenous law evolution and fixed public reward. Observable-law probes are finite projections, not recovery of an infinite-dimensional transition kernel. Local information or good next-state prediction cannot establish global structural identification or evolution skill. C1 composition, C2 evolution forecast skill against the same-belief frozen comparator, and C3 matched-budget memory gain require separate held-out experiments and paired uncertainty.
