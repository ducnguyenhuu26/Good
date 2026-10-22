# CLSM-R2 mathematical contract

This document defines the version-2 learner and its data boundary. It is an implementation contract, not evidence that all scientific gates have passed. CLSM-R1 checkpoints are rejected by R2 loaders through the required `architecture_revision: 2` field.

## Variables and timing

There are three binary structural mechanisms. At transition index `t`, `z_t` is an integer in `0..7`, `u_t` is the number of consecutive emissions already assigned to the current structural code, and `xi_t` is the three-dimensional latent strength. `z=0` is nominal; its strength is fixed at zero and it has no reset event. `o_t`, `a_t`, and `o_{t+1}` are public. The learner receives the static nominal schema and may receive a declared initial-law prior. It never receives realized `z`, `xi`, reset indicators, future observations, or modified simulator arrays.

For a known-nominal start, `P(z_0=0,u_0=0)=1`. An unknown start is uniform over all eight codes at age zero. The registered COMP calibration start is uniform over `{0,1,2,4}` at age zero. These are distinct configured priors.

## Global actuator partition and bounded strength

For `d_a` selected actuators, `G` is a binary `d_a x 2` matrix. Every row sums to one and both columns are nonempty. Candidate partitions are canonicalized by requiring actuator zero in group zero, removing only the group-label permutation. There are `2^(d_a-1)-1` candidates; at six actuators there are 31. R2 estimates one global partition from train-only public stationary calibration worlds by profile expected log likelihood. The 12-point Gauss-Hermite rule integrates each active scalar standard-normal strength. Candidate selection is outside online filtering and validation/test data do not affect it.

For active motor coordinates `j=1,2`, and damping coordinate `j=3`:

\[
m_j(\xi_j)=l_j+(h_j-l_j)\Phi(\xi_j),\quad
(l_1,h_1)=(l_2,h_2)=(0.50,0.85),\quad(l_3,h_3)=(1.30,2.00).
\]

An inactive mechanism has multiplier exactly one and zero strength gradient. These are declared benchmark-family bounds. At entry, `xi~N(0,1)` induces an exactly uniform physical multiplier; the distribution need not remain uniform after drift or conditioning.

Let `B` be the public nominal actuator-to-DOF matrix and `d_ref` the nominal damping vector. With `b(z)` the first two bits and `g_j` the columns of `G`, the effective force is

\[
\tau(o,a,z,\xi)=B\,\mathrm{diag}\{1+(m_1-1)z_1g_1+(m_2-1)z_2g_2\}a
-d_{ref}\odot(1+z_3(m_3-1))\odot v(o).
\]

The force features enter the existing two-hidden-layer SiLU map `F_phi`, with residual-state normalization. The output distribution is diagonal Gaussian over the next public observation. `damping_shape` is not learned. A separately collected nominal reference replay anchors the same network and variance calibration.

## Explicit-duration evolution and exact categorical recursion

For `z!=0`, the discrete Weibull hazard is

\[
h_z(u)=1-\exp\{-[(u+1)/\kappa_z]^\nu+[u/\kappa_z]^\nu\},\quad
\kappa_z=\kappa_0\exp(\kappa_{pop}|z|).
\]

At a structural boundary, a row-stochastic sparse-edit kernel selects `z'!=z` and age resets to zero. Without a structural boundary, age increments; `xi` either drifts under a diagonal process covariance on active coordinates or resets to its entry prior. The reset probability is zero for nominal code.

For every parent `(z,u)`, outgoing discrete mass is partitioned into `(1-h_z(u))(1-rho_z)` stay-drift, `(1-h_z(u))rho_z` same-code reset, and `h_z(u)A(z'|z)` switches. Their sum is one. R2 stores log mass in an `8 x (max_world_length+1)` table and does not prune age states. Observation emissions normalize the table with log-sum-exp. The event-conditioned forward-backward reference also retains stay-drift, same-code reset, and switch probabilities separately.

Conditional strength is one Gaussian per code. Each event template is corrected by positive symmetric cubature with `2p` points for `p` active coordinates, then same-code/age contributions are moment-matched. This is a continuous approximation; exact categorical mass conservation does not imply accurate continuous posterior moments. The implementation records branch likelihoods and cubature diagnostics. No 128-component or per-age top-K approximation is used.

## Hierarchical visit memory

Each world receives a fresh memory object. For each nonnominal code, the learned population prior has mean `Omega_z` and covariance `V_z`; a new visit uses the Gaussian prior `N(Omega_z, V_z + Sigma_within)`. A scratch comparator uses the population prior independent of previous visits. Completed visit posterior summaries update the hierarchy once with their boundary responsibility. A same-code reset closes the parameter visit but preserves structural age. Structural switches close the parent visit and start a target-code visit. Nominal code has no parameter visit.

The conjugate update is precision weighted. For independent visit summaries `(mu_i,S_i)` with soft responsibility `r_i`, the effective precision increment is `r_i S_i^{-1}` and the natural-parameter increment is `r_i S_i^{-1}mu_i`; a boundary ledger prevents replaying the same `(time,code)` update twice. This is an event-weighted approximation, not a generic no-negative-transfer guarantee.

## Forecast and evaluation boundary

Forecasting begins from the current public posterior, propagates the exact finite code-age law and the configured strength approximation, and never reads future evaluator arrays. Comparisons must separate learned-current/learned-evolution from oracle-current/learned-evolution and learned-current/oracle-evolution. Persistence freezes the current belief and is the required comparator. C1 composition, C2 future-law prediction, and C3 memory scores remain gated by the corresponding diagnostics and are not implied by the CPU pilot.
