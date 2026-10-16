# RWT Stage 2A — Learned compositional dynamics

Stage 2A replaces the Stage 1 oracle candidate library with neural transition
models trained from simulator-produced transitions. It does not tune Stage 1.5
policies, change NEO upstream, or implement Stage 2B/Stage 3.

## Reused NEO interface

The upstream `src/models/program_execution.py::ProgramExecutor.forward(state,
action)` is the reusable part: it is already an action-conditioned residual
state transition, with action FiLM modulation. `src/rwt/stage2a.py` adds a thin
adapter that reshapes continuous vectors as one-token tensors and uses the
upstream executor unchanged. The same adapter is used by the monolithic,
independent-expert and compositional models for a comparable function class.

The other NEO components were not forced into this task. The upstream
`TheoryProgrammer` maps current/target latent observations to a continuous
operation; RWT instead has a control action and learns a theory from transition
likelihood. The upstream vector quantizer discretizes operation vectors, not
dynamics factors. Both would require a larger semantic change than the thin
executor adapter.

## Model and learning objective

Let (x_t=[s_t,a_t]), and let each learned executor produce a residual
(f_j(x_t)). The compositional mean is

\[
\widehat{s}_{t+1}(x,z)=s_t+f_0(x)+z_1f_1(x)+z_2f_2(x),
\quad z\in\{(0,0),(1,0),(0,1),(1,1)\}.
\]

The conditional distribution is a learned diagonal Gaussian

\[
p(s_{t+1}\mid x,z)=\mathcal{N}(\widehat{s}_{t+1}(x,z),
\operatorname{diag}(\sigma^2)),
\]

with trainable per-coordinate scale. The main model learns by marginal
likelihood over the *three training codes* only:

\[
\mathcal{L}=-\sum_i\log\left[\frac{1}{3}\sum_{z\in\{00,10,01\}}
p(s'_{i}\mid x_i,z)\right].
\]

There are no regime IDs, primitive IDs or physical parameters in that loss.
After training, the fourth bit-composition `11` is predicted by executing the
same base and residual primitives together; it is never included in training or
validation likelihoods. At evaluation, a causal Bayesian theory belief starts
uniform over learned candidate codes, predicts before seeing each transition,
and updates from its likelihood afterward. An unknown-regime hazard prior
allows belief recovery after change points.

## Data split and verification

Train and validation schedules use only A/B/D. Evaluation alone includes C.
The checked environment settings satisfy `C.friction == B.friction` and
`C.rotation == D.rotation`, while C itself differs from A, B and D. Thus C
combines the friction mechanism seen in B with the rotation mechanism seen in
D. Dataset training objects contain only `(state, action, next_state)` arrays;
the schedule/regime sidecar is consumed only by the evaluator and by a clearly
separate privileged diagnostic.

The reported CPU run collected 1,160 train transitions, 284 validation
transitions and 682 evaluation transitions from 12 train seeds, 3 validation
seeds and 6 evaluation seeds. All learned baselines use the same transition
arrays. Parameter counts are 35,864 (monolithic), 37,032 (independent experts)
and 37,024 (compositional), within 3.3% of one another. Checkpoint selection
uses A/B/D validation likelihood only.

## Results

MSE and rollout MSE are averages over four state coordinates. NLL is the
one-step predictive mixture NLL per transition. The h=5 rollout starts from an
observed state and holds the inferred theory belief fixed while autoregressing
over five actions; only rollouts wholly inside one regime are scored. C gap is
`MSE(C) - mean(MSE(A), MSE(B), MSE(D))`.

| Model | Regime | One-step MSE | One-step NLL | 5-step rollout MSE |
|---|---:|---:|---:|---:|
| Monolithic | A | 0.0150 | -2.292 | 0.346 |
| Monolithic | B | 0.0240 | -0.145 | 0.448 |
| Monolithic | D | 0.0827 | 13.981 | 1.461 |
| Monolithic | C | 0.0746 | 12.003 | 1.078 |
| Independent experts | A | 0.0241 | -3.032 | 0.753 |
| Independent experts | B | 0.0295 | -2.238 | 0.617 |
| Independent experts | D | 0.0648 | 4.752 | 1.560 |
| Independent experts | C | 0.0668 | 5.101 | 1.105 |
| Learned composition | A | 0.0106 | -3.995 | 0.359 |
| Learned composition | B | 0.0194 | -2.498 | 0.454 |
| Learned composition | D | 0.0598 | 3.889 | 1.476 |
| Learned composition | C | 0.0764 | 5.770 | 1.283 |
| Privileged factor diagnostic* | A | 0.0038 | -5.188 | 0.086 |
| Privileged factor diagnostic* | B | 0.0056 | -4.663 | 0.099 |
| Privileged factor diagnostic* | D | 0.0070 | -4.300 | 0.140 |
| Privileged factor diagnostic* | C | 0.0219 | -0.134 | 0.425 |
| Oracle expected-transition reference | A | 0.00063 | — | 0.015 |
| Oracle expected-transition reference | B | 0.00062 | — | 0.005 |
| Oracle expected-transition reference | D | 0.00071 | — | 0.014 |
| Oracle expected-transition reference | C | 0.00065 | — | 0.005 |

The main unsupervised composition has C MSE **0.0764**, versus **0.0668** for
independent experts and **0.0746** for the monolithic model. Its held-out gap is
**+0.0464**; experts' gap is **+0.0274**; monolithic gap is **+0.0340**. Learned
composition therefore does **not** outperform either learned baseline on C in
this run, and its five-step C error (1.283) is worse than experts (1.105).

*The privileged diagnostic is not the proposed method. It trains with explicit
A/B/D labels mapped to bit-codes and evaluates with the true code, including
`11` for C. Its much better C MSE (0.0219) indicates the executor basis can
represent/use the intended factorization when assignments are supplied; the
large gap from the main model is evidence that latent factor assignment is not
identified reliably by the current unsupervised marginal-likelihood objective.
The diagnostic is an oracle-labeled upper bound only, not an eligible result for
the method claim.

The oracle reference reports only expected-transition MSE/rollout error. It
does not report NLL: the benchmark's injected velocity noise also determines
position increments, so the full four-dimensional noise covariance is
singular, and this reference is not a learned distribution model.

## Training curves and artifacts

`training_transitions.npz` persists the shared, blinded train and validation
arrays only. `results/rwt/stage2a/training_curves.png` and `training_curves.csv` contain
training and A/B/D validation NLL, including the labeled diagnostic on a separate
curve. `per_regime_metrics.csv` contains per-regime MSE, NLL, sample SD and
rollout errors. `evaluation_trace.csv` and
`privileged_factor_diagnostic.csv` keep ordinary and privileged evaluations
separate. Checkpoints and the run config are saved beside the metrics.

Run the small CPU configuration from the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
python experiments/rwt/run_stage2a.py --output-dir results/rwt/stage2a `
  --epochs 80 --hidden-dim 48 --batch-size 256 --device cpu
```

## Scope and limitations

The current data identify three predictive clusters, but do not uniquely
identify their decomposition into named mechanisms: the two primitive slots
can permute, the anchor/base choice is structurally ambiguous, and marginal
likelihood can allocate transition regions to arbitrary latent codes. The
privileged-label diagnostic isolates this assignment issue but does not solve
it. The current held-out C result is therefore evidence of a functioning
learned transition pipeline, not evidence that unsupervised theory induction
has recovered reusable causal primitives. Additional non-privileged
identifiability constraints or richer training compositions are open work.

No Stage 2B policy learning, model-based control, replay system or Stage 3 work
is included.
