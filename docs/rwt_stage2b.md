# RWT Stage 2B — Online structural disambiguation

This stage tests whether online observations can resolve the latent ambiguity
found in Stage 2A.1. It does not train an RL policy, use C for offline
selection, or change upstream NEO.

## Candidate theories

Given three seen transition maps, the agent keeps all three unseen completions:

\[
G_1=F_2+F_3-F_1,\quad G_2=F_1+F_3-F_2,\quad G_3=F_1+F_2-F_3.
\]

The composition bank contains the three seen models and these three completion
hypotheses. There is no single latent assignment baked into the agent. In the
main branch the seen predictors are the three unlabeled experts from an
independent-experts checkpoint trained from A/B/D transition arrays only. The
expert ordering is arbitrary. The evaluator aligns expert indices to A/B/D
using A/B/D probe points only; since C is the B+D composition with A as anchor,
the scored C candidate is the completion omitting the aligned A expert. This
alignment is kept in evaluator output and is never passed to an agent.

The diagnostic branch uses accurate A/B/D dynamics to isolate filtering and
probing from model error. Its oracle information remains separate from the main
learned-model results.

## Online filtering, memory, and probing

Before action selection, the filter forms a change-point prior

\[
p_t^-=(1-h)p_{t-1}+h p_{base},\qquad h=0.05.
\]

After the environment returns the next state, it scores the observed next
velocity under each candidate Gaussian and updates

\[
p_t(i)=\operatorname{softmax}_i(\log p_t^-(i)+\log p(v_{t+1}\mid s_t,a_t,i)).
\]

Position is omitted from the likelihood because the simulator sets
\(x_{t+1}=x_t+v_{t+1}\); it is not an independent noise observation. Once a
candidate posterior crosses 0.8, its identity is added to theory memory. The
change prior then allocates 70% of its hazard mass across remembered theories
and 30% uniformly across the full bank. Models stay frozen; recurrence reuse
means restoring posterior support to the cached theory, without retraining.

Active probing estimates mutual information from predictive distributions:

\[
I(T;V'\mid s,a)=H[p^-]-\mathbb{E}_{V'\sim\sum_i p_i^-p_i(V')}H[p(T\mid V',s,a)].
\]

The active policy chooses the feasible action with maximum expected information
gain during its allocated probe slots. Random probing uses the same finite
Stage 1.5 action set; passive control uses none. All actions obey the same
Euclidean norm cap and all methods receive the same 24-slot budget ceiling.
Reward is unchanged and is never clipped.

## Split and protocol

Offline training uses 12 fixed A/B/D episodes (seeds 42–53); validation uses
A/B/D only (142–144). Learned independent experts and monolithic models see
the same transition arrays and use A/B/D validation marginal likelihood for
checkpoint selection. Neither model receives labels. C appears only in the
online evaluation schedule

\[
A\to B\to D\to C\to A\to C
\]

with fixed segment lengths and evaluation seeds 3200–3219. The evaluator
records ground truth after each agent action, but does not pass it to action
selection, filtering, memory, or model updates. The fixed test trajectory is
never used for checkpoint or hyperparameter choice. Prediction MSE/NLL are
scored before the corresponding posterior update. Checkpoints are 0, 1, 2, 4,
8, 16 and 32 observed C transitions at both first appearance and recurrence.

Monolithic and independent-expert controls use the same online Bayesian
inference and transition budget but have no C completion candidates. The
monolithic bank has one model and therefore zero theory information gain; its
active variant naturally declines to spend probes. The expert baseline can
distinguish its three seen experts but cannot name a novel composition.

## Results

### Aggregate online results (20 paired evaluation seeds)

Values below are mean ± SD across seeds; `C acc` is the proportion of C
transitions assigned to the evaluator-aligned completion. Delay is capped at
48 if the posterior never crosses the identification threshold during that C
segment. NaN for expert/monolithic C identification means those baselines do
not represent a compositional C hypothesis; it is not a zero score.

| Branch / model | Policy | Return | C acc | First / returning C delay | Probe actions / cost | C prediction MSE | C cached at return |
|---|---|---:|---:|---:|---:|---:|---:|
| Oracle diagnostic | Passive | −252.6 ± 107.2 | .940 | 1.0 / 8.35 | 0 / 0.00 | .00244 | 1.00 |
| Oracle diagnostic | Random | −655.1 ± 371.7 | .977 | 1.0 / 3.95 | 23.0 / 13.04 | .00402 | 1.00 |
| Oracle diagnostic | Active EIG | −414.7 ± 383.1 | .936 | 1.0 / 8.35 | 2.95 / 2.60 | .00277 | 1.00 |
| Learned composition | Passive | −252.6 ± 107.2 | .077 | 13.3 / 48.0 | 0 / 0.00 | .00651 | .80 |
| Learned composition | Random | −655.1 ± 371.7 | .378 | 5.5 / 22.55 | 23.0 / 13.04 | .02998 | 1.00 |
| Learned composition | Active EIG | −307.6 ± 192.6 | .080 | 15.7 / 48.0 | .40 / .47 | .00780 | .80 |
| Independent experts | Passive | −252.6 ± 107.2 | n/a | n/a | 0 / 0.00 | .00562 | n/a |
| Independent experts | Active EIG | −672.1 ± 227.2 | n/a | n/a | 16.55 / 12.65 | .02771 | n/a |
| Monolithic MLP | Passive / active | −252.6 ± 107.2 | n/a | n/a | 0 / 0.00 | .00697 | n/a |

Paired active-minus-passive for learned composition: C accuracy +.0031
(95% paired bootstrap CI [−.0052, .0120]), first-C delay +2.4 transitions
(CI [−.25, 6.85]), return −54.9 (CI [−109.8, −8.5]) and probing cost
+.47 (CI [.18, .79]). There is no measurable identification gain, while
return is lower. Against random probing, active has 0.298 fewer correct C
assignments (active-minus-random CI [−.352, −.246]) and identifies 10.2 transitions later on first C
(CI [2.10, 19.15]); it spends 12.57 fewer cost units and has return higher by
347.5 (CI [176.9, 535.3]). Random probing therefore buys identification
accuracy at a large reward and probe-cost penalty. The oracle diagnostic tells
a different story: even passive identifies first C in one transition, so
active cannot improve the first-identification delay; active is also worse
than passive on return by 162.0 paired units (CI [−311.3, −34.5]).

At first C appearance, the oracle posterior on the correct completion rises
from .003 before observing C to .961 after one transition (passive, mean).
For learned composition the corresponding mean target posterior is .187 before
C, .288 after one transition, peaks near .745 at eight transitions, then falls
to .012 by 32; hypothesis accuracy over the full C segments remains .077–.080
for passive/active. That fall and the 48-step recurrence delay show that high
transient posterior mass/cache insertion do not yet yield reliable structural
identification or theory reuse. Random probing lifts C accuracy to .378, but
its prediction MSE and return degrade sharply. Learned-composition false-novel
alarm rate on known-regime segments is .081–.085 for passive/active and .104
for random. A-regime point-prediction MSE is .00221 initially and .00143 on
recurrence for learned passive; recurrent A identification accuracy is zero.
This is posterior-label forgetting without corresponding point-prediction
degradation (the learned predictors are frozen online).

### Per-regime prediction and identification

Mean one-step velocity MSE and regime-ID accuracy (fraction correct) over each
regime's online segment:

| Model / policy | A MSE / acc | B MSE / acc | D MSE / acc | C MSE / acc |
|---|---:|---:|---:|---:|
| Learned composition / passive | .00182 / .291 | .00144 / .003 | .04541 / .533 | .00651 / .077 |
| Learned composition / random | .00561 / .474 | .00809 / .715 | .05253 / .360 | .02998 / .378 |
| Learned composition / active | .00207 / .293 | .00141 / .026 | .04437 / .536 | .00780 / .080 |
| Independent experts / passive | .00172 / .280 | .00139 / .003 | .03996 / .839 | .00562 / n/a |
| Independent experts / active | .01050 / .239 | .02429 / .890 | .04520 / .595 | .02771 / n/a |
| Monolithic MLP / passive | .00193 / n/a | .00185 / n/a | .11430 / n/a | .00697 / n/a |

For learned composition, mean posterior entropy (nats) was .528 passive,
.528 active and .630 random; the exact-model diagnostic gave .406, .409 and
.165 respectively. Random probing's lower diagnostic entropy reflects its
greater posterior concentration, but in the learned branch model uncertainty
and mismatch keep the posterior diffuse despite higher C assignment accuracy.
The full CSV summaries include SDs, confidence intervals, NLL and results for
all policies.

The summary CSVs provide mean, SD, and 95% bootstrap intervals for every
reported aggregate; `paired_comparisons.csv` contains paired seed deltas with
bootstrap intervals. The adaptation CSVs retain MSE, NLL, target posterior,
novel mass, and identification rates at 0/1/2/4/8/16/32 observed C transitions.
The plots show the adaptation curves with 95% intervals and mark both C
change points.

### Checks and interpretation

The environment completed all 256 interactions in every episode: there were no
unstable-state terminations; the single end-of-horizon event is recorded as a
time-limit truncation. No reward clipping or dynamics modification was used.
The 20-seed result does **not** support the claim that EIG active probing
reduces identification delay. In the learned branch it does not improve
identification over passive, and its policy-selected probes are sparse because
the current one-step predictive bank often assigns little marginal information
to feasible actions. Random exploration is more diagnostic here but incurs a
large control cost. The recurrence-memory criterion is implemented but not
validated as useful: caching occurs in 80% of learned-composition seeds, yet
the target is never re-identified at returning C under passive/active.

This is an evaluation of online filtering over a finite enumerated set of
three additive completions, not proof that the learned decomposition is
identifiable in general. For learned models, expert ordering is aligned using
A/B/D only at evaluation; the alignment and regime labels are not passed to the
agent. The oracle branch is diagnostic only. No inference path uses the
environment's oracle dynamics in the learned branch, and C did not enter
offline training, validation, checkpoint selection, or policy selection.

All raw step traces, paired seed summaries, adaptation curves, training
histories, fixed-seed config and checkpoints live under `results/rwt/stage2b/`.

## Interpretation and limits

The exact-model branch answers whether the candidate set and filter can
disambiguate a correctly represented composition. The learned branch tests
whether that result survives expert assignment ambiguity, predictive
uncertainty and model error. A posterior over the wrong or miscalibrated
completion is not evidence of successful identification. No model-based policy
training or MuJoCo evaluation is part of this stage.
