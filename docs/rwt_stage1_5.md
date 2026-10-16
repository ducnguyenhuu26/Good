# RWT Stage 1.5 — Identification Validation

Stage 1.5 validates the Stage 1 oracle-model diagnostic. It does not train a
transition model, alter NEO, or implement Stage 2.

## Inference review

At action time, a policy uses its posterior over candidate dynamics. The filter
then processes the observed transition in this order:

\[
p_t^-=(1-h)p_{t-1}+h p_0,
\qquad
p_t(i)\propto p_t^-(i)\,
\mathcal{N}(v_{t+1};\mu_i(s_t,a_t),\sigma^2 I).
\]

Here \(p_0\) is uniform over the oracle candidate library and \(h\) is the
configured hazard rate. This small reset component lets a previously unlikely
or returning candidate recover after a regime change. It is a prior over
candidate dynamics, not a simulator change signal. The update occurs only after
`env.step(action)` returns `next_state`; the reward is not used by the filter.

The likelihood uses next velocity. Position is the deterministic sum
\(x_{t+1}=x_t+v_{t+1}\), so it contributes no independent noise observation.
The model assumes known, isotropic Gaussian velocity noise with configured
standard deviation. It does not estimate noise online.

Expected information gain uses the same hazard predictive prior and estimates
\(I(\tau;v_{t+1}\mid s_t,a_t)\) by Monte Carlo. Candidate actions share common
normal noise samples to reduce ranking variance. During a fixed, evenly spaced
probe budget, pure-IG chooses the feasible action with the highest gain;
belief-active chooses the highest expected one-step task reward plus
\(\beta\,I\). Outside those probe slots, all probing approaches use the same
feedback controller. Random probing samples uniformly from the same finite
feasible action set. Thus probing count and maximum action norm are matched.

Pairwise transition KL has a closed form under the shared covariance:

\[
D_{KL}(i\Vert j\mid s,a)
=\frac{\|\mu_i(s,a)-\mu_j(s,a)\|_2^2}{2\sigma^2}.
\]

At zero velocity, friction-only differences cannot be detected from a single
transition. With non-zero action, gain and rotation become distinguishable;
with non-zero velocity, friction becomes visible even when action is zero.
`pairwise_kl.csv` records every candidate pair at rest/moving states and a set of
bounded actions. `probe_action_rankings.csv` ranks feasible actions by average
and weakest-pair KL. `results.csv` also reports KL separation under actions
actually taken by each policy.

## Environment safeguards

All policies share the same 2D action box and Euclidean action-norm cap. The
environment projects any in-box action outside the norm cap back onto the
boundary. The transition equation and task reward are unchanged. A finite state
whose norm exceeds the configured limit ends the episode and is counted as an
unstable termination; a non-finite state raises `FloatingPointError`. Rewards
are not clipped. The evaluator sees regime labels and change points separately;
the observation and `info` contain no such metadata.

## Run

```bash
python experiments/rwt/run_stage1.py \
  --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61 \
  --probing-budget 24 --max-action-norm 0.75 --hazard-rate 0.05 \
  --beta-sweep 0,0.25,1,4
```

The script requires at least 20 seeds. It writes `results.csv` (one row per
seed/policy), `summary.csv` (means and sample standard deviations),
`trajectories.csv` (step-level evaluator traces), `pairwise_kl.csv`,
`probe_action_rankings.csv`, and two PNG plots under
`results/rwt/stage1_5/`. Dashed vertical lines in the timeline plot show the
mean time of each change point across seeds. Use `--skip-plots` for a
CSV-only run.

The oracle candidate library includes the exact C composition for diagnosis,
even though C is absent from the training schedule. Results therefore measure
online identification with oracle candidates; they are not evidence that a
learned model can compose unseen mechanisms.

## Validation results (20 seeds)

Configuration: seeds 42–61, noise standard deviation 0.025, 24 evenly spaced
probe steps per episode, Euclidean action norm at most 0.75, hazard 0.05,
belief-active beta sweep 0/0.25/1/4. All 140 runs used their full 24-step probe
budget; none hit the finite-state termination threshold. Values below are
mean ± sample standard deviation across seeds. Regime accuracy is pooled within
regime across each episode before seed aggregation.

| Policy | Return | Identification | Delay (steps) | Entropy (nats) | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Passive Bayes | −38.47 ± 2.31 | 47.9% ± 8.4% | 11.54 ± 1.93 | 0.988 ± 0.102 | 84.7% | 18.6% | 14.8% | 64.9% |
| Random probe | −79.07 ± 20.86 | 92.5% ± 3.4% | 1.37 ± 0.54 | 0.131 ± 0.031 | 93.7% | 93.3% | 84.0% | 96.8% |
| Pure information gain | −83.61 ± 21.44 | 96.6% ± 2.3% | 0.63 ± 0.42 | 0.066 ± 0.013 | 96.4% | 96.5% | 94.3% | 99.0% |
| Belief-active, beta 0.25 | −40.82 ± 2.36 | 85.4% ± 6.1% | 2.65 ± 1.15 | 0.241 ± 0.033 | 89.6% | 84.3% | 73.4% | 90.6% |
| Belief-active, beta 1 | −41.43 ± 2.30 | 88.8% ± 4.7% | 1.99 ± 0.97 | 0.203 ± 0.039 | 92.9% | 84.9% | 81.9% | 94.4% |
| Belief-active, beta 4 | −42.21 ± 2.66 | 88.8% ± 5.1% | 2.04 ± 0.92 | 0.189 ± 0.032 | 92.6% | 85.6% | 82.7% | 92.2% |

Pure information gain gives the highest identification score and lowest delay,
with worse return than random probing. Beta 1 improves identification over
passive Bayes while losing about 3 return points on average. Raising beta from
1 to 4 did not improve aggregate identification in this sweep. These results
show a trade-off; they do not establish one active policy as universally better.

Pairwise KL confirms that action choice matters. At rest, friction-only pairs
have KL 0 under any action because their velocity is zero. For the moving
diagnostic state \(v=(1,-0.5)\), the diagonal action \((0.530,0.530)\) has the
largest mean pairwise KL (449.84 nats) and the strongest weakest-pair KL
(24.32 nats) among the tested actions. Under actual policy actions, mean
truth-to-alternative KL was 32.45 for passive, 89.12 for random, 151.29 for
pure-IG, and 46.68 for beta 1. The realized-action KL summaries are in
`summary.csv`; step-level values are in `trajectories.csv`.

The original Stage 1 three-seed run reported passive −32.16 return / 34.3%
identification, random −54,988 / 80.6%, and reward-plus-information −29.65 /
37.0%. Stage 1.5 changes the experiment: it adds a transition hazard, caps
action norm, allocates a matched 24-step probe budget, and terminates extreme
states. The before/after figures are therefore descriptive, not a paired causal
estimate of the hazard prior alone. `results/rwt/stage1/results.csv` preserves
the Stage 1 rows.
