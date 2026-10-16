# RWT Stage 2C — Online theory revision and recurrence memory

Stage 2C audits Stage 2B, adds a calibrated mismatch detector and small online
residual adapters, and measures whether a predictor adapted on first C can be
retrieved when C returns. All runs are CPU and use the existing Stage 2B
A/B/D-only learned checkpoints. No Stage 2A/2B result files or NEO modules were
modified; Stage 2C results are in `results/rwt/stage2c/`.

## Stage 2B audit

Stage 2B's “identification accuracy” is a **label-based** evaluator metric:
the posterior MAP code is compared with an evaluator-only alignment. Expert
permutation is aligned using A/B/D probe points only. The completion target
index for C is then assigned from the known additive-completion construction;
this is scoring metadata, never an input to inference. This metric alone did
not show whether the selected theory best predicted the actual transition.
Stage 2C therefore adds **behavioral MAP fit**: whether the posterior-MAP
candidate is also the candidate with the smallest observed one-step squared
velocity residual. It is a prediction-fit diagnostic, not semantic identity.

Across the 20-seed replay, learned-composition passive results were:

| Regime | Label-based accuracy | Behavioral MAP fit | MAP candidate MSE | Best candidate MSE | Aligned candidate MSE |
|---|---:|---:|---:|---:|---:|
| A | .280 | .461 | .00236 | .00103 | .00297 |
| B | .000 | .425 | .00228 | .00085 | .00298 |
| D | .508 | .489 | .04223 | .02523 | .03950 |
| C | .089 | .389 | .00540 | .00260 | .00954 |

The learned C completion is not consistently the label-aligned model, and is
not consistently the candidate with the best observed prediction either.
Nevertheless, the mixture's C MSE is relatively small. This separates a
structural-identification failure from a complete inability to predict C.

The candidate means are weakly separated relative to their predictive scales:
mean pairwise noise-normalized distance is .610 on C (mean pairwise velocity
distance .117); the minimum pairwise distance averages .059. This supports a
model-indistinguishability limit in the visited state-action support. It does
not prove every state-action is indistinguishable.

The Gaussian uncertainty is also miscalibrated by regime. For the aligned
candidate, C coordinate-wise 68% and 95% coverage are .946 and .991, while D
coverage is .456 and .637. Thus uncertainty is too wide on C and too narrow on
D. The C aligned candidate's standardized residual RMS is .50 versus 1.80 on
D. This is compatible with D model error and weak C separation; it is not
evidence for a missing C theory by itself.

The Stage 2B cache **was** present in the recurrence filter. On the first C
return, the aligned candidate was in memory in 95% of runs, but its mean prior
mass was only .0135. With hazard .05, memory support was diluted by the
posterior and other theories. This is a prior-weight limitation rather than a
missing-cache-code bug. The posterior target mass stayed near .004 on the
return segment, so candidate availability did not produce reliable retrieval.

Stage 2A and Stage 2B MSE values are not direct paired comparisons. Stage 2A
used six evaluation seeds, varied schedule dwell times and random actions;
Stage 2B used 20 seeds, fixed 256-step segment lengths and a target controller
with optional probes. Both report mean full-state squared error, but the
state-action support differs. Their NLLs are even less comparable: Stage 2A
scores a four-dimensional next-state distribution; Stage 2B scores velocity
only to avoid double-counting the deterministic position increment.

## Revision model

The Stage 2B learned backbone is frozen. For a candidate velocity predictor
\(\mu_j(s,a)\), the online adapter predicts

\[
\hat v_{t+1}^{(j)}=\mu_j(s_t,a_t)+\phi(s_t,a_t)^T W_j,
\quad \phi=[1,v_x,v_y,a_x,a_y]^T,\quad W_j\in\mathbb R^{5\times2}.
\]

Thus one residual adapter has 10 scalar weights. Its RLS update is

\[
K_t=\frac{P_t\phi_t}{1+\phi_t^TP_t\phi_t},\qquad
W_{t+1}=W_t+K_t r_t^T,\qquad
P_{t+1}=P_t-K_t\phi_t^TP_t,
\]

where \(r_t=v_{t+1}-\mu_j(s_t,a_t)-\phi_t^TW_t\) is formed only after the
transition arrives. The frozen network and its original predictions remain
available. Adapter covariance contributes a separate epistemic term to the
predictive scale; candidate disagreement is reported separately from the
base model's aleatoric scale. In the persistent arm, each adapter is stored
under its unlabelled base-candidate index with its flattened residual-weight
signature and RLS covariance; retrieval scores the resulting predictive
distribution before deciding whether to use that stored revision.

Two update paths are tested. `compositional_revision` and its memory ablation
create a residual theory only after at least two standardized surprises in a
three-transition window. The cutoff is the 99.5th percentile of the minimum
candidate standardized residual on A/B/D validation (2.762 for composition,
4.142 for experts). It does not use C. `compositional_streaming_revision`
updates a 10-weight adapter after each observed transition on the best-fitting
candidate; the no-memory form discards that adapter when the selected theory
changes. The memory form retains a separate adapter per selected predictive
candidate. Its online input contains only state, action and observed next
state; candidate selection, memory and updates receive no regime ID or
change-point schedule.

`monolithic_full_finetune` updates all 16,068 model parameters once per
observed transition. `sliding_window` freezes the same monolithic backbone and
refits a 10-weight RLS correction on the latest 24 transitions. Independent
experts use the Stage 2B expert bank with the mismatch-triggered memory path.
All approaches receive the same 256 environment transitions and common action
stream per seed. The stream is generated by a fixed controller, so returns
are identical by design and are not a control-quality comparison.

## 20-seed results

MSE is full-state pre-update MSE, NLL is pre-update velocity NLL. Values are
mean ± SD. The C label score is omitted for methods with no C completion label.

| Method | C MSE | First-C MSE | Returning-C MSE | C NLL | C label accuracy | C behavioral MAP fit | Updates / online parameter budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| Composition frozen | .00691 ± .00156 | .01195 | .00187 | −2.296 | .091 | .388 | 0 / 0 |
| Composition + mismatch revision | .00691 ± .00156 | .01195 | .00187 | −2.296 | .091 | .388 | 0 / 10 |
| Composition + mismatch revision + memory | .00691 ± .00156 | .01195 | .00187 | −2.296 | .091 | .388 | 0 / 10 |
| Composition + streaming adapter, no memory | .00670 ± .00139 | .01165 | .00176 | −2.307 | .092 | .407 | 256 / 10 |
| Composition + streaming adapter memory | .00497 ± .00129 | .00897 | .00098 | −2.461 | .075 | .488 | 256 / 10 per adapter |
| Independent experts + mismatch memory | .00575 ± .00104 | .00975 | .00174 | −2.245 | n/a | .515 | 0 / 10 |
| Sliding-window adapter | .00622 ± .00148 | .01046 | .00198 | −2.204 | n/a | n/a | 256 / 10 |
| Monolithic full finetuning | .00425 ± .00063 | .00712 | .00137 | −2.476 | n/a | n/a | 256 / 16,068 |

Paired seed results:

- Streaming adapter memory reduces C MSE against frozen composition by .00193
  (95% paired bootstrap CI [.00144, .00253]); C NLL also improves by .165
  (baseline-minus-memory CI [.154, .177]). It improves against the matched
  10-weight sliding-window adapter by .00125 C MSE (CI [.00039, .00211]).
- It does not beat full monolithic finetuning on all-C MSE: the paired
  memory-minus-finetune difference is +.00073 (CI [.00018, .00126]). It does
  use far fewer online parameters and compute: 10 per adapter, about 5.9
  adapters stored at episode end (about 59 weights), versus 16,068 updated
  parameters; mean CPU time is .412 versus 1.388 seconds per episode. The
  paired compute difference is −.976 seconds (CI [−1.015, −.941]).
- On returning C, streaming memory's MSE is .00098 versus .00137 for full
  finetuning; the paired memory-minus-finetune difference is −.00040
  (CI [−.00050, −.00031]). On A recurrence, its MSE is .00096 and its delta
  from initial A is −.00060, so no A prediction forgetting appears under this
  fixed stream.
- The mismatch-triggered path generated zero revisions on C and had zero
  known-regime false alarms. This is because the candidate bank's uncertainty
  covers the observed C transitions under the validation-calibrated rule. This
  is a negative detection result, not a successful mismatch detector.

`model_recovery_delay_recurrence` is an evaluator-only sample count: the first
four-transition window whose pre-update MSE is within 25% of the final eight
first-C predictions. Its mean is 4.10 for frozen composition and 3.45 for
streaming memory; paired CI for frozen-minus-memory is [−2.05, 3.85], so this
small difference is not resolved. Frozen models can meet this threshold too;
it is a predictive recovery criterion, not evidence that an adapter update
caused the recovery. C detection delay for mismatch-triggered revision is
capped at 48 transitions in both C appearances because it never alarms.

### What “memory retrieval” means here

The streaming memory arm selected a candidate adapter that had been updated
during first C at the first transition of returning C in all 20 seeds. Its
predictive retrieval delay is therefore 1 transition. But evaluator alignment
shows the selected candidate was **not** the designated C completion: exact
completion selection was 5.2% on first C and 0% on returning C, and posterior
MAP label accuracy on returning C was 0%. The method reuses a predictive
residual adapter, but does not recover the correct latent structural theory.
That is a behavioral recurrence result, not structural identification.

The C-adapted candidate is selected on some A/B/D transitions too; evaluator
false-retrieval rate on known regimes is .25 for the streaming-memory arm.
Despite that, A/B/D prediction error does not rise on this shared test stream:
MSE for A/B/D is .00156/.00118/.03453 versus .00188/.00155/.04778 frozen.
This does not rule out forgetting on other state-action support.

## Close-regime control

The separate controlled test uses two fixed Gaussian predictive theories with
mean separation set to .20 or .60 predictive standard deviations, equal prior,
and 20 independent sequences of up to 32 observed transitions. It is an
identification-limit diagnostic, not RWT environment training or a tuned
hyperparameter experiment.

| Mean gap / predictive SD | Correct theory after 8 | After 16 | After 32 | Mean target posterior after 32 |
|---:|---:|---:|---:|---:|
| .20 | .60 | .70 | .80 | .630 |
| .60 | .55 | .85 | .95 | .906 |

When predictive distributions overlap, repeated observations help but do not
guarantee rapid identification. This is consistent with the low pairwise
separation and weak label accuracy in the learned C bank.

## Completion assessment and remaining issues

Stage 2C demonstrates an online learned residual that improves C prediction
over frozen composition and a matched-parameter sliding-window correction,
without updating the shared backbone. The persistent adapter is retrieved on
the first transition when C returns and improves recurrence prediction.
However, the retrieved candidate has the wrong evaluator-aligned completion
code, and mismatch-triggered new-theory creation does not activate on C. Thus
the prediction/reuse criterion is met behaviorally, but reliable **structural**
theory identification is not. Monolithic full finetuning remains best on
average C MSE when allowed to update 16k parameters. No model-based policy
training, MuJoCo or NEO upstream changes were made.

Reproduce from the repository root (CPU):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONPATH='src'
python experiments/rwt/run_stage2c.py --output-dir results/rwt/stage2c `
  --seeds 3300,3301,3302,3303,3304,3305,3306,3307,3308,3309,3310,3311,3312,3313,3314,3315,3316,3317,3318,3319
```

`summary.csv` and `paired_comparisons.csv` provide aggregate mean, SD and
bootstrap confidence intervals. `online_traces.csv`, `adaptation_curves.csv`,
`stage2b_audit_*`, `stage2b_pairwise_*`, `stage2b_recurrence_posterior.csv`,
`recurrence_memory_audit.csv`, `mismatch_calibration.csv` and
`near_regime_control.csv` retain the per-transition/evaluator evidence.
