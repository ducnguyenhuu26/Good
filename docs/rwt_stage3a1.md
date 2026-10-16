# RWT Stage 3A.1 — Final CPU preflight

This is a correctness and numerical sanity pass for the existing 17D/6D
HalfCheetah-v5 prototype. It does not implement Stage 3B or claim that the
model/controller is competitive. All offline fitting and checkpoint selection
used A/B/D only. The physics audit and other diagnostics involving C ran after
the selected MLP checkpoint was fixed.

## Dataset and environment audit

The existing Stage 3A dataset contains exactly six NPZ files: 1,800 train and
200 validation transitions for each of A, B, D. Train seeds were
`31415/32415/33415`; validation seeds were `131415/132415/133415`. The split
seed sets are disjoint, and hashes of all transition rows found **zero exact
train/validation overlaps**. The audit checks manifest split identities and
counts regardless of entry order, plus NPZ SHA-256 when present. There is no offline C NPZ, C target in model
selection, or C observation in input/delta/reward normalization. Each MLP
checkpoint's normalization buffers were compared with statistics recomputed
from the pooled train arrays only. NPZ files contain transitions, rewards and
termination flags, with no regime or physics fields.

For 12 matched physical `(qpos, qvel, action)` probes, the mean L2 distance
between next observations was:

| Regime pair | Mean next-state L2 |
|---|---:|
| A–B | 0.3595 |
| A–D | 2.9823 |
| A–C | 2.8879 |
| B–C | 2.9277 |
| D–C | 0.3239 |

All perturbations therefore changed dynamics on this probe set. Reset
restored the configured friction and motor gain for every regime. The
12-transition `A2→B2→D2→C2→A2→C2` test replayed exactly from the same seed.
The controller receives the 17D observation and a 6D bounded action space;
regime labels, physics values, change flags and simulator state are evaluator
only. The simulator state clone used below is never given to a model or MPC.

## One-step dynamics and reward

The persistence predictor uses `s' = s`. Ridge fits normalized state/action to
normalized delta with a fixed ridge penalty of 1.0. All means and scales come
from A/B/D train data. The MLP has two width-64 hidden layers and a learned
reward head. Three CPU seeds ran for at most 35 epochs with patience 7 and
checkpoint selection by pooled A/B/D validation loss. All three reached epoch
35 while still improving; the early-stop rule was present but did not trigger.
No hyperparameter or checkpoint was chosen using C.

| Regime | Predictor | Raw next-state MSE | Train-scale normalized MSE |
|---|---|---:|---:|
| A | Persistence | 8.4550 | 1.2011 |
| A | Linear ridge | 0.4842 | 0.1041 |
| A | MLP, mean of 3 seeds | 0.5287 | 0.0994 |
| B | Persistence | 8.7290 | 1.2391 |
| B | Linear ridge | 0.4970 | 0.1081 |
| B | MLP, mean of 3 seeds | 0.5777 | 0.1083 |
| D | Persistence | 3.5703 | 0.5260 |
| D | Linear ridge | 0.2873 | 0.0559 |
| D | MLP, mean of 3 seeds | 0.2846 | 0.0520 |

Across equal 200-transition validation regimes, raw MSE was **6.9181** for
persistence, **0.4228** for ridge and **0.4637** for MLP; normalized MSE was
**0.9887 / 0.0894 / 0.0866**, respectively. Ridge has lower raw MSE than the
MLP on this split, while the MLP has slightly lower normalized MSE. This is a
negative result for a blanket nonlinear-model advantage. The 17 per-dimension
results are in `per_dimension.csv`; dimensions 13–16 were among the largest
raw MLP errors on A/B.

MLP validation loss for seeds 41/42/43 was **0.11348 / 0.11477 / 0.11083**
(sample SD **0.00201**). Corresponding pooled raw next-state MSE was
**0.4773 / 0.4663 / 0.4475**. Training losses and gradients stayed finite.
Seed 43 was selected for subsequent diagnostics solely by A/B/D validation.
The reward head's validation MSE averaged **0.00586 / 0.00439 / 0.00342** on
A/B/D, or **0.00456** pooled across regimes/seeds.

## Multi-step evaluator rollouts

For three fixed simulator initial states and fixed action sequences, an
evaluator cloned `(qpos, qvel)` into each regime simulator, then compared
free-running model predictions with MuJoCo at endpoint horizons 1/3/5/10.
The table shows mean raw next-state MSE across the three initial seeds. C is
held-out evaluation only.

| Regime | Predictor | h=1 | h=3 | h=5 | h=10 |
|---|---|---:|---:|---:|---:|
| A | Ridge | 0.527 | 2.302 | 0.720 | 0.615 |
| A | MLP | 0.441 | 2.097 | 0.803 | 1.013 |
| B | Ridge | 0.527 | 1.161 | 1.273 | 1.065 |
| B | MLP | 0.441 | 1.200 | 1.371 | 1.599 |
| D | Ridge | 0.339 | 2.326 | 0.808 | 0.748 |
| D | MLP | 0.395 | 2.324 | 0.817 | 0.822 |
| C | Ridge | 0.339 | 1.726 | 0.485 | 0.481 |
| C | MLP | 0.395 | 1.856 | 0.782 | 0.627 |

Endpoint error is non-monotonic with horizon because states oscillate; these
are not cumulative trajectory averages. Persistence results are in
`rollout.csv`. All predicted states remained finite; the maximum absolute
predicted coordinate was **10.593**. These small fixed probes do not establish
long-horizon stability.

## Reward and MPC sanity

Zero action, uniformly random action and learned-model MPC were evaluated on
the same two initial seeds, each with a 48-step
`A8→B8→D8→C8→A8→C8` schedule. All actions satisfied `[-1,1]^6`. Predicted
episode return below is the sum of one-step reward predictions at the actual
observed states and chosen actions; it is comparable across policies and is
distinct from MPC's planned 3-step value.

| Policy | Predicted return | Actual return | One-step state MSE | Mean action L2 | MPC/action |
|---|---:|---:|---:|---:|---:|
| Zero | 0.192 | 0.284 | 0.033 | 0.000 | — |
| Random | −17.668 | −18.314 | 1.334 | 1.371 | — |
| MPC-CEM | 1.164 | −1.802 | 1.665 | 1.547 | 3.45 ms |

Individual MPC actual returns were **−9.45** and **5.84**. It did not reliably
beat zero action in this smoke test. MPC overpredicted its teacher-forced
episode return by about **2.97** on average. The diagnostic warns when a
selected predicted plan leaves a 6-train-SD support envelope in any state
dimension: this happened in one of the two MPC episodes (maximum flagged plan
fraction 1/3). There were no nonfinite candidate values. The planner now
records this warning and substitutes a zero action if every CEM candidate has
a nonfinite predicted value; it does not clip model states or rewards.

## Adapter and recurrence diagnostics

The 17D RLS adapter updated from observed transitions only. We checked finite
weights/covariance and nonnegative covariance eigenvalues after 20 updates.
The memory was saved as NPZ, reloaded without pickle, and its predictive
signatures matched the stored adapter weights. Retrieval uses three **past
observed** transitions and evaluates the next five transitions before any
new adapter update. It receives no regime label. Snapshots of two adapters
after the first A and C segments used evaluator-known segment boundaries;
this is a **diagnostic with privileged segmentation**, not a demonstrated
online change detector.

| Returning segment | Retrieved index | Fresh MSE | Retrieved MSE |
|---|---:|---:|---:|
| A | 0 | 0.9918 | 0.9116 |
| C | 1 | 0.4722 | 0.3677 |

The two retrievals matched the evaluator's segment identity (2/2), and the
stored signatures differed by L2 **6.737**. This is far too little evidence
for reliable recurrence reuse. Separate fixed 16-transition probe sets for
A/B/D/C expose a limitation: memory index 0 **worsened** A probe MSE from
**1.041** to **1.419**, while index 1 improved C probe MSE from **0.290** to
**0.235**. The retrieval diagnostic's gain is therefore context-specific,
and no online retraining/tuning was applied to force it.

## Corrections, artifacts and GPU readiness

This pass added a finite/positive RLS denominator check and finite parameter
check; an MPC invalid-candidate fallback and out-of-support plan warning; and
checkpoint resume with optimizer, epoch and RNG states. It also corrected a
resume reporting issue that otherwise overwrote original collection/training
timing with null/zero. New collection manifests include SHA-256 of each NPZ
and verify those hashes on resume. The preflight auditor also verifies these
hashes if present; the earlier Stage 3A dataset predates the hash field. The optimized Stage 3A CPU runner was
re-run in a separate output directory; its prediction MSE and return matched
the original CPU smoke run exactly. Resume reused the dataset and checkpoint
without training again. The batched multi-seed MPC path completed a small
CPU integration check. CUDA execution remains unverified.

Preflight outputs are under `results/rwt/stage3a1_cpu_preflight/`:
`summary.json`, `one_step.csv`, `per_dimension.csv`, `reward_validation.csv`,
`training_curves.csv`, `training_seeds.csv`, `rollout.csv`, `mpc_episodes.csv`,
`mpc_steps.csv`, `adapter_memory.npz`, `memory_retrieval.csv`,
`regime_probes.csv` and three A/B/D-selected checkpoints. Stage 1–3A
artifacts were not changed.

Three run configs remain separate:
`experiments/rwt/configs/stage3a/cpu_smoke.yaml`,
`experiments/rwt/configs/stage3a/gpu_pilot.yaml`, and
`experiments/rwt/configs/stage3a/gpu_full.yaml`. The local environment check
reported Python 3.11.9, PyTorch 2.13.0+cpu, Gymnasium 1.3.0, MuJoCo 3.3.7,
and `cuda_available=false`. On the GPU VM, from the repository root:

```powershell
pip install -e ".[rwt-mujoco,test]"
$env:PYTHONPATH='src'
python experiments/rwt/run_stage3a.py --check-gpu
python experiments/rwt/run_stage3a.py --config experiments/rwt/configs/stage3a/gpu_pilot.yaml --device cuda
```

If interrupted, rerun the last line with `--resume` and the same config,
device and output directory. `gpu_full.yaml` remains prepared but was not
run. The local CPU preflight command is:

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python experiments/rwt/run_stage3a1.py --config experiments/rwt/configs/stage3a/preflight_cpu.yaml
```

Final verification: **52 RWT unit tests passed**; `compileall` on `src/rwt`,
`experiments/rwt` and all RWT test files, plus `git diff --check`, completed
without diagnostics. The CUDA environment
check on this CPU machine reported unavailable, as expected. GPU pilot and
full training were not run.
