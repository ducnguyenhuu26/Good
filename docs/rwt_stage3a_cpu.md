# RWT Stage 3A CPU: HalfCheetah mini prototype

## Scope and implementation

This is an end-to-end **pipeline smoke test**, not a comparative benchmark or
evidence of robust model-based control. The upstream NEO implementation and
Stage 1–2C artifacts were left intact.

`RecurrentHalfCheetah` wraps Gymnasium `HalfCheetah-v5`. The installed version
is Gymnasium 1.3.0 with MuJoCo 3.3.7. Observations are 17D; actions are 6D,
bounded by `[-1,1]` in every coordinate. Reset with a seed and deterministic
action sequence replays exactly in the CPU unit test. The environment did not
set `terminated` in the short tested sequences; the wrapper sets `truncated`
when its configured schedule ends. A separate direct Gymnasium check reached
the default 1,000-step limit with `(terminated=False, truncated=True)`.

The wrapper stores exact copies of `geom_friction` and `actuator_gainprm` and
restores them before each reset and on close. Regimes multiply those parameters:

| Regime | Friction multiplier | Motor gain multiplier |
|---|---:|---:|
| A | 1.0 | 1.0 |
| B | 0.6 | 1.0 |
| D | 1.0 | 0.7 |
| C | 0.6 | 0.7 |

The fixed-gain motor actuator's `actuator_gainprm[:,0]` is modified; the
action bounds do not change. A unit test confirmed that B and D produce
different trajectories from A under the same seed and actions. The wrapper
switches physics after the final transition of each dwell period. Regime
properties are read by the evaluator, never inserted in observations, rewards,
`info`, training arrays, adapter updates or MPC inputs.

The offline collector uses separate seeds/episodes for train and validation.
It saved **1,800 train plus 200 validation transitions for each of A, B, D**;
C has no offline file. NPZ fields are `state`, `action`, `next_state`,
`reward`, `terminated`, `truncated`. Checkpoint selection uses pooled A/B/D
validation only. The manifest records regime/seed metadata separately.

The MLP has two SiLU hidden layers of width 64 and predicts state delta and
scalar reward from `(s,a)`. Train-set-only normalization is saved in the
checkpoint. The loss is

`L = mean[((Δ_hat-Δ)/σ_Δ)^2] + mean[((r_hat-r)/σ_r)^2]`.

The state/reward prediction interface is `model.predict(state, action)`. The
online adapter is dimension-independent RLS with features `[1, normalized(s),
normalized(a)]`; for 17D/6D it has `(1+17+6)*17 = 408` trainable weights.
It updates only after observing the next state. The shared MLP remains fixed
through online evaluation.

MPC-CEM rolls the learned next-state and learned reward heads forward. It uses
horizon 3, 32 action sequences, two CEM iterations and clipped valid actions.
It never calls MuJoCo while planning. The CPU evaluation schedule was
`A12→B12→D12→C16→A12→C16`, 80 transitions per seed; this external schedule
truncates each test episode. Seeds were 701 and 702.

## CPU run

Command from repository root (PowerShell):

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES=''
python experiments/rwt/run_stage3a.py --config experiments/rwt/configs/stage3a/cpu_smoke.yaml --device cpu
```

The run completed with PyTorch 2.13.0+cpu. It collected 6,000 transitions at
about **16,531 transitions/s** including NPZ writes, trained for five epochs
in **0.299 s**, saved and reloaded the best checkpoint, and evaluated two
short episodes. The finite train/validation loss curves decreased from
`1.831/1.560` in epoch 1 to `0.659/0.536` in epoch 5. Best epoch was 5.
Offline A/B/D validation next-state MSE was **3.5011** in raw observation
units, and reward MSE was **0.01243**.

| Online metric | CPU smoke result |
|---|---:|
| Pre-update next-state MSE, adapter active | 4.2876 |
| Pre-update next-state MSE, frozen MLP on same transitions | 11.8354 |
| MSE on A / B / D / C transitions, adapter active | 8.7500 / 6.2338 / 1.6585 / 1.1968 |
| Return, seed 701 / 702 | −18.7773 / 2.4442 |
| Mean return over two 80-step episodes | −8.1666 |
| MPC action computation, mean / p95 | 2.89 / 3.23 ms |
| Online adapter updates / parameters | 80 per episode / 408 |

The large MSE and variable return show that this small model/controller is
only a functioning prototype. The MPC action distribution differs from the
smooth random behavior used for offline collection; one seed had a
pre-update step MSE above 47. This test does not establish control quality,
generalization to C, structural identification, recurrence memory, or a
ranking against any other policy. The single online RLS adapter is reset for
each episode and does not implement Stage 2C theory memory.

## Artifacts and checks

- `results/rwt/stage3a_cpu_smoke/data/`: six A/B/D train/validation NPZ files.
- `results/rwt/stage3a_cpu_smoke/data_manifest.json` and
  `resolved_config.json`: collection seeds, splits, versions and configuration.
- `results/rwt/stage3a_cpu_smoke/checkpoints/best_dynamics.pt`: MLP and
  normalization buffers, selected using A/B/D validation.
- `results/rwt/stage3a_cpu_smoke/training_curve.csv`, `episodes.csv`,
  `evaluation_steps.csv`, `metrics.json`: learning and time-series metrics.

Five Stage 3A unit tests cover schedule switching, exact physics restoration,
deterministic replay, physical perturbation effect, model gradients and
checkpoint round-trip, arbitrary-dimensional RLS, MPC rollout shapes/bounds,
and offline C rejection. The relevant regression suite passes 60 tests in
total. No GPU or full benchmark was run.

## Prepared GPU run

Install the extra dependencies if needed: `pip install -e ".[rwt-mujoco,test]"`.
The larger config collects 90,000 train plus 10,000 validation transitions per
A/B/D regime, trains for 60 epochs, uses horizon 5 with 64 candidates, and
evaluates 20 short schedule seeds. It uses the same runner and preserves C
outside offline training/validation. **This command is prepared but untested
on CUDA and was not executed:**

```powershell
$env:PYTHONPATH='src'
python experiments/rwt/run_stage3a.py --config experiments/rwt/configs/stage3a/gpu_full.yaml --device cuda
```

MuJoCo data collection still runs on CPU. CUDA applies to the MLP and MPC
rollouts; the 408-weight RLS update remains on CPU. A full comparative
benchmark would additionally need matched control/replay baselines and enough
seeds to support statistical claims.

## GPU path optimization after the CPU smoke run

The runner now preallocates offline NPZ arrays instead of appending one Python
object per transition. The full config collects independent A/B/D splits with
three spawned CPU workers. Training keeps loss accumulation on device and
checks numerical finiteness once per CUDA epoch, avoiding a host synchronization
for every minibatch. The serial MPC caches the adapter's device tensors until
the next observed update and scores candidates without storing intermediate
rollout states. The full config also plans for all active evaluation seeds in
one model batch: candidate rollouts have effective batch size up to
`20 seeds × 64 candidates = 1,280`, while each seed keeps an independent
planner random stream and RLS adapter. Environments still step independently
on CPU; the policy receives no regime metadata.

The CPU numbers above were recorded **before** this optimization pass. The
optimized runner and the CUDA path have not been benchmarked or experimentally
validated yet; no speedup or prediction-equivalence claim is made here. The
batched GPU planner samples normals from NumPy's per-seed generators, so its
action trajectories should not be compared bit-for-bit with the serial
PyTorch-generator CPU smoke run.

The local `results/` tree remains available for prior-stage reproducibility.
It is excluded by `.gitignore` so a new source repository does not accidentally
include large generated traces and checkpoints.
