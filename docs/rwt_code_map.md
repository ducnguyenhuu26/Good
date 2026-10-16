# RWT Stage 1 code map

This note records the upstream interfaces inspected before adding the CPU
benchmark. Stage 1 is an independent package and does not change the original
NEO task experiments.

## Upstream reusable components

| Role | Source | Existing interface and relevance |
|---|---|---|
| Shared latent rollout | `src/models/neo.py` | `NEO.forward` owns encode/program/quantize/execute/decode rollouts and task objective hooks. |
| Theory programmer | `src/models/theory_programmer.py` | `TheoryProgrammer.forward(current_state, target_state)` returns continuous action tokens. It is goal-conditioned and has no environment action input. |
| Program executor | `src/models/program_execution.py` | `ProgramExecutor.forward(state, action)` applies one FiLM-conditioned residual latent update. |
| Primitive quantizer | `src/models/quantizer.py` | `ActionQuantizer.forward(actions, training_mode=...)` returns code indices, code vectors, and quantization losses. |
| Observation-pair contract | `src/data/episode.py` | `Example(input, output)` and `Episode(support, query)` provide domain-independent episode containers. |
| GridWorld primitives | `src/tasks/gridworld/data/primitives.py` | Four symbolic 2D grid movements; these are not continuous physical dynamics. |
| GridWorld data access | `src/tasks/gridworld/data/dataset.py` | HDF5-backed episode datasets and batch loaders. |
| GridWorld evaluation artifacts | `src/tasks/gridworld/theorizer_evaluation.py` | Defines ID, compositional OOD, and length OOD evaluation artifact contracts. |
| Reproduction entry points | `scripts/generate_data.py`, `scripts/train.py`, `scripts/evaluate.py` | Existing NEO data, train and evaluation commands. |
| Upstream checks | `tests/` | CPU-compatible unit and integration tests for NEO task contracts. |

The continuous control benchmark does not force NEO's goal-conditioned latent
programmer or image/grid data contracts onto a sequential `(s, a, s')` problem.
They remain candidates for Stage 2, after adding action conditioning and
learning mechanisms from observed transitions.

## Stage 1 additions

| Role | Source | Contract |
|---|---|---|
| Physics and oracle hypotheses | `src/rwt/dynamics.py` | `DynamicsParameters` implements friction, gain, action rotation and external force; `CandidateLibrary` enumerates their combinations, including C. C is absent from the training schedule but available as a composed oracle hypothesis in this diagnostic baseline. |
| Hidden schedule | `src/rwt/schedule.py` | Seeded irregular dwell schedule; training omits C, evaluation uses A → B → A → D → C → B. |
| Gymnasium environment | `src/rwt/environment.py` | Observation is `[x, y, vx, vy]`; shared 2D action limits, finite-state termination and unchanged reward; `info` contains no regime data. |
| Inference and policies | `src/rwt/agents.py` | Bayesian transition-prior update after each observed transition; passive, random, pure-IG and reward-plus-IG action selection under one probe budget. |
| KL diagnostics | `src/rwt/diagnostics.py` | Closed-form pairwise Gaussian transition KL, state-action rankings and realized-action separation. |
| Evaluation-only scoring | `src/rwt/evaluation.py` | Reads schedule metadata separately to score per-regime identification, entropy and change delay. It is never passed to a policy. |
| Multi-seed experiment | `experiments/rwt/run_stage1.py` | Runs 20+ seeds, beta sweeps, per-step traces, KL tables and timeline/trade-off plots. |
| Stage 1.5 protocol | `docs/rwt_stage1_5.md` | Defines filter timing, assumptions, action constraints, artifacts and limitations. |
| New tests | `tests/test_rwt_stage1.py` | Checks transition prior/update, information gain, KL identifiability, regime timing, budget, action limits, stability and leakage. |

## Stage 2A additions

| Role | Source | Contract |
|---|---|---|
| Continuous executor adapter and learned theories | `src/rwt/stage2a.py` | Reuses `models.program_execution.ProgramExecutor` unchanged behind a one-token continuous-vector adapter. Fits diagonal-Gaussian predictors from blinded `(state, action, next_state)` arrays; the main composition model uses a learned base plus two residual primitives and marginalized latent bit-codes. |
| CPU experiment | `experiments/rwt/run_stage2a.py` | Collects A/B/D train and validation transitions plus held-out evaluation schedules; trains matched-budget monolithic, latent independent-expert and compositional baselines; reports oracle reference and a separate privileged-label diagnostic. |
| Stage 2A evaluation | `docs/rwt_stage2a.md` | Records exact NEO reuse decision, likelihood, held-out-C protocol, CPU metrics and the observed latent-decomposition identifiability limit. |
| Stage 2A tests | `tests/test_rwt_stage2a.py` | Tests bit composition, gradient flow, C train/validation exclusion, checkpoint round-trip, output shapes and transition-only data contract. |

## Stage 2A.1 additions

| Role | Source | Contract |
|---|---|---|
| Structural diagnostics | `src/rwt/identifiability.py` | Computes ground-truth additive closure on matched state/action probes, constructs a three-code assignment ambiguity witness, and implements exact HMM forward evidence/filter steps. |
| Temporal and structured learners | `src/rwt/stage2a.py` | Adds sequence-marginal HMM training, train-only input normalization and a state/velocity/action structured linear baseline; weak calibration is a separately named ablation. |
| Fixed-seed ablations | `experiments/rwt/run_stage2a1.py` | Uses A/B/D-only train/validation, fixed A/B/C/D evaluation streams, data budgets, Stage 2A checkpoint comparison, hierarchical paired bootstrap summaries, and composition/occupancy diagnostics. |
| Stage 2A.1 report | `docs/rwt_stage2a1.md` | Documents closure, algebraic non-identifiability, temporal/structural ablations, final CPU metrics and limitations. |
| Stage 2A.1 tests | `tests/test_rwt_stage2a1.py` | Covers ground-truth closure, assignment ambiguity, HMM forward equivalence/gradients, and structured primitive composition. |

## Stage 2B additions

| Role | Source | Contract |
|---|---|---|
| Online hypothesis bank and inference | `src/rwt/stage2b.py` | Enumerates all three additive completions from three seen dynamics predictors, scores predictive velocity distributions, applies post-observation Bayesian updates with recurrence memory, and selects actions by expected information gain. Oracle and learned prediction-set factories are separate. |
| Paired online evaluation | `experiments/rwt/run_stage2b.py` | Trains unlabeled A/B/D-only independent-expert and monolithic models, evaluates A→B→D→C→A→C across 20 fixed seeds, and records online prediction, identification, return, probe cost, recurrence, and adaptation checkpoints. |
| Stage 2B report | `docs/rwt_stage2b.md` | Defines candidate construction, posterior timing, evaluator-only alignment, splits, online protocol and results. |
| Stage 2B tests | `tests/test_rwt_stage2b.py` | Tests completion enumeration, post-transition updates, memory prior, predictive-distribution EIG, candidate likelihoods, schedule sequence and no-label agent interface. |

## Stage 2C additions

| Role | Source | Contract |
|---|---|---|
| Online residual adapters and theory memory | `src/rwt/stage2c.py` | Freezes learned Stage 2B predictors; adds a 10-weight RLS residual, predictive adapter uncertainty, delayed mismatch-triggered revision and per-candidate streaming adapter memory. Public update methods accept observed transitions only. |
| Audit and online evaluation | `experiments/rwt/run_stage2c.py` | Replays Stage 2B for label/behavior and cache audits; calibrates mismatch cutoff on A/B/D validation; compares frozen/revised/memorized composition, expert memory, sliding-window correction and full monolithic finetuning on 20 common streams; includes a separate close-Gaussian identification control. |
| Stage 2C report | `docs/rwt_stage2c.md` | Reports Stage 2B audit, protocol differences, pre-update per-regime performance, paired results, recurrence reuse semantics and remaining structural-identification limits. |
| Stage 2C tests | `tests/test_rwt_stage2c.py` | Tests RLS adaptation/uncertainty, persistence threshold, recurrence retrieval, online-only interfaces and no-memory reset behavior. |

Run the 20-seed Stage 1.5 sweep from the repository root:

```bash
python experiments/rwt/run_stage1.py --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
```

Install the optional benchmark dependency with `pip install -e ".[rwt,test]"`.

## Stage 3A CPU additions

| Role | Source | Contract |
|---|---|---|
| Recurrent MuJoCo environment, 17D model, generic RLS, mini MPC-CEM | `src/rwt/stage3a.py` | Perturbs actual HalfCheetah-v5 friction and motor gain, restores physics on reset/close, keeps labels evaluator-only, and plans through learned state/reward predictions. |
| Dataset, training and online smoke runner | `experiments/rwt/run_stage3a.py` | Collects A/B/D-only NPZ train/validation, selects checkpoint without C, evaluates a short A→B→D→C→A→C schedule and stores all CPU metrics. |
| CPU/GPU run specifications | `experiments/rwt/configs/stage3a/cpu_smoke.yaml`, `experiments/rwt/configs/stage3a/gpu_full.yaml` | Separate small CPU and prepared larger CUDA configurations. |
| Report and tests | `docs/rwt_stage3a_cpu.md`, `tests/test_rwt_stage3a.py` | Records exact CPU evidence, performance limits, environment/replay/checkpoint/no-leakage tests and unrun GPU command. |

## Stage 3A.1 CPU preflight additions

| Role | Source | Contract |
|---|---|---|
| Train-only linear ridge baseline and generic adapter cache | `src/rwt/stage3a1.py` | Fits normalized delta dynamics on A/B/D, stores 17D RLS revisions and retrieves them from already observed transitions without regime labels. |
| Final CPU audit | `experiments/rwt/run_stage3a1.py` | Audits physics/data leakage, runs three-seed MLP training, persistence/ridge baselines, fixed-state rollouts, reward/MPC checks and recurrence probes. |
| GPU readiness | `experiments/rwt/run_stage3a.py` and `experiments/rwt/configs/stage3a/gpu_pilot.yaml` | Adds environment-only CUDA check, exact-data checkpoint resume, independent pilot config and serial/batched CPU/GPU runner paths. |
| Report and tests | `docs/rwt_stage3a1.md`, `tests/test_rwt_stage3a1.py` | Records negative results, per-regime preflight metrics, memory limits and resume/numerical contracts. |
