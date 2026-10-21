# CLSM implementation audit — 21 October 2026

## 1. Source, environment, and implementation map

The new work is isolated on branch `duc`, based on source commit `87b865b2a620ff473dfc246f73666f8ff310e8cc` from `codex/scope-law-architectures`. The implementation now has a complete executable path for its approximate generalized-EM architecture, clean-source checkpoint replay, and C1/C2/C3 diagnostic evaluators. This remains distinct from a completed scientific acceptance run: the hard gates below are still open. The original `main` checkout is Git-clean; its former local work remains in two named recoverable stashes. No earlier Stage 1–3, TIF, NEO or results source was modified. CLSM code is under `src/rwt/clsm/`, evaluator and CLI code under `experiments/rwt/clsm/`, tests under `tests/clsm/`, and this documentation under `docs/clsm_*`; `pyproject.toml` adds only optional extras.

The existing `stage3a.py`, `stage3b.py`, `tif/scope_law.py` and identifiability routines were inspected for timing, separate dynamics/reward treatment and prior failure modes; no old functions were mutated. Python 3.11.9, PyTorch 2.13.0 CPU, Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6, SciPy 1.17.1, PyYAML 6.0.3, pytest 9.1.1, Stable-Baselines3 2.9.0 and Windows are installed. CUDA is unavailable to PyTorch; ManiSkill is not installed. Current run artifacts are local and ignored by Git. Each checkpoint records its source commit, source fingerprint, dirty status, resolved config, public-data hashes, versions and RNG/optimizer state. Only clean-source checkpoint replay is accepted by evaluators.

## 2. Executable mathematical components

| Specification | Executable symbols | Current verification |
|---|---|---|
| (3)–(7) force pathway | `MechanismForce.forward`, learned nonnegative normalized `W`, positive damping shape | Nominal anchor, inactive-coordinate gradient, dissipation and environment shape tests |
| (8)–(9) Gaussian emission | `ExecutableLaw.forward`, `.log_prob`; variance bounds and reference calibration | Reference initialization and emission gradient test |
| (10)–(13) law evolution | `LawEvolution.hazard`, `.structural_log_transition`, `.predict_branches` | Weibull, sparse kernel, same-code reset age and exact finite-state timing tests |
| (14)–(15) hierarchical memory | `TheoryMemory`, `HierarchyParameters`, scratch/reuse evidence | Expected completed-visit mass, scratch-only and one-update tests |
| (16)–(18) current-law filter | `LawFilter.observe`, `.evolve`, `ekf_correct`, `compress_components` | Float64 Cholesky/Joseph, ancestry, support and zero-mass tests |
| (19) approximate GEM | `approximate_estep`, `backward_*`, `expected_*_nll`, `generalized_em_mstep` | Toy gradients through dynamics, `W`, evolution, hierarchy; HalfCheetah CPU smoke |
| (20)–(22) future laws | `LawForecaster.sample_laws`, `.sample_signatures`, `energy_score` | All horizons, frozen same-current-belief comparator, fixed independent probe bank and evaluator-only oracle snapshots |

The M-step uses six-point latent Gaussian cubature for the expected emission loss, smoothed event probabilities for Weibull/sparse-edit/reset loss, approximate Gaussian drift second moments, and mean-field hierarchy loss. Optimization is bounded, clipped and selected by validation prequential NLL. This is *approximate* generalized EM, not exact EM or a guarantee of monotonic likelihood improvement. The soft online prototype updates remain a mean-field approximation that needs serial-dependence calibration.

## 3. Deviations and hard gates

1. **Long-world inference fails the registered mass gate.** With the specified 128 components, the final-source 64-budget C3 replay reached 4.854% maximum discarded mass for reuse and 4.203% for scratch, against the 0.1% maximum. No different-age or different-code mixtures were merged to hide that loss. The CPU training smoke uses eight-step worlds only to exercise the training path; it is not an accuracy fix. The 500-step full world configuration must not be promoted until this is resolved by a named accuracy/computation ablation or a justified model revision.
2. **Swimmer-v5 is excluded before test scores.** The installed nominal controlled DOFs `[3,4]` have zero damping; the required multiplicative damping mechanism cannot act. `benchmark_exclusions.json` preserves the exact finding. The four-environment panel cannot be claimed under this installed model.
3. **ManiSkill PushCube-v1 remains blocked.** Official controller documentation defines PD target-position semantics, not direct torque actions. The optional package and controller/physics audit are unavailable locally; `full_maniskill.yaml` exists but runners error explicitly rather than silently using a MuJoCo torque adapter. The native PD interface requires a separate validated implementation.
4. **Baseline panel is incomplete.** `pooled_mlp` has an independent validation checkpoint; scratch-only and frozen-current are executable matched-data comparators. Causal history MLP, noncompositional switching experts, geometric-duration, factorization ablation, matched-compute scratch and privileged oracle upper bound are not implemented. No native published RL baseline or downstream control result is claimed.
5. **Confirmatory data and uncertainty are absent.** Only HalfCheetah short smoke datasets and one-seed diagnostics have run. Full 150k/30k/30k data per track and environment, five training seeds, twenty evaluation worlds, long-horizon age-support validation and paired confidence intervals have not run. The aggregate script returns a diagnostic summary and leaves confirmatory intervals null.
6. **Remaining numerical and coverage checks.** No sigma-point likelihood cross-check, measured merge approximation bound, model uncertainty calibration, joint/action occupancy sufficiency proof, state-dependent ManiSkill adapter, or long-world resume repeat is available. No model was selected on held-out composition, oracle probes, or downstream return.

## 4. Learner/evaluator information boundary

The learner accepts only public `obs`, `action`, `next_obs` arrays and static nominal `EnvSchema`. Its likelihood does not use reward; public reward is the stationary function `-0.001||o||²-0.01||a||²`. The strict loader permits exactly `obs, action, reward, next_obs, terminated, truncated, world_id, world_step`, checks hashes and world ordering, and rejects an injected `code` column even when its file hash matches the manifest. No module in `src/rwt/clsm/` imports the evaluator physics wrapper or world schedules.

Evaluator-only code owns modified physics arrays, code/parameter/reset schedules and full MuJoCo snapshots. Oracle signatures and hidden schedule files are under `results/rwt/clsm_evaluator/`; learner checkpoints contain no such fields. The fixed 64-probe calibration bank is shared by methods and evaluation worlds, and its full-state snapshots stay evaluator-only. This directory separation is not an operating-system ACL guarantee.

## 5. Executed commands and reproducibility

Commands were run in the `duc` worktree on Windows PowerShell after editable installation with `python -m pip install -e '.[rwt-clsm,test]' --no-deps`. Local smoke artifacts generated while source edits were in progress are diagnostic; clean-source replay and provenance checking are required before a result is treated as reproducible.

| Command or check | Exit | Observation |
|---|---:|---|
| `python -m pytest tests/clsm -q` | 0 | 50 tests passed, including exact discrete enumeration, continuous backward conditional, zero-mass support, hidden-column rejection, probe restoration, checkpoint next-step replay, scratch/reuse conditional parity and synthetic Weibull recovery. |
| `audit_physics.py` on HalfCheetah, Walker2d, Reacher (`--probes 8`) | 0 each | Repeated-query error 0, physics restored and declared reward invariant. |
| `build_datasets.py` on HalfCheetah COMP/EVOL/MEM with their CPU smoke configs | 0 each | Separate 32/16/16 public transitions and hashed manifests. Eight-step validation/test worlds happened to contain only code 000. |
| `run_train.py` on HalfCheetah COMP/EVOL/MEM CPU smoke, default full-model path | 0 each | Three nominal epochs, one GEM outer iteration, five M-steps, validation-selected best/latest checkpoints. Validation prequential NLL: COMP 9.4412; EVOL/MEM 9.4460. These are smoke diagnostics. |
| `run_train.py --evaluate-only` and `--resume` on COMP smoke | 0 each | Loaded checkpoint and repeated validation NLL 9.4412; resume found no remaining outer iteration. This is not a multi-step resumed-training equivalence test. |
| `run_baselines.py --method pooled_mlp` on COMP smoke | 0 | Same public/reference hashes; pooled validation NLL 9.1432, better than CLSM's 9.4412. |
| `run_comp_eval.py` with 64 fixed probes, four held-out codes | 0 | One-seed diagnostic; pooled beats CLSM on all four codes. |
| Final-source `run_train.py` on HalfCheetah COMP/EVOL/MEM smoke | 0 each | COMP/EVOL/MEM validation prequential NLL 9.44068/9.44546/9.44546; eight-step training worlds have only numerical-roundoff dropped mass. |
| Final-source `run_train.py --evaluate-only` on COMP | 0 | Replayed validation NLL 9.44068 and dropped mass `3.33e-16` from the saved clean-source checkpoint. |
| Final-source `run_baselines.py --method pooled_mlp` | 0 | Pooled validation NLL 9.14319, better than CLSM's 9.44068 on this smoke split. |
| Final-source `run_comp_eval.py` with 64 probes, four held-out codes | 0 | `ES_pooled - ES_CLSM` = `-0.36746, -0.46912, -0.32320, -0.28461` for codes `3,5,6,7`; all four favor pooled. One-seed diagnostic only. |
| Final-source `run_forecast_eval.py` with horizons `0,1,8,24,64` | 0 | EVOL skill `0, -0.00415, +0.04824, -0.00079, -0.01935`; only horizon 8 is positive. |
| Final-source `run_memory_eval.py` on MEM with budgets `4,8,16,32,64` | 0 | One-world same-data scratch/reuse diagnostic; scratch-minus-reuse gains approximately `0, 0, -0.02245, +0.02797, +0.01469`. Max dropped mass 4.854% reuse / 4.203% scratch. `inference_degraded=true`; methods are not compute matched. |
| `python -m pytest tests/clsm -q` after implementation changes | 0 | 50 passed in 16.44 seconds. |
| `fit_nominal_controller.py --env Reacher-v5 --steps 20`, then isolated full-collector interface test of 8/4/4 transitions | 0 | Controller/collector wiring only. The actual full config requires at least 100,000 nominal fit steps. |
| `summarize.py --results-dir results/rwt/clsm` | 0 | Includes negative effects; no confidence interval or confirmatory label from one seed. |
| Final branch recheck: `audit_physics.py` on HalfCheetah, Walker2d and Reacher (`--probes 8`) | 0 each | Repeated-query max error `0.0`; physics arrays restored on all three valid environments. Swimmer remains excluded by its zero nominal controlled damping. |
| Final branch recheck: `run_train.py --check-data` on COMP/EVOL/MEM CPU smoke configs | 0 each | Train and validation splits load for all three tracks; manifests retain separate train hashes and shared smoke validation hash. |
| Final branch recheck: `run_train.py --evaluate-only` on EVOL and MEM | 0 each | Both clean-source checkpoints replay at validation prequential NLL `9.4454564488`, maximum dropped mass `4.44e-16`. |
| Added spec-gate regressions for terminal timing and stationary censored dwell fit | 0 | Terminal transition is retained under the old law before advancing; a no-event stationary sequence fits a low age-zero switch hazard. |
| Final branch recheck: `python -m pytest tests/clsm -q` | 0 | 52 passed in 10.52 seconds; `compileall` and `git diff --check` also pass. |

## 6. Physical audit and coverage

Eight matched MuJoCo probes per valid environment gave strictly positive nominal/intervention separations in at least the sampled perturbations. Across five variants the ranges of raw next-observation norm separation were HalfCheetah 0.511–3.345, Walker2d 0.020–2.044 and Reacher 0.010–0.626. These small audits establish intervention execution, restoration and deterministic repeatability, not sufficient policy occupancy or structural identifiability. New dataset manifests report action-coordinate, joint-velocity and observation-norm quantiles; contact strata and oracle pairwise separation *under the full collector distribution* remain unmeasured.

## 7. Results by claim (diagnostic only)

- **Current law / C1:** COMP pooled validation NLL 9.14319 versus CLSM 9.44068 on the same short validation data. On four held-out masks with four public adaptation transitions and one fixed 64-probe bank, `ES_pooled - ES_CLSM` was `-0.36746, -0.46912, -0.32320, -0.28461` for codes `3,5,6,7`. All favor pooled. One seed and short worlds cannot establish general superiority or failure.
- **C2:** EVOL single-world `ES_frozen - ES_CLSM` at `h=0,1,8,24,64` was `0, -0.00415, +0.04824, -0.00079, -0.01935`. Improvement at only one horizon is not evolution success; this diagnostic does not support C2.
- **C3:** Final-source MEM one-world `ES_scratch - ES_reuse` at budgets `4,8,16,32,64` was approximately `0, 0, -0.02245, +0.02797, +0.01469`. Maximum dropped mass reached 4.854% for reuse and 4.203% for scratch, so the evaluator marks inference degraded. Reuse and scratch used 64/32, 128/64, then 128/128 components at the first three reported budgets; measured inference time was 154.53 versus 143.57 seconds. The matched-compute baseline is still missing, and one world is not confirmatory C3 evidence.
- **Downstream:** no policy, action-value, ranking or return result.

COMP, EVOL and MEM checkpoints and all C1/C2/C3 evaluator results were generated from the final implementation source. Raw model predictions, oracle signatures, fixed probe hash, checkpoint hashes and all rows are retained in ignored local result directories. They are smoke diagnostics, not paper-facing estimates.

## 8. Computation and diagnostics

The latest 52-test suite passed in 10.52 seconds; each HalfCheetah GEM CPU smoke took roughly two minutes. The five-budget MEM evaluation took approximately 298 seconds of measured inference time across reuse and scratch. No GPU was available, and peak CPU memory was not measured. The eight-step training smoke had discarded mass at numerical-roundoff scale; 16–64-step diagnostics breached the gate substantially. The best COMP validation NLL was worse than pooled. Posterior collapse, actuator support identification, covariance calibration, event/hazard learning and memory negative transfer need larger paired studies. The score files keep negative rows and do not select a best horizon or seed.

## 9. Open scientific questions versus implementation defects

The finite probe projection can equate different physical laws; the locked weakening/increased-damping family may not produce sufficiently separated effects under every policy. A good next-observation fit cannot identify future-law evolution. The 128-component long-world mass loss, missing baseline panel, missing ManiSkill PD adapter and missing full run are implementation/acceptance failures, not negative scientific conclusions. Five train seeds and independent world schedules are needed before testing C1–C3.
