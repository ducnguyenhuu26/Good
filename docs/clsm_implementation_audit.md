# CLSM implementation audit

## Implementation map (21 October 2026)

- Base source commit: `87b865b2a620ff473dfc246f73666f8ff310e8cc` (`btl/codex/scope-law-architectures`). First CLSM implementation commit: `4410671` on new branch `duc`. CLSM lives only in `src/rwt/clsm/`, `experiments/rwt/clsm/`, `tests/clsm/`, and new `docs/clsm_*` files, plus the additive `pyproject.toml` extra.
- Existing source inspected: `src/rwt/stage3a.py` supplies a physics wrapper timing example; `src/rwt/stage3b.py` supplies the separate state-delta and reward-model pattern; `src/rwt/tif/scope_law.py` supplies a prior particle-filter diagnostic. These are references, not imported into the CLSM learner. `src/rwt/identifiability.py` provides an independent finite-state evidence example. No upstream functions are mutated or copied into the new package.
- Installed: Python 3.11.9, PyTorch 2.13.0, Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6, SciPy 1.17.1, PyYAML 6.0.3, pytest 9.1.1, Windows. ManiSkill has not been inspected or validated.
- Learner boundary: public observation, commanded action, stationary public reward, next observation, and done flags only. Nominal actuator topology and nominal damping may be supplied as static setup metadata. Hidden masks, intervention parameters, switch/reset times, modified simulator arrays, and oracle counterfactual snapshots belong exclusively to evaluator modules and never enter learner arrays, model checkpoints, normalization, or checkpoint selection.
- Scientific boundary: the attached specification defines a restricted Gaussian-drift/reset CLSM, whereas the formulation also permits broader Levy increments. C1, C2, and C3 require separate locked experiments. A smoke run is not a scientific confirmation.

## Implemented mathematical components and actual symbols

| Specification | Implemented symbol | Verification boundary |
|---|---|---|
| (3)–(7) executable motor/damping effort | `MechanismForce.forward`, `.W`, `.regularizer` | Nominal/mask/gradient/dissipation unit tests |
| (8)–(9) shared force-to-delta Gaussian law | `ExecutableLaw.forward`, `.log_prob` | Shape/finite tests and reference-only initialization |
| (10)–(13) Weibull hazard, sparse edits, drift/reset | `LawEvolution.hazard`, `.structural_log_transition`, `.predict_branches` | Exact algebra and age tests |
| (14)–(15) hierarchy and entry prior | `TheoryMemory.prior_for_entry`, `.update_from_completed_visit` | Method-level tests only; not integrated into the filter |
| (17)–(18) Gaussian update | `ekf_correct`, `LawFilter.observe` | Cholesky/Joseph, nominal subspace, causality tests |
| Discrete timing reference | `forward_log_evidence_discrete` | 3-state trajectory enumeration at `<1e-10` |
| (20)–(22) projected-law samples and score | `LawForecaster.sample_laws`, `.sample_signatures`, `energy_score` | `h=0`, frozen-belief and RNG-isolation tests; no trained-law evaluation |

The filter is approximate and does not yet retain proposal-corrected ancestry. The hierarchy's update method is not called by the filter. `run_train.py` explicitly refuses to call the reference-initialized backbone a complete CLSM checkpoint. No optimizer has jointly fitted `W`, `K_eta`, and prototype hyperparameters by generalized EM.

## Deviations, failures, and user-visible effect

1. **Swimmer-v5 excluded before test scores.** Installed nominal `dof_damping=[0,0,0,0,0]` and controlled DOFs `[3,4]`. Multiplicative damping cannot implement the specified third mechanism. Evidence and reason: `experiments/rwt/clsm/benchmark_exclusions.json`. The four-environment MuJoCo panel is therefore unavailable under this installed model without a separately approved benchmark redesign.
2. **ManiSkill PD adapter unavailable.** Its actual controller semantics have not been inspected, so no PushCube-v1 claim or fake torque interface is present.
3. **Smoke collection uses smooth random public actions.** The required nominal-controller mixture is not implemented. Full collection configuration errors deliberately. Smoke coverage cannot validate sustained policy occupancy or law identifiability.
4. **No integrated memory posterior or backward smoother.** `TheoryMemory` supports one update per completed visit, but the filter does not yet generate those visit summaries, update scratch/reuse evidence, or implement FFBS ancestry. This blocks full generalized EM and C3. It also prevents any claim of trained C1/C2.
5. **No full matched baselines, independent oracle signature evaluator, or hierarchical statistical report.** Existing code can sample learned-model future laws but cannot make a confirmatory C2 comparison. Evaluation scripts and full per-track result directories are not delivered as complete implementations.
6. **Forecast and filter numerical limits.** The fixed-budget filter logs discarded mass, but has no sigma-point likelihood cross-check, merge-cost approximation report, or continuous smoother. These missing acceptance tests keep the full experiment gate closed.

These are incomplete requirements, not favorable scientific results. No architecture or evaluation rule was changed to make the smoke run pass.

## Learner/evaluator boundary evidence

- `src/rwt/clsm/` imports only `EnvSchema`, which contains nominal static topology, nominal damping, bounds, time step, and explicit public-observation velocity indices. No module under this learner namespace imports `experiments/rwt/clsm/physics_wrappers.py` or `world_schedule.py`.
- The public dataset contains only `obs, action, reward, next_obs, terminated, truncated, world_id, world_step`. The public manifest contains data hashes, world IDs, and collector ID, without oracle labels or oracle paths. Oracle arrays and oracle manifest are under a separate `results/rwt/clsm_evaluator/` namespace. This is namespace separation, not an operating-system access-control guarantee.
- Reference-only checkpoint stores model state, resolved config, data hash, source commit and versions. It receives nominal reference data collected with fixed default physics; it contains no intervention labels. Current tests check public keys and code-level interface. A complete checkpoint leakage audit remains pending.

## Commands, exit status, and observed artifacts

All commands below were run in the `duc` worktree with `PYTHONPATH=src;.` on Windows PowerShell unless noted.

| Command | Exit | Observation |
|---|---:|---|
| `python -m pytest tests/clsm -q` | 0 | 27 tests passed after the signature-normalization edit, following editable installation with `python -m pip install -e '.[rwt-clsm,test]' --no-deps` |
| `python experiments/rwt/clsm/audit_physics.py --env HalfCheetah-v5 --config experiments/rwt/clsm/configs/comp.yaml --probes 8` | 0 | Repeated-query max absolute error 0; physics arrays restored; stationary reward invariant |
| Same audit for `Walker2d-v5` and `Reacher-v5` | 0 each | Repeated-query max absolute error 0; physics arrays restored; stationary reward invariant |
| `python experiments/rwt/clsm/build_datasets.py --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/cpu_smoke.yaml` | 0 | Public transition counts 32 train, 16 validation, 16 test; manifest hash reproduced byte-for-byte on rerun |
| `python experiments/rwt/clsm/run_train.py --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/cpu_smoke.yaml --device cpu --reference-only` | 0 | Three-epoch nominal reference initialization, final normalized MSE 1.0142676405356033. Rerun produced identical data hash, checkpoint hash and loss curve. |
| Same `run_train.py` without `--reference-only` | 1 | Deliberate release gate: missing smoother, integrated prototype update, generalized EM. |
| `python experiments/rwt/clsm/run_forecast_eval.py --env HalfCheetah-v5 --track evol --checkpoint results/rwt/clsm/comp/HalfCheetah-v5/reference_only/41/checkpoint_reference.pt` | 1 | Deliberate gate: reference checkpoint is not a locked joint CLSM model. |
| `python experiments/rwt/clsm/summarize.py --results-dir results/rwt/clsm` | 1 | Deliberate gate: full score and memory-adaptation files are absent. |

Local ignored smoke outputs: `results/rwt/clsm/physics_audits/*.json`, `results/rwt/clsm/comp/HalfCheetah-v5/data/`, `results/rwt/clsm/comp/HalfCheetah-v5/reference_only/41/`, and `results/rwt/clsm_evaluator/comp/HalfCheetah-v5/`. These are reproducible from the committed commands and are not full research result directories.

The original `main` checkout is Git-clean after two named, recoverable stashes: `pre-CLSM local work 2026-10-21` and `pre-CLSM formerly ignored local files 2026-10-21`. No previous tracked source or result artifact was deleted by this branch.

## Physical audit and coverage

Eight matched probes per valid environment yielded strictly positive measured nominal-vs-intervention separation in this small sample. The pooled ranges across five intervention variants were: HalfCheetah 0.511–3.345, Walker2d 0.020–2.044, Reacher 0.010–0.626 in raw next-observation Euclidean norm. These values are diagnostics on eight sampled probes, not policy-occupancy coverage or structural identification. Contact, low/high action, and low/high velocity strata have not been reported; this is another full-run gate.

## Results by scientific claim

- **Current law:** no held-out law score; only reference nominal training loss.
- **C1 composition:** no held-out pair/triple law result or matched baselines.
- **C2 evolution:** no locked multi-horizon oracle Energy Score; model sampling unit tests do not show forecast skill.
- **C3 memory:** no matched adaptation-risk result; memory is not yet integrated.
- **Downstream:** no action-value or return experiment.

## Computation and diagnostics

Observed unit suite completed in about six seconds; the reference-only CPU command completed in about six seconds on the installed environment. Peak CPU/GPU memory and per-step training/inference latency were not measured. There is no CUDA device visible to PyTorch, and `nvidia-smi` is unavailable. Posterior collapse, W-group identifiability, hazard fitting, covariance inflation, OOD probe coverage, memory negative transfer, and repeated mixture mass loss have not been measured on a trained CLSM.

## Unresolved research assumptions separate from implementation defects

The finite public probe projection may merge physically different laws. The locked actuator weakening and damping-increase family may not produce well-separated, policy-relevant effects in every environment. Five training seeds and independent world sequences are needed to test C1–C3; a good nominal fit or successful physics restoration does not imply any claim. The zero-damping Swimmer finding is an environment-schema incompatibility; the missing smoother, memory integration, baselines, and oracle evaluator are implementation work.
