# CLSM-R2 evidence and results

**Status:** versioned R2 architecture and an informative CPU identification pilot are implemented. The pilot does not complete generalized EM or establish C1, C2, or C3. The exact remaining gates are listed below.

## Source-derived evidence

- Starting point: branch `codex/clsm-r2`, based on `duc` commit `36a20c217127e06a6f81e2211a6c71d50c7cc547`.
- Tested architecture source fingerprint: `66120427728b8c7b9514678c7c8da6b6976c18ee204be65d17b3c52aca212dd8`.
- R2 requires `architecture_revision: 2`; old or unversioned configs/checkpoints are rejected. Exact run-length filtering uses a float64 `[8,501]` age table and no per-age pruning.
- The prior audit's 128-component emission-free 500-step failure is the documented motivation for replacing the old filter. That historical error is not reported as a posterior-error estimate.
- `tests/clsm` covers arbitrary code emissions at 500 steps against an independent probability recursion, exhaustive short categorical paths, event-conditioned reset/switch paths, categorical normalization, partition and strength invariants, memory conjugacy/visit accounting, and strict public-data loading.

## New experiments

### Gate 0 CPU pilot: HalfCheetah-v5

The pilot config is [clsm_r2_cpu_pilot.yaml](clsm_r2_cpu_pilot_resolved_config.yaml). Its resolved data manifest is [clsm_r2_halfcheetah_pilot_manifest.json](clsm_r2_halfcheetah_pilot_manifest.json). Pilot data and checkpoints are local under `results/rwt/clsm_r2_pilot/`; the report includes their hashes so ignored experiment outputs need not be mistaken for committed source.

| Split | Transitions | Independent worlds | Public SHA-256 |
|---|---:|---:|---|
| Train | 4,000 | 40 | `3a893eb49fb5cef0bf36582855ec0287966c46e3ab4f449dca251de4ab5fa38b` |
| Validation | 1,000 | 10 | `d8a1560446674c792ee464ca29adeac0ab8e598c71d6325d04e9071adc95c970` |
| Test | 1,000 | 10 | `8708d2dc7cb14cea7143d713f10c6c4df624755101a857374b034afa5725a72c` |

The 4,000 training transitions have 1,000 each from nominal and the three singleton modes, with ten independently initialized worlds per mode and independently drawn physical strengths per world. The collector uses independent multisine frequencies/phases and bounded noise. The public loader sees only public observations, actions, reward, endings, and world boundaries. Evaluator labels are in a separate file; they were read only by the post-freeze pilot audit.

The separate nominal anchor used 5,000 transitions. Normalized training MSE fell from `0.6990` to `0.0970` over 20 epochs (reference data SHA-256 `64883477d07cf637d8af852963e567d9d22aaf6af999f35fc45f3f1f99dd33ed`). This measures nominal replay fit only.

For the selected 6-actuator schema, the train-only profile search evaluated all 31 canonical disjoint partitions with 12-node Gauss-Hermite integration. Across initialization seeds `4100`, `4101`, and `4102`, it selected the same partition, matching the evaluator partition up to label permutation after freeze. Best-to-second train log-evidence gaps were `1112.16`, `1271.99`, and `1244.14`. Each run's selected partition won `1000/1000` train-world bootstrap replicates. The initializer seeds were predeclared in the pilot config; the matching support is evidence for this controlled generator pilot, not a general support-recovery guarantee.

Train-occupancy diagnostics, computed after support freeze, found 1,000 transitions per mode. Active physical-strength standard deviations were `0.0818` (motor 1), `0.0929` (motor 2), and `0.1962` (damping). The empirical action covariance eigenvalues ranged from `0.1915` to `0.2385` (condition number `1.2455`). A finite-difference, variance-normalized local sensitivity Fisher diagnostic had diagonal `[1646.41, 819.74, 1109.83]`, rank `3`, minimum eigenvalue `819.74`, and condition number `2.008`. These quantities use the fitted nominal decoder and evaluator values only in the separate audit; they do not measure posterior parameter recovery.

With the train-selected support frozen, held-out public profile NLL on 1,000 test transitions was `-2.0871` (test SHA above). This is a diagnostic profile likelihood, not a C1 score or a comparison to a scientific baseline.

### Reproduction commands

Run from the repository root with `PYTHONPATH=src;.` in PowerShell:

```powershell
python -m pytest tests/clsm -q
python experiments/rwt/clsm/build_datasets.py --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot --dry-run
python experiments/rwt/clsm/build_datasets.py --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot
python -m experiments.rwt.clsm.run_train --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot --check-data
python -m experiments.rwt.clsm.run_train --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot --reference-only
python -m experiments.rwt.clsm.run_train --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot --train
python -m experiments.rwt.clsm.run_train --env HalfCheetah-v5 --track comp --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot --evaluate
python experiments/rwt/clsm/audit_r2_pilot.py --env HalfCheetah-v5 --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot
```

The runner's `--train` stage currently performs nominal-anchor loading and the train-only finite partition profile M-step. It does **not** perform the full alternating generalized-EM updates of `F_phi`, evolution, and population hierarchy; the generated report says so. `--evaluate` therefore reports only frozen-support public profile likelihood.

## Gate checklist

Test source fingerprint for this run: `66120427728b8c7b9514678c7c8da6b6976c18ee204be65d17b3c52aca212dd8`; base commit: `36a20c217127e06a6f81e2211a6c71d50c7cc547`.

| Gate | Result | Evidence / remaining work |
|---|---|---|
| 0 — informative data | **PASS for this CPU pilot** | 6,000 balanced stationary intervention transitions, three predeclared seeds, full-rank local sensitivity, repeated independent strengths, support bootstrap, held-out public split. No claim of sufficient sample size beyond this generator/environment. |
| A — oracle decoder | **NOT RUN** | Need isolated oracle-G/true-z/true-strength decoder and the two required ablations on held-out physical-state probes. |
| B — known-code strength identification | **NOT RUN** | Fisher rank was measured, but no known-z/unknown-xi posterior calibration or multiplier recovery was run. |
| C — exact age and hybrid filter | **PARTIAL** | Exact discrete 500-step arbitrary-emission and short exhaustive event tests pass. The small continuous quadrature-grid comparison, cubature error curve, predictive calibration, and batched/scalar timing remain unrun. |
| D — evolution | **NOT RUN** | Need switching-rich worlds, duration parameter recovery, event-stratified forecasts, oracle-current/learned-K and frozen-persistence comparisons. |
| E — memory | **NOT RUN** | Need paired `mem_correlated` and `mem_independent` runs with fresh test prototypes, budgets `{4,8,16,32,64}`, confidence intervals, and full hierarchy training. |
| F — broader environments | **NOT RUN** | Reacher pilot was retained locally but has only two actuators and cannot assess global partition recovery. HalfCheetah is the one six-actuator pilot; no multi-environment experiment or baseline suite was run. ManiSkill remains outside the validated control interface. |

## Interpretation and decision

The exact categorical age recursion and support grammar passed the implemented software checks. The controlled HalfCheetah pilot recovered the known disjoint partition consistently across three network initializations and resampled worlds. The nominal reference model fit the separate nominal data, and public held-out profile likelihood is available for review.

This is not a completed empirical CLSM-R2 study. Full generalized EM, oracle decoder adequacy, posterior strength calibration, hybrid approximation error, evolution forecasting, and memory positive/negative controls remain open. C1/C2/C3 effectiveness is therefore **not established**, and the pilot should guide design changes rather than be cited as scientific confirmation.
