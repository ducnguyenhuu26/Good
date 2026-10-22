# CLSM-R2 research gates A–C: diagnostic results

**Execution date:** 2026-10-22
**Source branch:** `codex/clsm-gate-a`
**Base commit:** `63a170f10b8a3388fdf68c4e0c33ef030425564b`
**Scope:** Gates A–C only, per the user's instruction to stop at Gate C. Gate B and C were run as explicitly diagnostic work after Gate A failed; they do not establish a sequential A–E result.

## Decision summary

| Gate | Result | Evidence-based interpretation |
|---|---|---|
| A — force-compositional representation | **FAIL** | On registered matched probes for code `101`, both singleton-trained and mixed-seen structured decoders were worse than the oracle-conditioned unstructured MLP on normalized MSE/signature error. Trajectory-occupancy results were favorable, but do not cancel the matched-probe failure. |
| B — known-code strength posterior | **INCONCLUSIVE** | The analytic linear-Gaussian fixture matched to `1.1e-16`, but the numerical posterior reference did not converge: the maximum coarse/fine xi-mean discrepancy was `0.79346`, versus the registered `0.01` tolerance. Do not interpret cubature-versus-reference metrics as posterior accuracy. |
| C — joint semi-Markov/strength filter | **INCONCLUSIVE** overall; categorical subgate **PASS** | Exact categorical regressions passed. An independent 96-history synthetic study found material code-age and age-conditioned strength discrepancies, lower approximate prequential score, and undercoverage. Required production CPU profile and scalar/batch parity were not run, so Gate C cannot pass. |
| D–E | **NOT RUN** | Stopped at Gate C as requested. Their prerequisites A–C are not satisfied. |

## Gate A

The frozen result and full per-code tables are in [the Gate A report](clsm_r2_gate_a_results.md), with machine-readable scores and confidence intervals in [the Gate A metrics](clsm_r2_gate_a_metrics.json). Its registered failure is specific to the matched-probe force-response representation on code `101`; the report preserves the favorable trajectory-occupancy results and the state-distribution reversal.

## Gate B

Run command:

```powershell
python scripts/run_clsm_r2_gate_b_diagnostic.py
```

The diagnostic used 12 independent worlds for each of seven singleton/mixed codes, 5 adaptation budgets (`4, 8, 16, 32, 64`), and a common 64-probe held-out bank. The analytic fixture passed to floating-point precision. However, the quadrature convergence audit reached a maximum xi posterior-mean difference of `0.79346` (tolerance `0.01`), so the reference is not accurate enough to decide whether R2's cubature posterior is faithful. Mean cubature/reference xi errors were largest for code `111` (`0.188`) and code `011` (`0.185`), followed by code `101` (`0.118`); all per-world and budget rows are retained in the CSV. Physical-multiplier and predictive scores are also retained, but remain descriptive under the unconverged reference. Action/velocity summaries are recorded; a separate Fisher-information calculation was not implemented, which is another missing Gate B deliverable.

Artifacts: [posterior and quadrature rows](../results/rwt/clsm_r2_gates/gate_b_diagnostic/posterior_vs_quadrature.csv), [physical calibration](../results/rwt/clsm_r2_gates/gate_b_diagnostic/physical_multiplier_calibration.csv), [reference/convergence record](../results/rwt/clsm_r2_gates/gate_b_diagnostic/strength_reference.json), [split manifest](../results/rwt/clsm_r2_gates/gate_b_diagnostic/dataset_manifest.json), and [coverage/Fisher notes](../results/rwt/clsm_r2_gates/gate_b_diagnostic/coverage_and_fisher.md).

## Gate C

Run command:

```powershell
python scripts/run_clsm_r2_gate_c_joint_filter_diagnostic.py
python -m pytest tests/clsm -q
```

The synthetic diagnostic fixed a two-code, age-dependent hazard system and crossed duration scale (`6`, `20`), emission separation (weak/strong), drift variance (`0.005`, `0.1`), and reset probability (`0`, `0.15`). It ran 6 independent histories per cell, 40 observations per history (96 total histories). The finite reference used a 201-point xi grid on `[-4, 4]`; xi was represented jointly with code and age in the reference recursion. The R2-like comparison retained exact code-age mass but used one Gaussian strength posterior per code.

| World-level metric | Mean | 95% bootstrap CI |
|---|---:|---:|
| Joint code-age total variation | 0.298 | [0.278, 0.320] |
| Next-switch probability absolute error | 0.0182 | [0.0153, 0.0211] |
| Age-conditional xi mean absolute error | 0.154 | [0.143, 0.166] |
| Approximate minus grid prequential log score, per step | -0.0536 | [-0.0756, -0.0349] |
| Exact-grid 90% xi interval coverage | 90.6% | [89.1%, 91.9%] |
| Shared-per-code Gaussian 90% xi interval coverage | 86.7% | [84.6%, 88.6%] |

A negative log-score difference means the factorized approximation assigned lower predictive log score than the grid reference. All 96 histories showed at least two same-code age-conditioned posterior means above the diagnostic mass threshold. This supports the targeted concern that one shared Gaussian per code discards strength-by-age dependence. It does **not** by itself justify implementing an age-conditioned Gaussian in production: the experiment is synthetic, uses a finite grid with boundary handling, and has only six independent histories per factorial cell.

The categorical regression subset passed `6/6`; the full CLSM suite passed `71/71` (`19.52 s`). These tests establish only the software invariants they exercise. Gate C remains inconclusive because the production `AgeLawFilter` profile over lengths `16/32/64/100` and fixed-RNG scalar/batched parity required by the work order were not completed.

Artifacts: [joint-grid summary and decision](../results/rwt/clsm_r2_gates/gate_c_diagnostic/decision.json), [categorical reference result](../results/rwt/clsm_r2_gates/gate_c_diagnostic/categorical_reference.json), [production-profile status](../results/rwt/clsm_r2_gates/gate_c_diagnostic/CPU_profile.json), [per-observation event/state metrics](../results/rwt/clsm_r2_gates/gate_c_diagnostic/event_marginals.csv), [per-history summaries](../results/rwt/clsm_r2_gates/gate_c_diagnostic/world_level_metrics.csv), and [factorial-cell curves](../results/rwt/clsm_r2_gates/gate_c_diagnostic/approximation_error_curves.csv).

## What the results support

1. Preserve Gate A's negative matched-probe finding; do not claim zero-shot mixed-composition success.
2. Improve or validate the low-dimensional Gate B numerical reference before interpreting posterior calibration.
3. The Gate C synthetic counterexample warrants a more controlled follow-up of age-conditioned strength dependence, plus production profiling and scalar/batch parity. Current evidence does not yet warrant changing the production representation.
4. Do not start generalized-EM evolution training or memory evaluation under this gate sequence. They were not run.

All output directories are diagnostic artifacts and do not replace the prior R2 pilot or its checkpoints.
