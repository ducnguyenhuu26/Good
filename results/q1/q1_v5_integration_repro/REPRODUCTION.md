# Q1 V5 integration reproduction

## Scope and verdict

The source files were copied from the supplied `Q1_V5_SOURCE.zip` into the checkout at base commit `eff3d5c460f273917f132b37adf2d503ff558280`, on branch `codex/q1-v5-identifiability-active`. The ZIP SHA-256, environment, seeds, code inputs, and exact train/development seed formula are recorded in the adjacent JSON files. The original protected test split was not accessed.

Both exploratory pilot seeds ran end to end from this checkout. Functional-class accuracy at N=100 reproduced to the displayed precision: 0.8482 / 0.8571 ID-like and 0.9435 / 0.9405 OOD. U4 effect summaries did not fully reproduce. In this checkout, old+EB ID-like skill was 0.9810 / 0.9759 against population skill 0.9773 / 0.9748; the ZIP pilot reports 0.9756 / 0.9758 against 0.9703 / 0.9746. Seed 2 is close; seed 1 differs more. The attached pilot was run against locally reconstructed modules, and exact dependency versions or data hashes were not supplied, so the cause cannot be isolated. This mismatch is retained as an open reproducibility discrepancy.

These runs are reproduction diagnostics, not the fresh frozen Phase B benchmark. They do not contain matched V2/V3/M4/control rows, ordered-pair metric tables, or per-background absolute effect MSE/energy. No deployable rule was selected using train/inner validation. Therefore the fixed-policy result is **INCONCLUSIVE**, not a pass or a fresh-development fail.

## Commands and tests

```powershell
python -m pytest -q tests/test_q1_v5_model.py tests/test_stage_contrast.py tests/test_q1_v5_identifiability.py
python -m pytest -q tests
python -m compileall -q q1_v5 q1_v4 scripts/q1_v5_experiment.py scripts/q1_v5_identifiability.py
python scripts/q1_v5_experiment.py --seed 20261024444 --train-backgrounds 40 --dev-backgrounds 16 --output results/q1/q1_v5_integration_repro/seed_20261024444
python scripts/q1_v5_experiment.py --seed 20261024618 --train-backgrounds 40 --dev-backgrounds 16 --output results/q1/q1_v5_integration_repro/seed_20261024618
```

Focused V5/stage tests: **15 passed**. Full repository suite: **315 passed, 3 failed**. The three failures are pre-existing Windows path expectation mismatches in `tests/test_reproduction_commands.py` (`/output` and `/observation` expected, while the implementation returns `D:\\output` and `D:\\observation`); none is in Q1 or in the V5 changes. `compileall` passed.

## Protocol audit and limits

- The model and refit modules do not import the simulator. Training data are generated with opaque codes and the shared response/operator fit consumes observed transitions. Simulator truth is used for generation and for explicitly labeled single-effect diagnostics / after-the-fact scoring.
- Candidate ranking scores all 21 ordered programs. The learned action operator composition uses the row-major Kronecker Jacobian. The ranking likelihood uses a Woodbury/Laplace approximation and adaptation prefixes from one continuous stage trajectory; independent queries are generated separately.
- The input runner itself does not write split manifests or provenance, and chooses no train/inner-validated single estimator. The manifests in this folder were reconstructed from its deterministic seed formulas after execution. They establish seed disjointness for these pilot runs, but they are not an independently preregistered Phase B manifest.
- Mechanism accuracy uses training-only before/after trajectories. True program identities enter only the evaluation loop/decoding and the oracle-effect diagnostic; they are not passed to candidate likelihood scoring. No protected split was opened.
- Source docstring terminology was corrected from polynomial to positive saturating mixture. The source ZIP's original runnable implementation remains otherwise unmodified.

See `split_manifest.json`, `config.json`, `provenance.json`, and `tests.json`. Each seed folder contains the full per-program and effect rows plus the runner's metrics JSON.
