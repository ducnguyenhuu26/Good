# Q1 V5 final decision record

## Decision

**V5_FIXED_INCONCLUSIVE.** Integration and the two supplied pilot seeds were reproduced end to end, but the specified fresh, frozen Phase B benchmark with matched V2/V3/M4 and controls was not run. The supplied runner has no predeclared inner-validation estimator selection or baseline-matched evaluation interface. No single deployable inference rule is selected, and no V5 pass is claimed.

**Q1_ACTIVE_EXCITATION: NOT RUN.** The local Phase C audit supports practical weak excitation and nuisance collinearity, which is a reason to study active excitation as a separate protocol. No active policy/control experiment was implemented in this run, so there is no active verdict or evidence of benefit.

## Provenance and protocol

- Repository remote: `btl` -> `https://github.com/ducnguyenhuu26/Good.git`.
- Starting/base source commit: `eff3d5c460f273917f132b37adf2d503ff558280`.
- Work branch: `codex/q1-v5-identifiability-active`.
- Supplied source archive SHA-256: `f3f5124d651ee8ea644a0a9446197e60d07214424cf13bebbf63af0dee85a163`.
- Environment: Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1, PyTorch 2.13.0+cpu, Windows.
- Fixed protocol inputs retained for reproduction: `dt=0.05`, `sigma=0.0005`, old support 96, adaptation prefixes `{5,10,20,50,100}`, 40 training backgrounds and 16 development backgrounds per regime per seed, all 21 ordered programs. Original protected test split was not accessed.
- Reproduction seeds: `20261024444`, `20261024618`. The output manifests reconstruct the runner's deterministic train/development seed formulas. They are exploratory manifests recorded after the source runner executed, not a preregistered fresh Phase B split.
- Runner commands and per-seed rows are in `results/q1/q1_v5_integration_repro/`; the exact configuration and environment/source hashes are in that directory's JSON files.

## Reproduction results

Functional-class accuracy at N=100 matched the ZIP pilot: ID-like `0.8482 / 0.8571`, background-OOD `0.9435 / 0.9405` across the two seeds. Current-checkout old+EB U4 ID-like skill was `0.9810 / 0.9759` versus population skill `0.9773 / 0.9748`. Paired background-bootstrap 95% intervals for old+EB minus population were `[-0.0026, 0.0093]` and `[-0.0044, 0.0069]`; both include zero. Background-OOD paired intervals were `[0.0716, 0.1888]` and `[0.0719, 0.1865]`.

The ZIP pilot reports slightly different U4 results, especially for seed 1. This is an unresolved reproduction mismatch; its source was reported as locally reconstructed modules, without environment/data hashes. These exploratory runs do not supply matched-baseline comparisons, ordered-pair summaries, or per-background absolute effect MSE and true-effect energy.

## Identifiability result

The Phase C audit used the same two seed values, 16 ID-like and 16 OOD backgrounds each, and 5/10/20/50/96 prefixes. It includes six task-parameter nuisance columns and five shared-response nuisance coordinates. At N=96, median efficient information for `log(b)` was `3.69` ID-like and `8.23` OOD; corresponding local standard-error bounds were `0.52` and `0.35`. Median cosine between linear- and quadratic-drag velocity features was about `0.96`. Information was zero at N=5 because the short design could be spanned by nuisance columns. This indicates practical weak excitation/collinearity at small budgets, not structural nonidentifiability. Full per-background diagnostics are in `results/q1/q1_v5_identifiability/`.

## Tests and code status

- Focused V5, stage-contrast, and identifiability tests: **15 passed**.
- Full repository suite after integration: **315 passed, 3 failed**. All failures are unrelated Windows path expectation mismatches in `tests/test_reproduction_commands.py`; those tests expect POSIX `/output` and `/observation` paths.
- `compileall` passed for the integrated modules and scripts.
- Existing runner/model/stage-contrast sources came from the supplied ZIP. The model docstring was corrected to describe the positive saturating mixture accurately. Added an SVD-based efficient-information diagnostic, focused tests, manifests, and mathematical claim-boundary documentation.

## Remaining work required for scientific completion

Phase B must first freeze one regime-blind rule using training/inner-validation only, then evaluate it on fresh disjoint backgrounds against matched V2/V3/M4/population/identity/direct-context baselines and report all specified effect and ordered-pair metrics. Phase D requires a separately named active protocol with matched controls and new backgrounds. These steps remain open; no source in this branch or report should be interpreted as satisfying them. The complete theorem matrix, proof sketches, and open claims are in `docs/q1_v5_identifiability_and_recovery.md`.
