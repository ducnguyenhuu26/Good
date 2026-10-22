# CLSM-R2 sequential research gate results

**Run date:** 2026-10-22 (local CPU)

**Source:** `codex/clsm-r2`, pinned source commit `75e3b70ac006ef434c8963e16c117a0d1b2a3f16`

**Architecture fingerprint:** `66120427728b8c7b9514678c7c8da6b6976c18ee204be65d17b3c52aca212dd8`

**Scope:** sequential work order in `CODEX_CLSM_R2_SEQUENTIAL_RESEARCH_GATES.md`. The attached document supplies requested gates and criteria; this file records local evidence and does not treat those criteria as findings.

## Gate summary

| Gate | Status | Evidence / reason | Next action |
|---|---|---|---|
| Preflight | **PASS** | `python -m pytest tests/clsm -q`: 70 passed in 50.00 s. Public train/validation/test hashes match the pilot manifest; evaluator label hashes match the separate evaluator manifest. The held-out 16- and 32-step replay completed from the existing checkpoint, with checkpoint/source fingerprints matching and no refit. | Gate A only. |
| A: force-compositional decoder | **FAIL** | Updated by the independent Gate A experiment below: the mixed-seen structured control fails against the oracle-conditioned MLP on code `101` matched probes despite 20 mixed-training worlds, substantial force-space support, and full conditional Fisher rank. | Review force-response representation before posterior/generalized-EM gates. |
| B: continuous posterior reference | **NOT RUN** | Sequential policy stopped at A. | Wait for Gate A PASS. |
| C: evolution forecasting | **NOT RUN** | Sequential policy stopped at A. | Wait for Gate B PASS. |
| D: generalized EM and law learning | **NOT RUN** | Sequential policy stopped at A. | Wait for Gate C PASS. |
| E: recurrence/memory | **NOT RUN** | Sequential policy stopped at A. | Wait for Gate D PASS. |

The gate runner is [run_clsm_r2_research_gates.py](/C:/Users/CNC-NVCUONG-PGD/.codex/worktrees/clsm-implementation/learning-to-theorize/scripts/run_clsm_r2_research_gates.py). Its CLI and dry-run were checked. It records missing scientific prerequisites as `INCONCLUSIVE` and stops `--through E` at the first unresolved gate. Machine-readable evidence is written under `results/rwt/clsm_r2_gates/<run_id>/`; these experiment artifacts are ignored by Git.

## Preflight details

- Branch and source lineage: `codex/clsm-r2` at the pinned commit above. The R2 test checkpoint exists at `results/rwt/clsm_r2_pilot/comp/HalfCheetah-v5/r2/4100/checkpoint_r2.pt` (SHA-256 `6135a2103926fae0357bf56bbde1f0463083d5b00f96c73f80df459c5ecf5861`). It reports the same source fingerprint as the evaluator code and `checkpoint_training_dirty=false`.
- Public split SHA-256: train `3a893eb49fb5cef0bf36582855ec0287966c46e3ab4f449dca251de4ab5fa38b`; validation `d8a1560446674c792ee464ca29adeac0ab8e598c71d6325d04e9071adc95c970`; test `8708d2dc7cb14cea7143d713f10c6c4df624755101a857374b034afa5725a72c`.
- Evaluator-only split SHA-256: train `4543a6bdc43a1be82bc5e74a8b900123048147b79c75f7e0bb5bfc74241fbfe0`; validation `fd189fbd32e5b734165c29fa6a8d7cc9bbef2d2a56076d7f51b09d94899e0608`; test `7c658d51d52a783cd5492ef46609aa1d61ef1d10098ddc52a6821f946c3f31da`. Public and evaluator arrays are in separate roots; `load_public_split` verifies the manifest hash and exact public key set. Test `test_training_loader_rejects_oracle_column_even_with_valid_hash` passed.
- Reproduction command: `python scripts/evaluate_clsm_r2_effectiveness.py --env HalfCheetah-v5 --config experiments/rwt/clsm/configs/clsm_r2_cpu_pilot.yaml --output-dir results/rwt/clsm_r2_pilot --seed 4100 --max-steps 32`.
- 16-step prefix: 10 worlds, 160 transitions; R2 filter NLL/dimension `-0.2109655`, nominal-only `-0.0266268`, paired difference `-0.1843387`, paired-world bootstrap 95% CI `[-0.3269521,-0.0620197]`; R2 better on 7/10 worlds; code accuracy `71.875%`.
- 32-step prefix: 10 worlds, 320 transitions; R2 filter NLL/dimension `-0.2224548`, nominal-only `-0.0319791`, paired difference `-0.1904757`, paired-world bootstrap 95% CI `[-0.3394671,-0.0684255]`; R2 better on 8/10 worlds; code accuracy `74.6875%`. Plug-in oracle-code/strength NLL was `-0.2019515`; it is evaluator-only and is not a strict Bayes-risk upper bound.
- This is a favorable **short-prefix stationary one-step filtering diagnostic** versus nominal-only on these ten worlds. It does not establish composition generalization or overall architecture effectiveness. The pilot has ten test worlds, so the world-level uncertainty remains material.
- `run_train.py --train` is `partition_profile_only`; it fits a nominal anchor and train-only finite partition profile, not the decoder/evolution/population hierarchy with full generalized EM. Config values Weibull scale 20 and shape 1.4 remain initialization values; this replay is not evidence that evolution parameters were learned.
- The pilot design and learner/evaluator boundary are auditable, but its held-out test split is singleton-only. It cannot satisfy Gate A's mixed-composition claim.

## Gate A decision

Gate A is **INCONCLUSIVE**, not PASS or FAIL: the relevant held-out composition experiment and matched model controls have not been run. The existing stationary replay is retained as useful evidence for online short-prefix inference under the pilot conditions. It does not answer whether the force-conditioned decoder learned transferable interactions.

Required before attempting Gate B:

1. Freeze separate singleton train/validation and composition diagnostic splits with fresh worlds and strength draws; reserve confirmatory worlds.
2. Train the prescribed oracle-conditioned structured decoders and unstructured/pooled MLP controls genuinely, with validation-only model selection and distinct `oracle_diagnostic` checkpoint tags.
3. Evaluate zero-shot held-out codes on both trajectory occupancy and a fixed evaluator-only matched-state/action probe bank; report per-code NLL, normalized mean error, vector-signature error, world bootstrap intervals, and severe per-code failures.
4. Calculate the full active-coordinate 3×3 Fisher/Jacobian diagnostic on mixed compositions, including eigenvalues, rank, and condition numbers.
5. Decide Gate A from the registered criteria without tuning against confirmatory test worlds.

No Gate B–E experiment was run. No result in this report supports C1, C2, C3, or full-method acceptance.


## Gate A updated result (2026-10-22)

The prior INCONCLUSIVE status above described the state before the composition experiment. The completed, preregistered experiment supersedes it. See [clsm_r2_gate_a_results.md](clsm_r2_gate_a_results.md) for the full evidence. Gate A is **FAIL for the tested representation on the registered probe domain**: code `101` mixed-seen structured minus oracle-conditioned MLP had matched-probe normalized MSE difference `+0.0221` (95% world-bootstrap CI `[+0.0132,+0.0307]`) and signature difference `+0.0312` (`[+0.0186,+0.0431]`). The trajectory-occupancy results favor structured models, so the failure is domain-specific and does not imply all mixed-composition prediction failed.

The runner wrote the final decision from existing fitted checkpoints and scores; no confirmatory worlds were scored. Gate B–E remain unrun because Gate A did not pass.
