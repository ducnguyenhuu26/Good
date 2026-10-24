# Q1 `duc1` repair implementation and development result

## Run protocol

The first supported repair branch was O: train M3 primitives on both absolute after-dynamics and paired functional-effect prediction. The ablation used the existing frozen M2 checkpoint, all 48 training backgrounds, 16 validation backgrounds, the existing four-code map, 100 epochs, and 1,200 optimizer updates per variant. It did not use test environments or run M4 training.

The after-only control (`beta=0`) and effect variants (`beta=0.25`, `beta=1`) shared the same initial primitive weights, code map, train examples, optimizer, minibatch RNG stream, and update count. Beta candidates and per-code gate thresholds were fixed in `configs/q1_repair_m3.yaml` before the run. Validation selected `beta=0.25` by the declared max-min per-code effect-skill criterion, with after-MSE gain as tie-breaker.

Training uses simulator-only paired conditional means on identical training state-action probes. For each intervention, the simulator produces `y_after(u)` and `y_before(u)`; the learner receives only the public `(state, action, delta-v)` query during inference and the opaque training code. This adds a stronger train-time weak-supervision channel than opaque group membership, and is disclosed here. The three-variant run used 18,432 unique train state-action pairs and 958,464 simulator response evaluations including scale estimation and repeated minibatches. Measured CPU runtime was 81.22 seconds on this host.

Exact reproduction command:

```bash
uv run --extra q1 python scripts/q1_train_effect_ablation.py --config configs/q1_repair_m3.yaml --run-dir results/q1/repair_v1_m3_effect_ablation --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt --device cpu --threads 1
```

Checkpoint files are local and Git-ignored. The command fails clearly if either checkpoint is absent; no automatic download or random-weight fallback is used. It refuses to overwrite a nonempty run directory.

## Validation result

Effect skill is `1 - effect-MSE / zero-effect-MSE`. After gain is `(frozen-M2 persistence after-MSE - learned after-MSE) / frozen-M2 persistence after-MSE`. Each reported interval is a normal-approximation 95% CI across the 16 validation backgrounds. Thresholds require both effect skill `>=0.10` and after-MSE gain `>=0.05` for every code.

| Variant | U1 skill / after gain | U2 skill / after gain | U3 skill / after gain | U4 skill / after gain | Gate |
|---|---:|---:|---:|---:|---|
| After-only `beta=0` | `0.877 / 0.882` | `0.678 / 0.454` | `-2.720 / 0.038` | `-300.739 / -0.013` | **FAIL** |
| Effect `beta=0.25` | `0.910 / 0.899` | `0.882 / 0.522` | `0.550 / 0.135` | `-1.326 / 0.028` | **FAIL** |
| Effect `beta=1.0` | `0.916 / 0.898` | `0.884 / 0.518` | `0.578 / 0.134` | `-1.368 / 0.017` | **FAIL** |

The paired-effect objective improves U3: effect skill moves from `-2.720` under after-only to `+0.550` at beta 0.25, with `13.5%` after-MSE gain. It does not satisfy U4. U4's validation effect energy is only `4.45e-8`; beta 0.25 achieves `-1.326` effect skill and `2.8%` after gain. Increasing beta to 1.0 does not fix it. All variants therefore fail the predeclared per-code gate; no M4 fine-tuning or test evaluation follows from this run.

This is a negative one-seed development result, not evidence that paired effects are ineffective generally. It shows the tested loss/floor/beta range repaired U3 but did not represent U4's very small effect. Do not tune further on the same validation set and call it confirmation. Any next beta/floor hypothesis should be versioned and frozen before a fresh-seed replication, with the U4 effect scale and after-vs-persistence behavior diagnosed first.

## Files and checks

- P0 audit: `results/q1/repair_v1_audit_full/` (32.69 seconds; no training).
- P1 diagnosis: `results/q1/repair_v1_diagnosis/` (433.47 seconds for full ranking/direct-fit audit; no training).
- Branch O ablation: `results/q1/repair_v1_m3_effect_ablation/` (81.22 seconds; three M3 development variants).
- Focused Q1 verification after implementation: `21 passed` across simulator, split, model-contract, no-leakage, P0 metrics, and paired-effect tests.

The existing `cpu_confirmatory_impl` M1–M4 artifacts and the original M4 gate result remain unchanged. No five-seed protocol, Q2, policy, or control experiment was run.
