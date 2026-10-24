# U4 root-cause gate report

## Phase status

| Phase | Status | Simulator/query accounting | Optimizer steps | CPU wall time |
|---|---|---:|---:|---:|
| 0: equation and measurement verification | PASS | 2,052 float64 transition evaluations plus normalization checks | 0 | 0.21 s |
| 1: physical oracle | COMPLETE | 2,983,680 candidate transition evaluations per noise condition (5,967,360 including matched sigma-zero); plus class-probe and stage-pair trajectories | 0 | 20.80 s |
| 2: frozen-M2 latent oracle | COMPLETE / INCONCLUSIVE | 230,528 conditional-mean/old-support/reachable-context evaluations plus 18,432 sampled fit/query trajectory transitions (estimated) | 115,200 vectorized Adam steps; four independent latent candidates per step | 167.95 s |
| 3: M3 loss-scale ablation | SKIPPED_WITH_REASON | 0 | 0 | 0 s |

Phase 0 checked that U3 changes only `d <- 1.4*d`, U4 changes only `b <- 1.5*b`, and the paired targets are exactly `-0.4*dt*d*v` and `-0.5*dt*b*||v||*v`. Conditional targets contain no independent transition noise; the noisy paired sample stores the observed difference separately. Normalizer round-trip error was 2.38e-07 in float32.

## Measured decision evidence

- Physical oracle U4 functional-class accuracy at N=100: **0.922**; 95% background-cluster CI **[0.883952724773778, 0.959797275226222]**. The matched sigma-zero control reached **1.000**. At N=100 the oracle ranked U4 `118/128`, identity `7/128`, and `(3,3)` `3/128`; closest wrong class was identity in `91/128` and `(3,3)` in `37/128.
- Natural U4 effect energy was 1.14e-08 squared delta-v, RMS 0.000101 delta-v, with effect/noise RMS ratio 0.201. Mean per-step KL versus identity/U3/`(3,3)` was 0.0455/4.44/28.4 nats. U1/U3 oracle controls are in the same JSON.
- Frozen M2 U4 free-latent beta=0.25 on disjoint query probes: effect MSE `2.29e-08` / effect energy `1.59e-08` squared delta-v, effect skill **-0.758**, after-MSE gain **0.812%**. Selected fit-probe effect skill was 0.680; its query gap is retained per background. After-only beta=0 selected fit/query effect skills were -635.011/-923.604.
- Local Jacobian U4 projection skill averaged 0.776 on fit and 0.797 on query probes. This is evidence of a local direction, not a global representability result. U1/U3 beta=0.25 controls reached query effect skills 0.975/0.641; finite-gradient failures: 0.
- The untouched learned beta=0.25 U4 operator had effect skill -3.195 and after-MSE gain 1.627%. **M3 U4 DEVELOPMENT GATE: FAIL** under unchanged per-code thresholds (effect skill >= 0.10 and after-MSE gain >= 0.05). The new floor ablation is **NOT RUN**.

Phase 3 is **SKIPPED_WITH_REASON**: Phase 2 disjoint-query U4 latent oracle does not pass the effect/after-fit criteria; selected fit improves do not generalize, while local Jacobian projection remains strong. This leaves optimization ambiguity and does not justify a floor ablation. The 1e-2 floor was not further tuned on these same validation backgrounds. No M4 training, test/OOD scoring, or five-seed confirmatory protocol was run.

## Verification

`tests/test_q1_u4_diagnostics.py`: 13 passed. Existing focused Q1 suite: 21 passed. `compileall`: exit code 0. Exact commands and exit codes are in `verification.json`. No test split was evaluated.
