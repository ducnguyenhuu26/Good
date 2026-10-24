# Q1 U4 root-cause investigation

## Scope and conclusion

This validation-only development diagnosis follows the original split manifest and keeps U4 strength, simulator, exploration, noise, stage protocol, and acceptance thresholds fixed. It ran P0 measurement verification, P1 physical identifiability, and P2 frozen-M2 latent fitting. It did not run M4, the new M3 floor ablation, test/OOD scoring, or five-seed confirmatory training.

The evidence does not isolate a single root cause. The exact physical oracle can distinguish U4 at N=100, so the benchmark is not physically unidentifiable at that budget. The short-budget oracle is much weaker. A free latent fit improves U4 effect skill on its selected fit probes, but the effect does not generalize to disjoint query probes; after-MSE gain is also below the fixed 5% gate. At the same time, a local Jacobian projection captures much of the U4 effect on both fit and query probes, and U1/U3 controls work. The 300-step multi-start optimizer therefore does not establish M2 impossibility or demonstrate that the learned operator is only blocked by the scale floor.

**UNRESOLVED / MIXED.** The next decisive test is a predeclared longer-budget or improved fit-selection protocol for the frozen-latent oracle, retaining disjoint query probes and the original split. Do not retune the floor or beta against this validation set.

## Main measurements

- Phase 0 passed: U3 changes only `d <- 1.4*d`; U4 changes only `b <- 1.5*b`; float64 paired effect equations matched within `4e-16`; normalizer float32 round-trip error was `2.38e-7`. The conditional-mean target excludes independent simulator noise.
- On the 16 original validation backgrounds, U4 functional-class accuracy was `0.344, 0.688, 0.789, 0.820, 0.922` for `N={5,10,20,50,100}`. At `N=100`, the background-cluster 95% CI was `[0.884, 0.960]`. With matched `sigma=0`, accuracy was `1.0` for every budget. U1 and U3 controls are included in the JSON.
- At `N=100`, 118/128 U4 cases ranked the exact `(3,)` program first. Of the 10 remaining cases, 7 ranked identity and 3 ranked `(3,3)`. The closest wrong functional class was identity in 91/128 cases and `(3,3)` in 37/128. Per-case candidate errors are saved.
- Natural U4 effect energy was `1.14e-8` squared delta-v units, RMS `1.01e-4`; the per-component noise RMS is `5e-4`, giving an effect/noise RMS ratio of `0.20`. Mean per-step Gaussian KL was `0.0455` nats versus identity, `4.44` versus U3, `28.42` versus `(3,3)`, and `0.102` versus repeated U4. These are conditional one-step separations on observed states, not closed-loop success bounds. Velocity feature Gram condition number averaged `334` for U4 and median speed was `0.132`.
- On disjoint U4 query probes, the beta `0.25` free-latent fit had effect MSE `2.29e-8` / true effect energy `1.59e-8` squared delta-v, giving skill `-0.758`; after-MSE gain was `0.81%`. Selected-fit effect skill was `0.680`. After-only fitting selected on fit probes had skill `-635` on fit and `-924` on query, showing extreme denominator sensitivity and no useful effect transfer. Query local-Jacobian projection skill averaged `0.797`; the fit value was `0.776`. This local linear result is not a global representability proof.
- U1 and U3 beta `0.25` free-latent controls had query effect skills `0.975` and `0.641`, with zero finite-gradient failures. The frozen M2 `E(D_after_long)` U4 reference only improved after MSE by about `2.5%` over persistence. The untouched learned beta `0.25` U4 operator had effect skill `-3.195` and after-MSE gain `1.63%`, below both unchanged development thresholds.

Because the free-latent query oracle did not demonstrate reliable U4 effect generalization, Phase 3 was skipped as required. This leaves the suspected effect-scale floor untested as a cause. The M3 U4 development gate remains **FAIL**; this is not a claim that Q1 is solved or that the frozen executor cannot express U4.

## Artifacts

- Main results: `results/q1/u4_root_cause_v1/`
- Oracle confusion, budgets, and background-cluster intervals: `oracle_identifiability.json`, `oracle_identifiability_per_case.csv`
- Matched noise-free diagnostic: `oracle_identifiability_sigma0.json`, `oracle_identifiability_sigma0_per_case.csv`
- Frozen-M2 latent fit and per-background rows: `latent_oracle.json`, `latent_oracle_per_background.csv`
- Exact checkpoint/split hashes and software versions: `provenance.json`
- Test commands and exit codes: `verification.json`
- Gate and final action: `gate_report.md`, `diagnosis.md`

## Reproduction

The local beta checkpoint is `results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt`; M2 is `results/q1/cpu_confirmatory_impl/m2/context_model.pt`. The commands below show the phases as executed. Use a new empty output directory for a fresh reproduction; the script refuses to overwrite nonempty result directories.

```powershell
python scripts/q1_u4_root_cause.py --config configs/q1_u4_root_cause.yaml --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt --out results/q1/u4_root_cause_repro --phase oracle --device cpu --threads 1
python scripts/q1_u4_root_cause.py --config configs/q1_u4_root_cause.yaml --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/repair_v1_m3_effect_ablation/beta_0.25/primitives.pt --out results/q1/u4_root_cause_repro --phase latent --device cpu --threads 1
```

The first command reads the original split manifest and produces Phase 0 and Phase 1 results. The second uses the same manifest and adds Phase 2. Both are CPU-only; Phase 2 uses four independent latent starts, beta `{0, 0.25}`, learning rates `{0.003, 0.01}`, regularization `{0, 1e-4}`, and snapshots at 100 and 300 steps.
