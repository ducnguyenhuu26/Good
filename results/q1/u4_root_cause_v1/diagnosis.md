# U4 root-cause diagnosis

The intervention implementation and measuring boundary passed Phase 0. Exact physical ranking on the original 16 validation backgrounds distinguishes U4 at the largest budget: N=100 functional-class accuracy 0.922 (cluster CI [0.883952724773778, 0.959797275226222]); with matched zero noise it is 1.000. Thus the current exploration has low information at small N but does not impose a hard U4 identifiability ceiling at N=100. Natural U4 effect energy, per-background noise ratios, velocity feature conditioning, and candidate KL values are recorded in the oracle JSON.

The frozen-M2 latent fit is mixed. For U4, the beta=0.25 objective improves selected-fit effect skill to 0.680, but disjoint-query effect skill is -0.758 and query after-MSE gain is 0.812%. The after-only fit has a large fit/query effect gap. The local Jacobian projection skill is 0.797 on disjoint query probes, while `E(D_after_long)` has query after-MSE 1.65423e-05 versus old-context persistence 1.69653e-05. These measurements do not establish frozen-M2 impossibility: they point to latent optimization/objective generalization ambiguity, and the local tangent has U4 signal. The known-reachable mock control passes; the unreachable mock is reported as inconclusive.

Phase 3 was skipped because the free-latent query oracle did not establish U4 representability with reliable query generalization. The current scale floor remains a hypothesis, not a measured cause; no further floor or beta search is justified by these results. Learned U4 remains below the unchanged M3 development gate. M4 and confirmatory training were not run.

Reproduction commands are in `docs/q1_u4_root_cause_results.md`; per-background evidence is in `latent_oracle_per_background.csv`.

**UNRESOLVED / MIXED** — next decisive test: improve the frozen-latent oracle's internal fit/selection protocol and repeat the free-latent query evaluation with a predeclared longer-budget or multi-start optimizer, keeping the original simulator and split fixed.
