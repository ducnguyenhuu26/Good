# Q1 executable dynamics theory learning

This package implements the stage-aware Q1 pipeline in a custom 2D continuous dynamics benchmark. It covers simulator and oracle checks (M1), a shared context encoder/executor (M2), weakly supervised learned primitives (M3), composition and exhaustive program inference (M4). It does not include a policy learner or control-return claim.

## Install and run

Python 3.11 and CPU PyTorch are supported. From the repository root:

```bash
uv sync --extra q1 --extra test
uv run --extra q1 --extra test pytest -q tests/test_q1_simulator.py tests/test_q1_splits.py tests/test_q1_model_contracts.py
uv run --extra q1 python scripts/q1_m1_check.py --config configs/q1_smoke.yaml
uv run --extra q1 python scripts/q1_pipeline.py --config configs/q1_smoke.yaml --through m2
uv run --extra q1 python scripts/q1_pipeline.py --config configs/q1_cpu.yaml --through m4 --run-dir results/q1/cpu_seed20261023
uv run --extra q1 python scripts/q1_report.py --run-dir results/q1/cpu_seed20261023
```

The smoke config is marked `SMOKE`; it checks execution and is never scientific evidence. The CPU config is one adjustable development seed and evaluates four held-out backgrounds with ten direct-context optimization steps to keep the CPU pilot bounded. Increase `oracle.test_envs` and `training.direct_context_steps` for a larger local run. A failed gate writes its diagnostics and stops later stages. The pipeline refuses to overwrite an existing run directory; use `--resume` to reuse completed stage artifacts. To reuse an M2 checkpoint from another run, pass its `context_model.pt` path with `--reuse-m2-checkpoint`; the adjacent M2 gate report must also be present and pass.

## Q1 checkpoint repair diagnostics

The P0/P1 repair commands below perform **no training** and use validation environments only. They require the local M2, M3-before-composition, and M4-after-composition checkpoints; checkpoints are ignored by Git and are not downloaded automatically. A missing checkpoint produces a clear nonzero exit. Each run directory is write-once and stores checkpoint hashes, resolved config, metrics, and reports.

```bash
uv run --extra q1 python scripts/q1_repair_audit.py --config configs/q1_repair_diagnostic.yaml --run-dir results/q1/repair_v1_audit_full --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt --m4-checkpoint results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt --device cpu
uv run --extra q1 python scripts/q1_diagnose_feasibility.py --config configs/q1_repair_diagnostic.yaml --run-dir results/q1/repair_v1_diagnosis --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt --m4-checkpoint results/q1/cpu_confirmatory_impl/m4/primitives_m4.pt --device cpu
```

`q1_repair_audit.py` checks units, opaque code-map consistency, and physical-oracle identity versus functional-class scoring on the same uninterrupted old-to-new adaptation trajectories. `q1_diagnose_feasibility.py` compares encoder-only fresh contexts, direct context fitting with 10/40/200 optimizer steps, M3 and M4 true-code predictions, simulator-only paired effects, and all-candidate signed ranking margins. The previously inspected ordered-pair identities remain development data; only a separately frozen fresh-seed replication may be called confirmatory. These scripts never start the M2/M3/M4 training commands.

The current diagnosis selected Branch O for a targeted M3 after+effect ablation. This one-seed development command compares the after-only control with the preregistered beta candidates on the training split and selects beta on validation only:

```bash
uv run --extra q1 python scripts/q1_train_effect_ablation.py --config configs/q1_repair_m3.yaml --run-dir results/q1/repair_v1_m3_effect_ablation --m2-checkpoint results/q1/cpu_confirmatory_impl/m2/context_model.pt --m3-checkpoint results/q1/cpu_confirmatory_impl/m3/primitives_m3.pt --device cpu --threads 1
```

All three tested variants failed the per-code M3 validation gate because U4's small effect remained inaccurate. Stop before M4 or confirmatory training. See [diagnosis](docs/q1_duc1_diagnosis.md) and [repair results](docs/q1_duc1_repair_results.md) for measured evidence and runtime.

After a successful development pilot, run the separately declared five-seed protocol with:

```bash
uv run --extra q1 python scripts/q1_confirmatory.py --config configs/q1_confirmatory.yaml --through m4
```

It records every seed's exit code and gate status, including failed or incomplete seeds. The confirmatory run is intentionally a separate, explicit command because it is substantially more expensive than the pilot.

## Protocol

The dynamics parameters, parameter regions, train/validation/test environment counts, and selected/withheld ordered pairs are in `configs/q1_splits.yaml`. Train, validation, ID-test, and background-OOD environments use disjoint parameter bins. Each environment, state initialization, action sequence, and noise sequence has its own reproducible seed. Public neural inputs contain only `(s_t, a_t, delta_v_t)`; simulator parameters and program/group metadata stay in sampler-side structures.

The fixed exploratory policy collects one uninterrupted adaptation trajectory at each `N ∈ {5,10,20,50,100}`. Query trajectories use independent initial conditions and action streams. The oracle's functional equivalence classes use fixed probes and a declared tolerance; when two programs induce the same response, reports give both raw program identity and functional-class accuracy.

M3 uses training-only opaque same-intervention groups with a saved random code bijection. M4 composes the learned codes as `T_j(T_i(c))`; it ranks all 21 candidate programs using the old context and adaptation transitions. Its length prior is selected on validation backgrounds and permitted validation compositions. Composition OOD, background OOD, and joint OOD are reported separately. A single development seed is not confirmatory evidence; consult the stage `gate_report.md` and per-run `metrics.json` before making research claims.

## Artifacts

Each run stores resolved config, split manifest, model checkpoints, machine-readable metrics, training curves, per-seed CSV summaries, gate reports, and generated plots under `results/q1/<run_id>/`. Checkpoints are ignored except for the small saved stage diagnostics. `docs/q1_m1_report.md` records the M1 development gate and its limits; `docs/q1_development_pilot.md` summarizes the current one-seed M2–M4 gates, including the M4 failure.

## U4 root-cause diagnosis

The U4-specific physical-identifiability and frozen-M2 latent-oracle investigation is documented in [`docs/q1_u4_root_cause_results.md`](docs/q1_u4_root_cause_results.md), with machine-readable evidence in `results/q1/u4_root_cause_v1/`. The physical oracle reaches 0.922 functional-class accuracy at N=100, but the free-latent U4 fit does not generalize to disjoint query probes. The diagnosis is **UNRESOLVED / MIXED**; the conditional loss-floor ablation, M4, and confirmatory training were skipped.
