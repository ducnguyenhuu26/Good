# Q1 M1: simulator and oracle gate

## Scope and protocol

M1 implements the isolated 2D transition law in `q1/simulator.py`, the four physical interventions in the simulator-only module, fixed exploration, continuous stage adaptation, and oracle enumeration of all 21 programs. State is `(q_x,q_y,v_x,v_y)`, actions are bounded by `[-1,1]`, `dt=0.05`, and episodes are at most 240 transitions in the CPU config. Initial position and velocity coordinates are independently sampled from zero-mean Gaussians with standard deviation 0.25. State is never clipped; a norm above 100 raises a numerical-instability error.

Parameter ranges and split counts are in `configs/q1_splits.yaml`. Train, validation, ID-test, and background-OOD bins are disjoint in all of `B`'s diagonal entries and the positive scalars `d`, `b`, and `kappa`. Each environment has a saved ID and parameter seed. The benchmark's deterministic setting uses `sigma=0`; its initial noisy development setting uses `sigma=0.0005` velocity noise. Noise affects `v_next`, and `q_next` is computed deterministically from that new velocity. Initial-state, action, and dynamics-noise random streams use separate seed offsets.

The adaptation segment starts exactly at the final old-stage state and continues without reset. Query trajectories are generated from independent initial states and action streams. All methods will share the same fixed sinusoidal-plus-bounded-jitter exploration rule.

## Oracle results

The oracle knows the pre-change physical parameters and scores each candidate against the exact same observed adaptation trajectory. It predicts the deterministic velocity increment and ranks candidates by MSE. Functional equivalence is measured on 256 fixed state-action probes for each of 16 validation backgrounds, with maximum absolute response tolerance `1e-7`. The resulting 21 program IDs form 16 response classes because several intervention pairs commute. Therefore raw identity accuracy is shown for audit, while the gate uses class accuracy; it does not claim recovery of physically indistinguishable program IDs.

| Setting | N | Identity accuracy | Functional class accuracy | U1→U2 class accuracy | U2→U1 class accuracy |
|---|---:|---:|---:|---:|---:|
| Deterministic, smoke | 5 | 0.762 | 1.000 | 1.000 | 1.000 |
| Deterministic, smoke | 10 | 0.762 | 1.000 | 1.000 | 1.000 |
| Deterministic, smoke | 20 | 0.762 | 1.000 | 1.000 | 1.000 |
| Deterministic, smoke | 50 | 0.762 | 1.000 | 1.000 | 1.000 |
| Deterministic, smoke | 100 | 0.762 | 1.000 | 1.000 | 1.000 |
| sigma=0.0005, development | 5 | 0.667 | 0.874 | 1.000 | 1.000 |
| sigma=0.0005, development | 10 | 0.695 | 0.912 | 1.000 | 1.000 |
| sigma=0.0005, development | 20 | 0.732 | 0.961 | 1.000 | 1.000 |
| sigma=0.0005, development | 50 | 0.735 | 0.961 | 1.000 | 1.000 |
| sigma=0.0005, development | 100 | 0.760 | 0.997 | 1.000 | 1.000 |

The noncommuting pair's fixed-probe response RMSE is `0.0053441`. Per-budget confusion matrices, best-versus-second-best MSE margins, equivalence classes, split manifests, resolved configs, and provenance are in `results/q1/m1_deterministic_validation/` and `results/q1/m1_validation_low_noise/`.

An initial validation-only diagnostic at `sigma=0.01` reached only 0.604 functional-class accuracy at `N=100`. Following the prescribed development-only adjustment, initial position/velocity spread was raised from `(0.15, 0.08)` to `(0.25, 0.25)`, action excitation was increased from a 0.65 signal plus 0.35 jitter to a 0.85 signal plus 0.15 jitter, and noise was reduced to `sigma=0.0005`. The report preserves the before/after diagnostics under `results/q1/m1_validation/` and `results/q1/m1_validation_tuned/`; those intermediate folders are intentionally not part of the committed final M1 artifact. No final-test background or withheld composition was used for tuning.

## Gate decision

**M1 passes for development to proceed to M2.** The deterministic transition, stochastic reproducibility, shapes, continuous no-reset adaptation, composition-order, and split-disjointness tests pass. The noncommuting orders are distinct on fixed probes and recovered perfectly in the noisy development oracle. Under the selected low-noise development setting, functional-class accuracy reaches 0.961 by `N=20` and 0.997 by `N=100` over 16 validation backgrounds. The `N=5` result is lower and is retained as such; the data do not support claiming that every program is reliably identifiable from five transitions.

These are simulator and oracle diagnostics, not learned-method results or scientific confirmation. The deterministic table is explicitly marked smoke. M2 must demonstrate context use and improved held-out prediction before M3 can run; M3 and M4 remain gated by their own reports.

## Reproduction

```bash
python -m pytest -q tests/test_q1_simulator.py tests/test_q1_splits.py
python scripts/q1_m1_check.py --config configs/q1_smoke.yaml --output results/q1/m1_deterministic_validation
python scripts/q1_m1_check.py --config configs/q1_cpu.yaml --output results/q1/m1_validation_low_noise
```
