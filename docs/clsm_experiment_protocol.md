# CLSM experiment protocol and current execution state

## Locked design from the supplied specification

- Four MuJoCo environments: HalfCheetah-v5, Walker2d-v5, Swimmer-v5, Reacher-v5. ManiSkill PushCube-v1 is a separate controller-semantics audit, not a torque-motor substitution.
- COMP trains and validates on structural masks `000,100,010,001`; all pair/triple masks are test-only. EVOL permits all eight masks and evaluates in-family, mildly misspecified, and stationary schedules separately. MEM measures equal-data, equal-compute adaptation to recurrence and sparse revision.
- Full intended budget per MuJoCo environment and track: 150,000 train, 30,000 validation, 30,000 test physical transitions. Five independent training seeds `41..45`, disjoint paired evaluation worlds `801..820`. Track-specific worlds, schedules, and probe bank are fixed before test scores.
- Training-only normalization; validation-only checkpoint selection by prequential NLL. Fixed 64-probe public state-action bank from independent reference/validation calibration data. Forecast horizons `1,8,24,64`, plus `h=0` current-law diagnostic. Primary C2 score is projected-law Energy Score against a frozen **same-current-belief** comparator, with switch/survival calibration. C3 uses independent target probes at adaptation budgets `4,8,16,32,64`; uncertainty is clustered by training run and independent world sequence.
- Required baselines: pooled MLP, causal history MLP, switching experts, geometric duration, factorization ablation, scratch memory, frozen evolution, and clearly privileged oracle diagnostic. Each deployable baseline requires its own validation checkpoint. Public reward and downstream policy diagnostic stay separate from core likelihood.

## Executed subset on this branch

- Installed versions: Python 3.11.9, torch 2.13.0 CPU, Gymnasium 1.3.0, MuJoCo 3.3.7, NumPy 2.4.6, SciPy 1.17.1, PyYAML 6.0.3. CUDA and ManiSkill were unavailable or not validated.
- `Swimmer-v5` is excluded **before test metrics** because the installed nominal model has zero damping on both controlled DOFs. The exact nominal array and failure reason are recorded in `experiments/rwt/clsm/benchmark_exclusions.json`. No fictitious damping intervention was introduced.
- MuJoCo wrapper uses nominal direct-motor gain and damping arrays, applies law `t` before simulator step `t`, then advances schedule. Its public reward is `-0.001||o||²-0.01||a||²`. The returned `info` is empty. The toy smoke collector is a smooth random public-action policy. It is **not** the full required nominal-controller mixture, so full dataset collection is blocked deliberately.
- Smoke COMP for HalfCheetah generated 32/16/16 public transitions in three disjoint split world IDs. Public `.npz` and manifest have no oracle mask, parameter, age, switch, or physical array. Evaluator oracle files and manifest are in a separate `clsm_evaluator` directory. This is namespace isolation, not operating-system ACL isolation.
- Matched physics audits on eight probes each for HalfCheetah, Walker2d and Reacher found exact repeated-query output (`max abs error=0`), restored gain/damping arrays, and equal declared reward at matched public `(o,a)`. The audit files contain raw pairwise separations. Eight probes do not establish full policy-occupancy coverage.
- Reference-only HalfCheetah CPU initialization collected 32 independent nominal transitions and ran three epochs. It saved a reproducible reference checkpoint. It is not a complete CLSM fit, a selected full checkpoint, or a C1/C2/C3 result.

## Release gate

Full training/evaluation stays closed. The proposal-corrected ancestry, approximate continuous E-step and causal soft prototype updates are implemented, but generalized-EM M-steps, validation checkpoint selection, matched baselines, independent oracle law probes, ManiSkill PD adapter, and hierarchical statistical reports are not. The command without `--reference-only` fails explicitly. No scientific claim is made from the smoke artifacts.
