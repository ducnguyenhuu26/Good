# Q1 development pilot

This report records one CPU development seed from `results/q1/cpu_confirmatory_impl/`, generated with implementation commit `eb9f5bd`. It is a pipeline and gate diagnostic, not a confirmatory research result. The run used Python 3.11.9, PyTorch 2.13.0+cpu, seed 20261023, `dt=0.05`, `sigma=0.0005`, 48 train environments, 16 validation environments, four held-out test backgrounds per test region, and all five adaptation budgets. M2/M3/M4 training and ranking use the fixed-variance delta-v Gaussian likelihood. The M4 test count is deliberately reduced in `configs/q1_cpu.yaml`; `configs/q1_confirmatory.yaml` retains 24 test backgrounds and a five-seed protocol.

## Stage gates

| Stage | Result | Evidence |
|---|---|---|
| M1 simulator and oracle | PASS | 11 focused Q1 tests passed. The validation oracle reached 0.961 functional-class accuracy at N=20 and 0.997 at N=100. Both noncommuting orders were recovered at 1.000 class accuracy. |
| M2 context dynamics | PASS | Validation delta-v MSE was `2.6570e-5`, versus `1.1513e-4` context-free (76.92% relative improvement). Context-swap response RMSE was `0.001164`. Open-loop state RMSE was 0.0116, 0.0189, and 0.0301 at horizons 5, 10, and 20. Fixed-variance validation NLL was -3.553 per delta-v dimension. |
| M3 primitive transfer | PASS | Mean validation normalized operator MSE `0.03846`, persistence `0.11306`, and wrong-code control `0.16986`; minimum relative gain over both controls was 65.98%. Minimum pairwise functional response RMSE was `0.004396`. Per-code actual delta-v MSE and fixed-variance NLL are in the metric file. |
| M4 composition and inference | FAIL | Held-out composition regimes averaged 0.256 functional-class accuracy, below the predeclared 0.50 threshold. The best fresh-query MSE gain over direct context adaptation was 83.09%, but that alone does not pass M4. |

M2 support-size MSE was `3.0096e-5` at 24 transitions, `2.6570e-5` at 64, and `2.7206e-5` at 256, so the larger support budget did not materially change validation error in this seed.

## M4 split results

| Regime | Mean class accuracy over N | Mean learned-search query MSE | Mean direct-context query MSE |
|---|---:|---:|---:|
| ID | 0.392 | `6.34e-6` | `7.14e-5` |
| Composition OOD | 0.338 | `1.20e-5` | `6.27e-5` |
| Background OOD | 0.388 | `5.67e-5` | `2.00e-4` |
| Joint OOD | 0.175 | `6.78e-5` | `1.48e-4` |

These averages pool five adaptation budgets but only four test backgrounds and one training seed. Program ID accuracy is lower than functional-class accuracy because some programs are observationally equivalent. The learned search has lower fresh-query MSE than the direct context optimizer in this pilot, while its held-out program classification remains weak. No claim of generalization is made from these observations.

M4 selected a length penalty of zero using validation backgrounds and permitted validation compositions. It searched 21 candidates, compared with oracle-identity, simulator-oracle, single-only, persistence, and same-budget direct-context baselines. The full per-regime/per-budget confusion matrices, NLL-based inference costs, query errors, and rollout RMSE values are in `results/q1/cpu_confirmatory_impl/m4/metrics.json`; plots are in `results/q1/cpu_confirmatory_impl/figures/`.

## Decision

M1–M3 allow M4 development to run. The current M4 gate fails its functional-class threshold despite a lower fresh-query prediction error than the direct-context baseline. The next Q1 step is to inspect the saved confusion matrices and failed M4 diagnostics, then run the explicit five-seed confirmatory protocol only after that development issue is addressed. No Q2 planning or control-return conclusion follows from this pilot.
