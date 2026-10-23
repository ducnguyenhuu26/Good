# Q1 M1 gate report

Run kind: **DEVELOPMENT**

Python: 3.11.9; NumPy: 2.4.6; PyTorch: 2.13.0+cpu.

Dynamics use dt=0.05, action limits [-1.0,1.0], initial q~N(0,0.15²), v~N(0,0.08²), and sigma=0.01 velocity noise. Oracle programs are scored against the same continuous adaptation trajectory for each budget.

## Oracle discrimination

| N | Identity accuracy | Functional class accuracy | Mean best-second MSE margin | U1→U2 class acc. | U2→U1 class acc. |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.246 | 0.304 | 1.51e-05 | 0.500 | 0.625 |
| 10 | 0.339 | 0.423 | 1.11e-05 | 0.781 | 0.750 |
| 20 | 0.384 | 0.490 | 1.09e-05 | 0.812 | 0.875 |
| 50 | 0.426 | 0.536 | 1.16e-05 | 1.000 | 1.000 |
| 100 | 0.461 | 0.598 | 1.13e-05 | 1.000 | 1.000 |

Functional equivalence used 256 fixed state-action probes and max-response tolerance 1e-07. The U1→U2 / U2→U1 response RMSE is 0.0053441.

## Decision

Oracle M1 gate: **FAIL** against predeclared config thresholds. This run is **DEVELOPMENT**. Smoke results are not scientific evidence. This validation-only development diagnostic does not use final-test backgrounds or select a checkpoint. Neural stages remain gated on M1.
