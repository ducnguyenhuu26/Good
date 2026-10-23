# Q1 M1 gate report

Run kind: **DEVELOPMENT**

Python: 3.11.9; NumPy: 2.4.6; PyTorch: 2.13.0+cpu.

Dynamics use dt=0.05, action limits [-1.0,1.0], initial q~N(0,0.25²), v~N(0,0.25²), and sigma=0.0005 velocity noise. Oracle programs are scored against the same continuous adaptation trajectory for each budget.

## Oracle discrimination

| N | Identity accuracy | Functional class accuracy | Mean best-second MSE margin | U1→U2 class acc. | U2→U1 class acc. |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.667 | 0.874 | 1.58e-05 | 1.000 | 1.000 |
| 10 | 0.695 | 0.912 | 1.3e-05 | 1.000 | 1.000 |
| 20 | 0.732 | 0.961 | 1.46e-05 | 1.000 | 1.000 |
| 50 | 0.735 | 0.961 | 1.57e-05 | 1.000 | 1.000 |
| 100 | 0.760 | 0.997 | 1.53e-05 | 1.000 | 1.000 |

Functional equivalence used 256 fixed state-action probes and max-response tolerance 1e-07. The U1→U2 / U2→U1 response RMSE is 0.0053441.

## Decision

This run is **DEVELOPMENT**. Smoke results are not scientific evidence. This validation-only development diagnostic does not use final-test backgrounds or select a checkpoint. Neural stages remain gated on M1.
