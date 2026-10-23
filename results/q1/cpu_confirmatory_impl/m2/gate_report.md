# Q1 M2 gate report

Status: **PASS**.

Validation context MSE: 2.65704e-05; context-free MSE: 0.000115131; relative improvement: 76.922%.

Context-swap response RMSE: 0.0011640819320746232. Thresholds: 5.0% relative improvement and 0.0001 response RMSE.

M2 uses training-only normalization and independent support/query trajectories. The encoder and executor are frozen in `context_model.pt`; held-out test environments were not used. A failed gate blocks M3/M4.
