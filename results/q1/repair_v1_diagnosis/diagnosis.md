# Q1 feasibility diagnosis (P1)

Validation environments: 16. No training was performed. Mean M4 functional-class accuracy is 0.352; median signed true-vs-wrong-class margin is -0.10066.

First repair branch: **O (operator/effect fidelity)**, targeting codes [2, 3]. M4 improves single-code errors and ordered-pair query MSE, but held-out class identification remains weak; S is not supported because true-code candidates are often not the best predictive candidates. Representation is not proven fully solved. The persistence baseline is the frozen M2 prediction from the old context. Per-code M3/M4 effects are in `per_code_metrics.csv`; representability and ranking evidence is in `per_budget_regime_metrics.csv`; ordered-pair outcomes are in `composition_diagnostics.csv`.

All validation results are development evidence.
