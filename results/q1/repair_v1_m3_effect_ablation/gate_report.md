# Q1 paired-effect M3 ablation

This is a development training run using training-background simulator counterfactuals. The after-only beta=0 control and effect variants share initialization, code map, state-action examples, optimizer updates, and minibatch seeds. Beta selection uses validation only. No test environments were used.

Validation-selected beta: 0.25. Variant gate statuses: beta_0.00=FAIL, beta_0.25=FAIL, beta_1.00=FAIL. See per-code CIs in `metrics.json` and probe-level rows in `per_code_metrics.csv`.
