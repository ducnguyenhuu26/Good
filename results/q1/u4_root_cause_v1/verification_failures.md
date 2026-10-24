# Resolved verification failure

Command: `python -m pytest -q tests/test_q1_u4_diagnostics.py`

Exit code: `1`.

Failure excerpt:

```text
FAILED tests/test_q1_u4_diagnostics.py::test_config_uses_only_train_validation_and_fixed_loss_scale_variants
assert cfg["oracle_true_programs"] == [[], [2], [3], [2,3], [3,2]]
E assert [[], [0], [2], [3], [2, 3], [3, 2]] == [[], [2], [3], [2, 3], [3, 2]]
1 failed, 12 passed
```

The config correctly included the U1 physical-oracle control; the test expectation was stale. The test was corrected. Final reruns passed: 13 diagnostic tests, 21 existing focused Q1 tests, and `compileall` (all exit code 0).
