# Preserved failed v2 run

The first pipeline attempt was stopped before evaluation completed. A synthetic weak-label operator transfer test exposed an ordering mismatch in the helper that flattened basis output rows relative to the observed `(N,2)` delta-v targets. The resulting old posterior was invalid, so no scientific metrics from this run are interpretable. Artifacts are retained to preserve the failed execution and pretraining provenance. The fix is in `q1_v2/training.py`; rerun under the canonical dev1 directory only after the test passes.
