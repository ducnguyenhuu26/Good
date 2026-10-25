# Corrected V3 baseline reanalysis

V3 candidate selection uses `argmin(marginal_score)` on the unchanged Phase B development trajectories. The historical Phase B directory remains unchanged. See `summary.json` for corrected V3 metrics and recomputed V3-dependent gates.

{
  "protocol": "Q1_FIXED_POLICY",
  "selected_rule": "old_plus_eb",
  "selected_from_inner_only": true,
  "v5_fixed_gate": "FAIL",
  "gates": {
    "u4_positive_and_beats_population_both_regimes": false,
    "u3_u4_positive_both_regimes": true,
    "u1_u3_effect_noninferior_to_v3_margin_0.02": true,
    "ordered_pair_accuracy_at_least_0.80": true,
    "ordered_pair_class_noninferior_to_v3_margin_0.02": true,
    "class_accuracy_beats_v2_both_regimes": true,
    "ordered_pair_class_noninferior_to_V3_margin_0.02": true
  },
  "u4_gate_pass": false,
  "notes": [
    "All baseline rows use the same generated old/adaptation/query observations.",
    "V2/V3/M4 are frozen checkpoints from repository training outputs; evaluation backgrounds and trajectories are new.",
    "Population mean and identity are V5-aligned reference baselines; direct-context is prediction-only and has no class estimate.",
    "Protected test split not accessed.",
    "V3 rows were independently re-scored by minimum marginal_score; the original Phase B directory is unchanged."
  ],
  "v3_scorer_corrected": true,
  "historical_phase_b_artifacts_unchanged": true
}
