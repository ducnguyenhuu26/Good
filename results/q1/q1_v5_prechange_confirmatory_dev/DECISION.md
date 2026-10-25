# Q1 pre-change excitation development decision

**Verdict: `Q1_PRECHANGE_PASS`.** This is a separate development protocol. Original fixed-policy and post-change active verdicts remain FAIL.

The run used 32 independent ID-like and 32 OOD backgrounds, 40 training-only calibration backgrounds, all 21 programs, and N=5/10/20/50/100. Bootstrap intervals resample backgrounds.

## Primary results at N=100

| Regime | Block functional-class accuracy | U4 oracle-code skill | Train-matched population skill | Paired U4 difference | Block I_eff(log b) |
|---|---:|---:|---:|---:|---:|
| ID-like | 1 [1, 1] | 0.9939 [0.991, 0.9964] | 0.9693 [0.9581, 0.9794] | 0.0246 [0.01554, 0.03496] | 116.1 [95.19, 139.4] |
| OOD | 1 [1, 1] | 0.9987 [0.9981, 0.9992] | 0.6924 [0.68, 0.7044] | 0.3062 [0.294, 0.3183] | 646.4 [606.6, 686.3] |

## Energy-matched Fisher comparisons

| Regime | Block minus random I_eff | Block minus smooth I_eff | Decision |
|---|---:|---:|---|
| ID-like | 108.6 [89.43, 130.6] | 87.31 [68.46, 108.8] | PASS |
| OOD | 624.1 [580.7, 667.2] | 524.6 [476.3, 571.3] | PASS |

## Baseline functional-class accuracy at N=100 on block schedule

| Regime | V5 recalibrated | V2 | V3 corrected | M4 |
|---|---:|---:|---:|---:|
| ID-like | 1 [1, 1] | 0.4524 [0.4271, 0.4777] | 0.7768 [0.7515, 0.8006] | 0.3006 [0.2604, 0.3408] |
| OOD | 1 [1, 1] | 0.3661 [0.3482, 0.3824] | 0.7455 [0.7202, 0.7708] | 0.25 [0.2351, 0.2649] |

Ordered-pair accuracy and paired noninferiority to the sinusoidal arm are evaluated separately for both orders in `baseline_comparisons.json`. Program-selected effects and oracle-code U1–U4 effect diagnostics are distinct rows in `effect_per_background.csv`.

## Gate results

```json
{
  "protocol": "Q1_PRECHANGE_EXCITATION_CONFIRMATORY_DEVELOPMENT",
  "verdict": "Q1_PRECHANGE_PASS",
  "status": "development-only; protected final test unopened",
  "gates": {
    "block_Ieff_beats_energy_matched_random_and_smooth_both_regimes": true,
    "U4_oracle_effect_beats_train_matched_population_both_regimes": true,
    "U1_U3_effect_noninferior_to_sinusoidal_margin_0.02": true,
    "ordered_pair_accuracy_at_least_0.80_both_regimes": true,
    "corrected_V3_baseline_present": true,
    "at_least_32_independent_backgrounds_per_regime": true,
    "ordered_pair_class_noninferior_to_sinusoidal_margin_0.02": true,
    "U4_effect_skill_positive_both_regimes": true
  },
  "old_Q1_FIXED_POLICY_verdict": "FAIL (immutable)",
  "old_Q1_ACTIVE_EXCITATION_verdict": "FAIL (immutable)",
  "protected_test_used": false
}
```

No protected test split was opened. The protocol does not change the original fixed-policy or post-change active verdicts.
