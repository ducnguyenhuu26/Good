# CLSM-R3 R1 decisive decoder gate

**Decision: FAIL**

R1 scores use four new mixed compositions, independent physical worlds, and a fixed-state matched probe bank. The previously scored Gate-A diagnostic remains development evidence. Oracle labels were evaluator-only and were opened after the checkpoint freeze.

## Registered decision criteria

| Criterion | Result |
|---|---:|
| `mixed_seen_all_code_mse_and_signature_noninferiority` | FAIL |
| `singleton_all_code_mse_and_signature_noninferiority` | FAIL |
| `singleton_macro_mse_ci_upper_below_zero` | FAIL |
| `no_meaningful_trajectory_regression_over_5pct` | FAIL |

## Training and capacity

Residual width: 244; R3 trainable parameters: 76406; oracle MLP target: 76187.
Training runs: 6; seeds: 4100, 4101, 4102. Checkpoint selection used only regime-specific validation standardized Gaussian NLL.

## Paired world intervals

The table reports paired world-bootstrap 95% CIs, and the registered one-sided upper bounds are in `r1_decision.json`.

| Domain | Training regime | Code | Metric | R3 − matched MLP | 95% CI | 5% margin | One-sided upper |
|---|---|---:|---|---:|---:|---:|---:|
| trajectory | R3_mixed_seen | 3 | normalized_mse | 0.00724382 | [0.00488155, 0.0102509] | 0.00165032 | 0.00968323 |
| trajectory | R3_mixed_seen | 3 | normalized_signature_error | 0.0167772 | [0.0128333, 0.0210262] | 0.00870766 | 0.0202824 |
| trajectory | R3_mixed_seen | 5 | normalized_mse | 0.0148205 | [0.00498758, 0.0285312] | 0.00183904 | 0.0258025 |
| trajectory | R3_mixed_seen | 5 | normalized_signature_error | 0.025756 | [0.0137828, 0.0413003] | 0.00904721 | 0.038546 |
| trajectory | R3_mixed_seen | 6 | normalized_mse | 0.00444269 | [0.00319221, 0.00595741] | 0.00131942 | 0.00567942 |
| trajectory | R3_mixed_seen | 6 | normalized_signature_error | 0.012238 | [0.00958571, 0.0152718] | 0.00791639 | 0.0147552 |
| trajectory | R3_mixed_seen | 7 | normalized_mse | 0.00103806 | [0.000226159, 0.00182682] | 0.000757806 | 0.00169456 |
| trajectory | R3_mixed_seen | 7 | normalized_signature_error | 0.00395266 | [0.000960109, 0.00676237] | 0.0059743 | 0.00631755 |
| matched_probe | R3_mixed_seen | 3 | normalized_mse | 0.0596453 | [0.0554078, 0.0636867] | 0.00609379 | 0.0631407 |
| matched_probe | R3_mixed_seen | 3 | normalized_signature_error | 0.0767111 | [0.0715336, 0.0815352] | 0.0174489 | 0.0807961 |
| matched_probe | R3_mixed_seen | 5 | normalized_mse | 0.0811279 | [0.0729911, 0.0890606] | 0.00556116 | 0.0878281 |
| matched_probe | R3_mixed_seen | 5 | normalized_signature_error | 0.10473 | [0.0945974, 0.114423] | 0.0166589 | 0.112919 |
| matched_probe | R3_mixed_seen | 6 | normalized_mse | 0.0672414 | [0.0601353, 0.074152] | 0.00516223 | 0.0730856 |
| matched_probe | R3_mixed_seen | 6 | normalized_signature_error | 0.0911096 | [0.0820694, 0.099998] | 0.0160572 | 0.0986319 |
| matched_probe | R3_mixed_seen | 7 | normalized_mse | 0.0841683 | [0.0756072, 0.0920684] | 0.0059037 | 0.0908569 |
| matched_probe | R3_mixed_seen | 7 | normalized_signature_error | 0.105549 | [0.0956326, 0.114953] | 0.0171689 | 0.11362 |
| trajectory | R3_singleton | 3 | normalized_mse | 0.0150809 | [0.010182, 0.019649] | 0.00303146 | 0.0189966 |
| trajectory | R3_singleton | 3 | normalized_signature_error | 0.0313197 | [0.0224611, 0.0397352] | 0.0120378 | 0.0385171 |
| trajectory | R3_singleton | 5 | normalized_mse | 0.0222265 | [0.0107147, 0.0322163] | 0.00286084 | 0.0306655 |
| trajectory | R3_singleton | 5 | normalized_signature_error | 0.0453677 | [0.028511, 0.0605742] | 0.0115153 | 0.058298 |
| trajectory | R3_singleton | 6 | normalized_mse | 0.0320566 | [0.0234568, 0.0406942] | 0.00253333 | 0.0392309 |
| trajectory | R3_singleton | 6 | normalized_signature_error | 0.0613616 | [0.0467279, 0.0754754] | 0.0110228 | 0.0731955 |
| trajectory | R3_singleton | 7 | normalized_mse | 0.0100882 | [0.000527602, 0.0187609] | 0.00369826 | 0.0175538 |
| trajectory | R3_singleton | 7 | normalized_signature_error | 0.0220943 | [0.0068937, 0.0360336] | 0.0132664 | 0.0339068 |
| matched_probe | R3_singleton | 3 | normalized_mse | 0.0619748 | [0.0506495, 0.0722594] | 0.00678541 | 0.0707992 |
| matched_probe | R3_singleton | 3 | normalized_signature_error | 0.0771602 | [0.0632478, 0.0896263] | 0.0183214 | 0.0877695 |
| matched_probe | R3_singleton | 5 | normalized_mse | 0.11076 | [0.0977528, 0.123519] | 0.00581374 | 0.121541 |
| matched_probe | R3_singleton | 5 | normalized_signature_error | 0.135071 | [0.120102, 0.149543] | 0.0169906 | 0.14732 |
| matched_probe | R3_singleton | 6 | normalized_mse | 0.127775 | [0.11335, 0.142221] | 0.00568396 | 0.139978 |
| matched_probe | R3_singleton | 6 | normalized_signature_error | 0.151898 | [0.136869, 0.166302] | 0.016824 | 0.164271 |
| matched_probe | R3_singleton | 7 | normalized_mse | 0.125156 | [0.100392, 0.145394] | 0.00870511 | 0.142642 |
| matched_probe | R3_singleton | 7 | normalized_signature_error | 0.130732 | [0.106746, 0.150557] | 0.0206568 | 0.147809 |

## Per composition predictive metrics

Means and 95% world-bootstrap CIs average training seeds within each independent world before bootstrapping. Seed-specific contrasts are retained in the decision JSON.

| Model | Domain | Code | NLL / dim (95% CI) | Normalized MSE (95% CI) | Signature RMSE (95% CI) | Worlds |
|---|---|---:|---:|---:|---:|---:|
| R2_mixed_seen | trajectory | 3 | -0.33741 [-0.40611, -0.26029] | 0.035695 [0.029462, 0.042952] | 0.1818 [0.16669, 0.19841] | 40 |
| R2_mixed_seen | trajectory | 5 | -0.36621 [-0.44046, -0.2897] | 0.039024 [0.031717, 0.046833] | 0.18884 [0.17077, 0.20739] | 40 |
| R2_mixed_seen | trajectory | 6 | -0.37666 [-0.43524, -0.31528] | 0.032783 [0.028331, 0.037566] | 0.17658 [0.16424, 0.18935] | 40 |
| R2_mixed_seen | trajectory | 7 | -0.61042 [-0.63684, -0.58088] | 0.015732 [0.013461, 0.018283] | 0.12219 [0.11379, 0.13115] | 40 |
| R2_mixed_seen | matched_probe | 3 | 0.55715 [0.55151, 0.56329] | 0.11078 [0.11019, 0.11143] | 0.33283 [0.33195, 0.33379] | 40 |
| R2_mixed_seen | matched_probe | 5 | 0.95264 [0.84135, 1.0648] | 0.12907 [0.12319, 0.13501] | 0.35829 [0.35009, 0.36654] | 40 |
| R2_mixed_seen | matched_probe | 6 | 0.83052 [0.74986, 0.91369] | 0.12629 [0.1217, 0.13105] | 0.35475 [0.34831, 0.36135] | 40 |
| R2_mixed_seen | matched_probe | 7 | 0.96481 [0.84721, 1.0857] | 0.13277 [0.12616, 0.13965] | 0.36316 [0.35413, 0.37245] | 40 |
| R2_singleton | trajectory | 3 | -0.34201 [-0.38721, -0.29378] | 0.03295 [0.027444, 0.03905] | 0.17503 [0.16073, 0.19013] | 40 |
| R2_singleton | trajectory | 5 | -0.2512 [-0.30176, -0.1978] | 0.044924 [0.038081, 0.05242] | 0.20552 [0.18985, 0.2221] | 40 |
| R2_singleton | trajectory | 6 | -0.19238 [-0.24747, -0.13595] | 0.042958 [0.037946, 0.048137] | 0.20335 [0.19094, 0.21576] | 40 |
| R2_singleton | trajectory | 7 | -0.42324 [-0.44874, -0.39524] | 0.021976 [0.019429, 0.024798] | 0.14567 [0.1373, 0.15451] | 40 |
| R2_singleton | matched_probe | 3 | 0.22275 [0.21959, 0.22613] | 0.097327 [0.096841, 0.097851] | 0.31196 [0.31118, 0.3128] | 40 |
| R2_singleton | matched_probe | 5 | 0.63525 [0.54468, 0.73023] | 0.12485 [0.11837, 0.13165] | 0.35198 [0.34273, 0.36158] | 40 |
| R2_singleton | matched_probe | 6 | 0.5205 [0.4502, 0.59249] | 0.12026 [0.11488, 0.12581] | 0.34585 [0.33805, 0.35378] | 40 |
| R2_singleton | matched_probe | 7 | 0.65291 [0.55098, 0.75622] | 0.12968 [0.12187, 0.13761] | 0.35843 [0.34765, 0.36932] | 40 |
| R3_mixed_seen | trajectory | 3 | 0.49623 [0.48566, 0.50938] | 0.04025 [0.031989, 0.050527] | 0.19093 [0.17329, 0.21103] | 40 |
| R3_mixed_seen | trajectory | 5 | 0.51052 [0.49009, 0.53732] | 0.051601 [0.03553, 0.072714] | 0.2067 [0.17988, 0.23823] | 40 |
| R3_mixed_seen | trajectory | 6 | 0.4841 [0.47823, 0.49045] | 0.030831 [0.02624, 0.035788] | 0.17057 [0.15778, 0.18369] | 40 |
| R3_mixed_seen | trajectory | 7 | 0.46537 [0.46216, 0.46883] | 0.016194 [0.013686, 0.018905] | 0.12344 [0.11411, 0.13312] | 40 |
| R3_mixed_seen | matched_probe | 3 | 0.67683 [0.67162, 0.68222] | 0.18152 [0.17743, 0.18575] | 0.42569 [0.4209, 0.43061] | 40 |
| R3_mixed_seen | matched_probe | 5 | 0.6905 [0.6823, 0.69872] | 0.19235 [0.18592, 0.19881] | 0.43791 [0.43054, 0.44527] | 40 |
| R3_mixed_seen | matched_probe | 6 | 0.66247 [0.65534, 0.66955] | 0.17049 [0.16489, 0.17604] | 0.41225 [0.40544, 0.41895] | 40 |
| R3_mixed_seen | matched_probe | 7 | 0.70285 [0.69406, 0.71147] | 0.20224 [0.19533, 0.209] | 0.44893 [0.44107, 0.45653] | 40 |
| R3_singleton | trajectory | 3 | 0.5454 [0.53692, 0.55437] | 0.07571 [0.068955, 0.082863] | 0.27208 [0.26001, 0.28439] | 40 |
| R3_singleton | trajectory | 5 | 0.54983 [0.53726, 0.56357] | 0.079443 [0.069371, 0.09048] | 0.27567 [0.25821, 0.29419] | 40 |
| R3_singleton | trajectory | 6 | 0.55405 [0.54126, 0.56743] | 0.082723 [0.072517, 0.093429] | 0.28182 [0.26442, 0.2995] | 40 |
| R3_singleton | trajectory | 7 | 0.55599 [0.54829, 0.56393] | 0.084053 [0.077934, 0.09036] | 0.28742 [0.2766, 0.2982] | 40 |
| R3_singleton | matched_probe | 3 | 0.69822 [0.68817, 0.70873] | 0.19768 [0.18968, 0.20604] | 0.44359 [0.43468, 0.45288] | 40 |
| R3_singleton | matched_probe | 5 | 0.73458 [0.72082, 0.74913] | 0.22703 [0.216, 0.23866] | 0.47488 [0.46316, 0.48705] | 40 |
| R3_singleton | matched_probe | 6 | 0.75258 [0.73382, 0.77172] | 0.24145 [0.22651, 0.25671] | 0.48838 [0.47314, 0.5037] | 40 |
| R3_singleton | matched_probe | 7 | 0.82531 [0.8013, 0.84957] | 0.29926 [0.28011, 0.31858] | 0.54387 [0.52619, 0.56142] | 40 |
| oracle_conditioned_MLP_mixed_seen | trajectory | 3 | 0.46124 [0.45268, 0.47163] | 0.033006 [0.026802, 0.040542] | 0.17415 [0.15893, 0.19115] | 40 |
| oracle_conditioned_MLP_mixed_seen | trajectory | 5 | 0.46653 [0.45565, 0.47923] | 0.036781 [0.02891, 0.045986] | 0.18094 [0.16229, 0.20153] | 40 |
| oracle_conditioned_MLP_mixed_seen | trajectory | 6 | 0.45219 [0.44726, 0.45754] | 0.026388 [0.022809, 0.030261] | 0.15833 [0.14749, 0.16966] | 40 |
| oracle_conditioned_MLP_mixed_seen | trajectory | 7 | 0.43669 [0.43353, 0.44022] | 0.015156 [0.012856, 0.017717] | 0.11949 [0.11061, 0.12904] | 40 |
| oracle_conditioned_MLP_mixed_seen | matched_probe | 3 | 0.58381 [0.58174, 0.58621] | 0.12188 [0.12037, 0.12362] | 0.34898 [0.34686, 0.35144] | 40 |
| oracle_conditioned_MLP_mixed_seen | matched_probe | 5 | 0.56917 [0.56554, 0.57412] | 0.11122 [0.10859, 0.11482] | 0.33318 [0.32944, 0.3382] | 40 |
| oracle_conditioned_MLP_mixed_seen | matched_probe | 6 | 0.55825 [0.55566, 0.5612] | 0.10324 [0.10136, 0.10539] | 0.32114 [0.31828, 0.3244] | 40 |
| oracle_conditioned_MLP_mixed_seen | matched_probe | 7 | 0.57873 [0.57525, 0.5827] | 0.11807 [0.11554, 0.12095] | 0.34338 [0.3398, 0.34742] | 40 |
| oracle_conditioned_MLP_singleton | trajectory | 3 | 0.49801 [0.48734, 0.51079] | 0.060629 [0.052879, 0.06992] | 0.24076 [0.22613, 0.2569] | 40 |
| oracle_conditioned_MLP_singleton | trajectory | 5 | 0.49331 [0.47921, 0.51192] | 0.057217 [0.046987, 0.070715] | 0.23031 [0.21231, 0.25169] | 40 |
| oracle_conditioned_MLP_singleton | trajectory | 6 | 0.4845 [0.47679, 0.49314] | 0.050667 [0.045063, 0.056947] | 0.22046 [0.20845, 0.23372] | 40 |
| oracle_conditioned_MLP_singleton | trajectory | 7 | 0.51622 [0.50465, 0.53004] | 0.073965 [0.06557, 0.084006] | 0.26533 [0.25185, 0.28081] | 40 |
| oracle_conditioned_MLP_singleton | matched_probe | 3 | 0.60119 [0.59003, 0.61313] | 0.13571 [0.12757, 0.14443] | 0.36643 [0.35599, 0.37759] | 40 |
| oracle_conditioned_MLP_singleton | matched_probe | 5 | 0.5745 [0.56702, 0.58333] | 0.11627 [0.11084, 0.12269] | 0.33981 [0.33228, 0.34857] | 40 |
| oracle_conditioned_MLP_singleton | matched_probe | 6 | 0.57124 [0.5659, 0.57758] | 0.11368 [0.1098, 0.11829] | 0.33648 [0.33095, 0.34301] | 40 |
| oracle_conditioned_MLP_singleton | matched_probe | 7 | 0.65382 [0.63451, 0.67662] | 0.1741 [0.1601, 0.19061] | 0.41314 [0.39799, 0.43048] | 40 |

## Singleton macro and trajectory check

```json
{
  "singleton_macro": {
    "normalized_mse": {
      "mean_difference": 0.10641644698126623,
      "paired_stratified_world_ci95": [
        0.09814167270145469,
        0.11417928641994335
      ]
    },
    "normalized_signature_error": {
      "mean_difference": 0.1237152825344475,
      "paired_stratified_world_ci95": [
        0.11514756574461377,
        0.13171482319836342
      ]
    }
  },
  "trajectory_regression_checks": [
    {
      "against": "R2_singleton",
      "mean_difference": 0.0447805262418157,
      "macro_baseline_mean": 0.03570194735705939,
      "meaningful_deterioration_margin": 0.0017850973678529695,
      "difference_minus_margin_ci95": [
        0.04007478828753633,
        0.04592266909037188
      ]
    },
    {
      "against": "oracle_conditioned_MLP_singleton",
      "mean_difference": 0.019863046595814053,
      "macro_baseline_mean": 0.060619427003061034,
      "meaningful_deterioration_margin": 0.003030971350153052,
      "difference_minus_margin_ci95": [
        0.012182063015202737,
        0.02113402584993646
      ]
    }
  ]
}
```

## Mixed-seen variance calibration

The ratios and marginal coverage below are descriptive; no post-hoc threshold was used to alter the registered decision.

```json
{
  "trajectory/code_3": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.10316154829823801,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9959117647058823,
    "worlds": 40
  },
  "matched_probe/code_3": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.4643593847480006,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9581380208333334,
    "worlds": 40
  },
  "trajectory/code_5": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.13173892986063176,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9929509803921569,
    "worlds": 40
  },
  "matched_probe/code_5": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.49170123418132305,
    "mean_seed_and_world_marginal_90pct_coverage": 0.957797181372549,
    "worlds": 40
  },
  "trajectory/code_6": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.07890444405959371,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9980098039215686,
    "worlds": 40
  },
  "matched_probe/code_6": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.43564497556412257,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9637408088235296,
    "worlds": 40
  },
  "trajectory/code_7": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.041434056632177405,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9992549019607843,
    "worlds": 40
  },
  "matched_probe/code_7": {
    "mean_seed_and_world_standardized_residual_variance_ratio": 0.5164092183578413,
    "mean_seed_and_world_marginal_90pct_coverage": 0.9563878676470589,
    "worlds": 40
  }
}
```

## Training checkpoints

| Regime | Seed | Best epoch | Best validation standardized NLL | Trainable parameters | Runtime seconds |
|---|---:|---:|---:|---:|---:|
| singleton | 4100 | 100 | 0.6630701245228372 | 76406 | None |
| singleton | 4101 | 100 | 0.6610537541942954 | 76406 | None |
| singleton | 4102 | 100 | 0.6616157368386342 | 76406 | None |
| mixed_seen | 4100 | 100 | 0.49455487602961606 | 76406 | None |
| mixed_seen | 4101 | 100 | 0.4958548298403269 | 76406 | None |
| mixed_seen | 4102 | 100 | 0.4920578153242433 | 76406 | None |

Per-seed runtime was not retained after checkpoint resume; validation NLL and best epoch are available in checkpoint payloads. No runtime claim is made.

## CPU scoring and memory

```json
{
  "wall_seconds": 17.061850300000515,
  "process_cpu_seconds": 54.109375,
  "process_peak_rss_bytes": 349147136,
  "per_world_fields": [
    "amortized_scoring_wall_seconds_per_world",
    "amortized_scoring_cpu_seconds_per_world"
  ],
  "rss_semantics": "process peak working set observed during evaluation; stored on every per-world score row",
  "per_world_time_semantics": "model-call time divided by number of independent worlds in that score batch"
}
```

## Limits

This is an oracle-conditioned diagnostic of representation and compositional transfer. It does not establish that an unlabeled learner recovers the mechanism posterior. A failed locked criterion halts the CLSM-R3 family attempt; later inference and generalized-EM gates must not be run after R1 failure.

Fresh dataset and code freeze manifest: see the sanitized published preregistration and result manifest.
