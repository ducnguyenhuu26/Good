# Frozen V2 basis representational ceiling

Frozen basis checkpoint SHA-256: `e8dda7856e5fa58a2c37842a0ca87c68a7691615a00dcbcc628e5ff441d2fbcf`. Coefficients for each law were fit with 768 transitions; effect skill was scored on 1536 independent transitions per background/intervention with transition noise 0.0. No test split was used.

| Regime | Effect | Frozen-basis ceiling | Learned operators | Zero-effect | Backgrounds |
|---|---|---:|---:|---:|---:|
| IDLIKE_DEVELOPMENT | U3 | 0.985 | -1.032 | 0.000 | 16 |
| IDLIKE_DEVELOPMENT | U4 | -0.381 | -164.753 | 0.000 | 16 |
| BACKGROUND_OOD_DEVELOPMENT | U3 | 0.990 | -3.046 | 0.000 | 16 |
| BACKGROUND_OOD_DEVELOPMENT | U4 | -0.630 | -246.370 | 0.000 | 16 |

Skill is `1 - query MSE / mean(true effect squared)`; zero means no gain over predicting no effect, and negative values are worse. The basis estimate uses separately fit before/after coefficients and is an empirical frozen-representation ceiling under this excitation and sample budget. The operator prediction uses each saved V2 bank with the same frozen basis and the before-law coefficient fit. Noise-free fitting isolates representation and operator error from transition measurement noise.
