# CLSM-R2 Gate A: independent mixed-composition diagnostic

**Decision: FAIL — tested force-conditioned structured representation.** The oracle-G mixed-seen structured positive control is worse than the oracle-conditioned unstructured baseline on registered MSE/signature metrics for composition codes [`101`]; these codes had at least ten independent mixed-training worlds and the paired-world 95% CI excludes zero. This is evidence against the tested structured representation on those compositions.

This is a bounded diagnostic failure for the registered HalfCheetah-v5 intervention family. It does not mean every mixed composition failed: the structured models performed strongly on held-out trajectory occupancy. The falsifying result is the matched-probe domain, where code `101` stayed worse than the oracle-conditioned MLP even after the structured model was trained on mixed worlds. This domain reversal is retained explicitly.

## Frozen design and execution

Protocol was frozen in [`clsm_r2_gate_a_protocol.md`](clsm_r2_gate_a_protocol.md) at commit `b72ceebe82dfb4bed715ae9832954c4a9f36d66e`, before composition scoring. CLSM-R2 dynamics were unchanged; the diagnostic trainer/evaluator was isolated and its checkpoints carry `role=oracle_diagnostic`. Production loaders reject that role. Oracle identities and strengths were read only inside the diagnostic path; the primary learner continues to use unlabeled transitions.

| Split | Independent worlds | Transitions | Use |
|---|---:|---:|---|
| Singleton train | 40 | 4,000 | structured / oracle-conditioned diagnostic training |
| Singleton validation | 10 | 1,000 | validation-based model selection |
| Mixed train | 20 per code | 1,000 per code | oracle-G structured positive control |
| Mixed validation | 8 per code | 400 per code | positive-control model selection |
| Held-out composition trajectories | 40 per code | 2,000 per code | independent-world evaluation |
| Matched state-action probes | same 128 probes × 40 worlds per code | 20,480 total | controlled state/action comparison |
| Confirmatory reserve | 8 per code | 1,600 total | generated and hash-checked; not scored |

All five model families used seeds 4100–4102, the same 4,000 nominal replay transitions, and validation-only checkpoint selection. This produced 15 fitted checkpoints. Best validation NLLs (three seeds) were:

| Model | Best validation NLL per dim, seeds 4100 / 4101 / 4102 |
|---|---|
| `oracle_G_z_xi_singletons` | 0.1491, 0.1508, 0.1515 |
| `learned_G_oracle_z_xi_singletons` | 0.1491, 0.1508, 0.1515 |
| `oracle_G_z_xi_mixed_seen` | -0.4121, -0.4110, -0.4130 |
| `oracle_conditioned_unstructured_MLP` | 0.5446, 0.5405, 0.5434 |
| `pooled_MLP` | 0.5558, 0.5621, 0.5604 |

The mixed-seen decoder therefore completed validation selection on its declared mixed regime. Validation NLLs are regime-specific and must not be compared across singleton vs mixed-trained models as if they shared a validation distribution.

## Held-out predictive performance

Means below are averaged across 40 independent worlds per code, with the three training seeds averaged within each world. NLL is per observation dimension and normalized MSE is under the frozen nominal-reference scale. In the trajectory table the reported signature column is the within-world normalized transition RMSE (the square root of normalized MSE); the distinct fixed-bank law-signature result is measured on the 128 matched probes below. The paired-decision claim uses the matched-probe signature metric, not this duplicated trajectory RMSE. Per-code world-bootstrap intervals are in the linked machine-readable score files.

### Held-out trajectory occupancy

| Model | Code | NLL / dim | Norm. MSE | Signature error |
|---|:---:|---:|---:|---:|
| `oracle_G_z_xi_singletons` | `011` | -0.3370 | 0.0343 | 0.1784 |
| `oracle_G_z_xi_singletons` | `101` | -0.2503 | 0.0430 | 0.2026 |
| `oracle_G_z_xi_singletons` | `110` | -0.2376 | 0.0407 | 0.1978 |
| `oracle_G_z_xi_singletons` | `111` | -0.3733 | 0.0278 | 0.1644 |
| `learned_G_oracle_z_xi_singletons` | `011` | -0.3370 | 0.0343 | 0.1784 |
| `learned_G_oracle_z_xi_singletons` | `101` | -0.2503 | 0.0430 | 0.2026 |
| `learned_G_oracle_z_xi_singletons` | `110` | -0.2376 | 0.0407 | 0.1978 |
| `learned_G_oracle_z_xi_singletons` | `111` | -0.3733 | 0.0278 | 0.1644 |
| `oracle_G_z_xi_mixed_seen` | `011` | -0.3347 | 0.0369 | 0.1847 |
| `oracle_G_z_xi_mixed_seen` | `101` | -0.3816 | 0.0360 | 0.1828 |
| `oracle_G_z_xi_mixed_seen` | `110` | -0.3952 | 0.0327 | 0.1760 |
| `oracle_G_z_xi_mixed_seen` | `111` | -0.5552 | 0.0206 | 0.1399 |
| `oracle_conditioned_unstructured_MLP` | `011` | 0.5163 | 0.0739 | 0.2665 |
| `oracle_conditioned_unstructured_MLP` | `101` | 0.4887 | 0.0539 | 0.2276 |
| `oracle_conditioned_unstructured_MLP` | `110` | 0.4880 | 0.0532 | 0.2249 |
| `oracle_conditioned_unstructured_MLP` | `111` | 0.5276 | 0.0822 | 0.2812 |
| `pooled_MLP` | `011` | 0.5297 | 0.0820 | 0.2815 |
| `pooled_MLP` | `101` | 0.5283 | 0.0808 | 0.2797 |
| `pooled_MLP` | `110` | 0.5217 | 0.0765 | 0.2701 |
| `pooled_MLP` | `111` | 0.5501 | 0.0972 | 0.3064 |

Across trajectory occupancy, singleton structured MSE was lower than the oracle-conditioned MLP for codes 3, 5, and 6; code 7 was not clearly separated. The mixed-seen structured control was also lower than the oracle-conditioned MLP on all four compositions. This is favorable trajectory evidence, but it did not reproduce on the fixed matched probes.

### Matched state-action probes

| Model | Code | NLL / dim | Norm. MSE | Signature error |
|---|:---:|---:|---:|---:|
| `oracle_G_z_xi_singletons` | `011` | 0.1986 | 0.0986 | 0.3139 |
| `oracle_G_z_xi_singletons` | `101` | 0.6915 | 0.1401 | 0.3724 |
| `oracle_G_z_xi_singletons` | `110` | 0.5703 | 0.1336 | 0.3637 |
| `oracle_G_z_xi_singletons` | `111` | 0.7715 | 0.1478 | 0.3823 |
| `learned_G_oracle_z_xi_singletons` | `011` | 0.1986 | 0.0986 | 0.3139 |
| `learned_G_oracle_z_xi_singletons` | `101` | 0.6915 | 0.1401 | 0.3724 |
| `learned_G_oracle_z_xi_singletons` | `110` | 0.5703 | 0.1336 | 0.3637 |
| `learned_G_oracle_z_xi_singletons` | `111` | 0.7715 | 0.1478 | 0.3823 |
| `oracle_G_z_xi_mixed_seen` | `011` | 0.4411 | 0.1049 | 0.3239 |
| `oracle_G_z_xi_mixed_seen` | `101` | 0.8864 | 0.1324 | 0.3624 |
| `oracle_G_z_xi_mixed_seen` | `110` | 0.8107 | 0.1305 | 0.3598 |
| `oracle_G_z_xi_mixed_seen` | `111` | 0.9780 | 0.1390 | 0.3711 |
| `oracle_conditioned_unstructured_MLP` | `011` | 0.6077 | 0.1404 | 0.3730 |
| `oracle_conditioned_unstructured_MLP` | `101` | 0.5665 | 0.1103 | 0.3312 |
| `oracle_conditioned_unstructured_MLP` | `110` | 0.5845 | 0.1234 | 0.3496 |
| `oracle_conditioned_unstructured_MLP` | `111` | 0.6296 | 0.1563 | 0.3931 |
| `pooled_MLP` | `011` | 0.6520 | 0.1718 | 0.4116 |
| `pooled_MLP` | `101` | 0.6530 | 0.1723 | 0.4127 |
| `pooled_MLP` | `110` | 0.6399 | 0.1636 | 0.4025 |
| `pooled_MLP` | `111` | 0.7176 | 0.2204 | 0.4679 |

The code `101` matched probes are the predeclared falsification: singleton- and mixed-seen structured models both underperformed the oracle-conditioned MLP. For mixed-seen structured minus oracle-conditioned MLP, paired world-bootstrap differences were:

| Code | NLL diff [95% CI] | MSE diff [95% CI] | Signature diff [95% CI] |
|:---:|---:|---:|---:|
| `011` | -0.1666 [-0.1767, -0.1575] | -0.0354 [-0.0444, -0.0279] | -0.0490 [-0.0599, -0.0394] |
| `101` | +0.3199 [+0.1949, +0.4505] | +0.0221 [+0.0132, +0.0307] | +0.0312 [+0.0186, +0.0431] |
| `110` | +0.2262 [+0.1166, +0.3414] | +0.0071 [-0.0032, +0.0163] | +0.0102 [-0.0033, +0.0223] |
| `111` | +0.3484 [+0.2108, +0.4913] | -0.0173 [-0.0324, -0.0036] | -0.0221 [-0.0409, -0.0046] |

For code `101`, both error differences are positive and their 95% intervals exclude zero (MSE `+0.0221 [0.0132, 0.0307]`; signature `+0.0312 [0.0186, 0.0431]`). The singleton-trained structured model is similarly worse on code `101` matched probes: MSE `+0.0298 [0.0197, 0.0397]`, signature `+0.0412 [0.0273, 0.0545]`. Its trajectory contrasts go the other way, which establishes a material state-distribution dependence rather than a uniform failure.

The complete paired contrasts for both evaluation domains are recorded in `clsm_r2_gate_a_metrics.json`, including NLL, MSE, signature error, 10,000-world-bootstrap intervals and the descriptive per-world severe-MSE ratio. The confirmatory reserve was not opened.

## Effective-force coverage and simulator checks

Deterministic simulator snapshot replay maximum absolute error was `0`. The diagnostic effective-force coverage audit uses a train-only 95th-percentile leave-one-out nearest-neighbor distance, with forces standardized by singleton-training force SD. This cutoff is descriptive and was not a registered Gate threshold.

| Code | Mixed train worlds / transitions | Matched probes within singleton-train force reference (mean [95% CI]) | Within code-specific mixed-train force reference |
|:---:|---:|---:|---:|
| `011` | 20 / 1000 | 0.998 [0.996, 1.000] | 0.912 [0.887, 0.935] |
| `101` | 20 / 1000 | 0.926 [0.920, 0.932] | 0.856 [0.838, 0.874] |
| `110` | 20 / 1000 | 0.874 [0.862, 0.887] | 0.723 [0.704, 0.742] |
| `111` | 20 / 1000 | 0.966 [0.957, 0.974] | 0.578 [0.539, 0.618] |

For the decisive code `101`, about 92.6% of matched-probe force vectors fall within the singleton-training reference (world-bootstrap 95% CI 92.0–93.2%); about 85.6% fall within the code-specific mixed-training reference (83.8–87.4%). The probe domain is therefore substantially represented in training force space, though not completely. This supports interpreting the code-`101` positive-control deficit as an architecture/fit limitation, while the incomplete support and observed domain reversal limit how broadly to generalize it.

## Full conditional Fisher information

For each mixed code, Fisher is `sum_probe Jᵀ Σ⁻¹J` over the same 128 probes, averaged over 40 independent worlds. All active off-diagonal terms are retained. Inactive coordinates appear as zero rows/columns. The first block below is the singleton-trained oracle-G decoder; complete per-element world-bootstrap CIs and corresponding matrices for all three structured models are in the tracked [`clsm_r2_gate_a_metrics.json`](clsm_r2_gate_a_metrics.json).

### Code `011`

```text
[591.23, -11.16, 0.00]
[-11.16, 234.05, 0.00]
[0.00, 0.00, 0.00]
```

Active-coordinate rank 2 / 2; active eigenvalues 233.70, 591.58; condition number 2.531.

### Code `101`

```text
[614.63, 0.00, -236.73]
[0.00, 0.00, 0.00]
[-236.73, 0.00, 520.28]
```

Active-coordinate rank 2 / 2; active eigenvalues 326.08, 808.84; condition number 2.481.

### Code `110`

```text
[0.00, 0.00, 0.00]
[0.00, 212.02, -128.28]
[0.00, -128.28, 496.83]
```

Active-coordinate rank 2 / 2; active eigenvalues 162.77, 546.08; condition number 3.355.

### Code `111`

```text
[543.45, -10.76, -208.95]
[-10.76, 203.94, -131.65]
[-208.95, -131.65, 508.04]
```

Active-coordinate rank 3 / 3; active eigenvalues 133.49, 373.80, 748.13; condition number 5.604.

The conditional active-set rank is full for each code (2/2, 2/2, 2/2, 3/3). That says the fitted decoder has local sensitivity to active latent strengths on these probes. It does not repair the code-`101` predictive/signature deficit or establish global identifiability.

## Gate decision and scope

**Gate A: FAIL (representation on the registered probe domain).** The oracle-G mixed-seen structured positive control was trained with 20 independent worlds and 1,000 transitions for code `101`, passed validation selection, and was evaluated on 40 independent held-out worlds. On the matched probes it was significantly worse than an equally oracle-conditioned unstructured MLP on both normalized MSE and signature error, while the probe force cloud had substantial singleton and mixed-train support. This meets the frozen representation-failure rule. The trajectory results remain favorable and are reported above; they do not override the registered matched-probe result.

This evidence justifies reviewing the force-response representation. It does not establish which replacement will work, justify modifying CLSM-R2 silently, or authorize proceeding to posterior/generalized-EM validation as if Gate A passed. The confirmatory worlds remain unscored and should remain reserved for a revised, newly frozen experiment.


Manifest fingerprint:

- Source commit: `b72ceebe82dfb4bed715ae9832954c4a9f36d66e`; source fingerprint: `de33c5c27ebf0e489f420eb4048cd206c0d4fa94efa047e3f78b1f51b8f110fa`.
- Every public and evaluator-only split SHA-256 below was rechecked against the generated file (16/16 passed).

| Split | Worlds | Transitions | Public SHA-256 | Evaluator SHA-256 |
|---|---:|---:|---|---|
| `singleton_train` | 40 | 4000 | `3a893eb49fb5cef0bf36582855ec0287966c46e3ab4f449dca251de4ab5fa38b` | `4543a6bdc43a1be82bc5e74a8b900123048147b79c75f7e0bb5bfc74241fbfe0` |
| `singleton_validation` | 10 | 1000 | `d8a1560446674c792ee464ca29adeac0ab8e598c71d6325d04e9071adc95c970` | `fd189fbd32e5b734165c29fa6a8d7cc9bbef2d2a56076d7f51b09d94899e0608` |
| `mixed_train` | 80 | 4000 | `0d0c2b934a7d2a5410cf714e1e2c4919a10778761be5f47c7b770409ede8cc04` | `585531b0c66f859a7ed697dc6877b0c3d7c893f9e41924c0bc0e25f03a84d7e5` |
| `mixed_validation` | 32 | 1600 | `2f200d3119d995f5a01e0dedc114efb1a685c36925422828612b57e69365772f` | `c57af2e100217b0707adc4c64773bda1f541b858b291ca85756309d4fc6afbc3` |
| `composition_diagnostic` | 160 | 8000 | `ceba835b418e5337478ba69f1c4c77ba4a6f2d99b2040e45c2f4e2992f49d98d` | `5aab85956c2ed6fdec7faa635fc84465dff1b8f891880539efe63a9adc886181` |
| `composition_confirmatory_reserved` | 32 | 1600 | `a3e4a5602aa8d0a6489582d774e0a9366c9a6dd91dca91997dfc2960790bee97` | `9af4b9c057f8f13933a70cac196f1822be6b417cde6f753ae7aed9460f2e4f35` |
| `matched_probe_bank` | 0 | 128 | `5dab1819d3b07ec8c45ed3f6e1b3160948a99750d7e821977a8ae6b3cfd05af5` | `5951b390b67379460fef0b9f0374f9fcc788627c7f63694ea0416ec52c313ed9` |
| `matched_probe_outcomes` | 160 | 20480 | `821cca23138db471ff671950879defdc782e66df8bce66b4af793946a21eb6fa` | `f8ca616424c6d18fc2c11b3868425c93544bdcbb7b45fb74030bda61ba4c30f2` |

## Reproducibility artifacts

- Protocol: [`clsm_r2_gate_a_protocol.md`](clsm_r2_gate_a_protocol.md).
- Runner: [`../scripts/run_clsm_r2_gate_a_diagnostic.py`](../scripts/run_clsm_r2_gate_a_diagnostic.py).
- Tracked metrics, paired intervals, full Fisher matrices, split hashes, model selection hashes and decision: [`clsm_r2_gate_a_metrics.json`](clsm_r2_gate_a_metrics.json).
- Full raw dataset manifest and generated arrays remain under `results/rwt/clsm_r2_gates/gate_a_diagnostic/` (ignored experiment artifacts).

The experiment is limited to HalfCheetah-v5, this intervention/strength distribution, this nominal initialization lineage, three optimizer seeds, and the sampled state/action coverage.
