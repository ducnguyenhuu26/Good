# Gate B diagnostic coverage and missing Fisher analysis

Gate A failed, so this run is diagnostic-only. The first 64 snapshots of a shared 128 state-action bank were available to the known-code filter; the final 64 common probes were held out. Strength labels were used for scoring only. `strength_reference.json` records per-world action norm, nonzero-action fraction, and joint speed.

The numerical reference was compared at preselected lower and higher Gauss-Hermite orders. Its maximum posterior xi-mean discrepancy was 0.79346, far above the 0.01 tolerance; posterior-reference-dependent results are therefore descriptive only.

A parameter Fisher-information matrix was **not computed** in this Gate B diagnostic. The filename is retained to match the requested deliverable layout, but action/velocity coverage summaries do not substitute for Fisher information. This is an additional incomplete Gate B deliverable.
