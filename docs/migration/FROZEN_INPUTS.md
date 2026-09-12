# Why seven historical JSON files are included as controlled inputs

The first clean-checkout test run found 16 errors: existing baseline/candidate
loaders require T3A validation/occupancy and the requested-price frozen summary.
These are runtime model inputs, not optional plots. Dropping them prevents a
source-only clone from executing the existing test and baseline contracts.

Exactly seven files (about 2.46 MiB) are retained under `configs/frozen_inputs/`,
with byte-identical SHA-256 provenance. Bootstrap restores them to their original
ignored `artifacts/<run>/` paths, only if missing or identical. No algorithm,
default policy, metric, seed or budget is changed to make tests pass.

This promotes fixed runtime inputs to versioned configuration. Generated run
output remains excluded by AGENTS.md/.gitignore. The seven inputs are not r2
checkpoint/identity ledgers, new experiments, or qualification evidence upgrades.
The large old artifact tree still requires an independent backup for continuation.
