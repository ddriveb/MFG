# Independent repository, annotations and verification

Type: maintenance
Status: claimed
Blocked by: none

## Acceptance criteria

- Bootstrap explicitly selects Python 3.10 and fails on command errors.
- Full unittest suite and minimal config check pass in the new checkout.
- Core simulator edits are comments only, verified by AST comparison.
- README distinguishes current Token-MFG from historical Hedge models.
- No generated artifacts, environments or credentials are committed.
- The user's MFG remote receives verified code and provenance.

## Progress log

### Update: 2026-09-12 — Independent checkout verified

Status: partial

#### Goal
Complete local code migration and explain the new-machine entry points.

#### Changed
Added portable Python 3.10 bootstrap, Chinese code map, batch-information comments,
and seven hash-pinned frozen inputs required by existing loaders. Old scientific
source/configuration and original project directories remain unchanged.

#### Verification
Affected 36 tests passed after fixing missing inputs. The final bootstrap ran
`python -m unittest discover -s tests -v`: 836 tests, OK (skipped=1), 294.695 s;
`python -m mfg_hedge check --config configs/v1_minimal.json`: status ok.
All imported Python ASTs are unchanged. Frozen inputs preserve original hashes.

#### Artifacts
`docs/migration/VALIDATION.md`, `configs/frozen_inputs/manifest.json`,
`docs/migration/migration_change_audit.json`, `docs/NEW_COMPUTER.md`.

#### Decisions and risks
The first run's 16 failures were missing frozen inputs, not simulation changes.
Large artifacts and r2 continuation ledgers remain separate; no formal run started.

#### Next
Publish the verified commit to the user's MFG repository, then record completion.

### Update: 2026-09-12 — Migration started

Status: partial

#### Goal
Prepare an independent MFG checkout.

#### Changed
Imported the reviewed source/config/test/document file set into a fresh clone.

#### Verification
Checked every imported source against its reviewed SHA-256 before copying.

#### Artifacts
`docs/migration/source_import_manifest.json`.

#### Decisions and risks
Formal checkpoint/identity/budget ledgers stay in the old research-data backup.
No qualification or holdout run is started by migration tooling.

#### Next
Add bootstrap, documentation and annotations, then verify.
