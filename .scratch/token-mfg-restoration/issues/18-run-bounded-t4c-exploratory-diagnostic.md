# Run a bounded T4c exploratory diagnostic

Type: narrow implementation and experiment

Status: resolved

Blocked by: none

## Goal

Produce a fast trend diagnostic after the user cancelled the multi-hour formal
run, without presenting the reduced sample as ADR-0025 qualification.

## Frozen exploratory profile

- 1,024 deterministic identities.
- At most 6 rounds.
- Panel floor 4.
- `beta=1.0`, `eta=0.2`, existing physics and nine retained bins unchanged.
- At most 30,720 scheduler calls.
- Fresh run ID `token-t4c-exploratory-20260908-r1`.
- Output label and manifest must explicitly say `exploratory`; no BR, regret,
  Nash, MFG, or formal ADR-0025 claim.

## Acceptance criteria

1. Reduced parameters and terminal nonconvergence label are recorded exactly
   in the artifact rather than using formal defaults.
2. Focused tests pass before launch; full tests and config check run once after.
3. The run ends within its 30,720-call budget and writes no partial directory.
4. The result reports per-round residual/support/physics trends and its reduced
   statistical limitation.

## Progress log

### Update: 2026-09-08 — Claimed reduced T4c diagnostic

Status: partial

#### Goal

Replace the cancelled multi-hour run with a roughly 30-minute exploratory run.

#### Changed

- This ticket freezes 1,024 identities, 6 rounds, and panel floor 4.

#### Verification

- A real 16-identity timing probe measured about 0.0609 seconds per attempted
  scheduler call.

#### Artifacts

None yet.

#### Decisions and risks

- The reduced panel floor is exploratory only and cannot qualify a supported
  fixed point under ADR-0025.

#### Next

Make the existing parameterized runner report reduced-profile provenance
truthfully, then execute it.

### Update: 2026-09-08 — Completed bounded exploratory run

Status: completed

#### Goal

Run a bounded T4c trend diagnostic after cancelling the undisclosed multi-hour
formal profile.

#### Changed

- The parameterized runner now distinguishes `formal_adr0025` from
  `exploratory` executions.
- Per-round and total call limits, panel floor, episode count, maximum rounds,
  and the `not_converged_<rounds>` terminal label are derived from the actual
  execution profile and recorded in the artifact manifest.
- The exploratory profile used 1,024 identities, six rounds, panel floor 4,
  and the existing `beta=1.0`, `eta=0.2`, physics, bins, and CRN contracts.

#### Verification

- Real Red: the pre-existing runner omitted `execution_profile` and recorded
  formal limits for reduced runs; the new assertion failed with
  `KeyError: 'execution_profile'`.
- Focused T4c suite: 12/12 passed.
- Full suite: 597/597 passed in 151.104 seconds.
- Config check: exit 0, `status: ok`.
- The run used 27,600/30,720 calls, with 4,600 calls in each of six rounds.
- All nine retained bins were active in every round. The last-round effective
  sample count ranged from 6 to 135; no physical panel failed.
- Weighted policy L1 residual fell monotonically from 0.301464 to 0.085973;
  maximum active-bin residual fell to 0.117809 and population occupancy
  residual ended at 0.065830. It did not meet the 0.01 tolerance.
- Last-round mean latency was 6.826988, P99 was 27.862127, replay rate was
  0.0346651, hedge launches were 7,046, total executed work was 401,886.8001,
  and wasted work was 1,786.9150.

#### Artifacts

- `artifacts/token-t4c-exploratory-20260908-r1/`

#### Decisions and risks

- The result is `not_converged_6` with `execution_profile=exploratory`. It is
  neither ADR-0025 qualification nor a supported soft fixed point, BR, regret,
  Nash, or MFG result.
- Panel floor 4 was a runtime bound, not a statistical claim. One last-round
  active bin had only six panels, so rare-bin Q estimates remain noisy.
- Latency, replay, and wasted work rose slightly while residuals declined. This
  is evidence that the current zero-price private response can move toward a
  self-consistent policy without improving the system objective.
- The cancelled formal run wrote no artifact. Its exact consumed-call count is
  unavailable because the old path only committed accounting at normal exit.

#### Next

Do not restart the 4,096-by-20 formal run yet. Use this bounded evidence to
design and test the missing price feedback, then compare the price-mediated
response against NIIN under the same identity library.

## Answer

The bounded exploratory run completed within budget and showed a clear
downward residual trend, but not convergence. It also showed slightly worse
system metrics, so a long zero-price formal rerun is not justified before the
price-feedback layer is introduced.

## Comments

- 2026-09-08 correction: ticket 20 proved that this artifact started from
  NIII rather than the declared NIIN fallback. Keep it only as historical bug
  evidence; its policy path and performance trend are not a valid NIIN-based
  comparison and must not support a price or MFG conclusion.
