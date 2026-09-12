Type: execution

Status: resolved

Blocked by: 17

## Goal

Run the accepted ADR-0025 T4c diagnostic on 4,096 identities for at most 20
rounds, without changing code, the physical engine, or the T4c statistical
protocol.

## Frozen execution contract

- Namespace: `token-mfg-restoration:t4c:soft-fixed-point:v1:library`.
- Macro seed: `20260915`.
- Episode identities: `4,096`; maximum rounds: `20`.
- Maximum scheduler calls: `409,600`.
- Fixed `beta=1.0`, `eta=0.2`, T4 physical protocol, nine retained bins,
  N/D/I, complete drain, and identity-only trace reconstruction.
- Every round must retain active/inactive/unresolved bins, occupancy mass,
  weighted policy residual, maximum active-bin residual, N/D/I probability
  changes, panel counts, scheduler calls, latency, Replay, Hedge, work, and
  Protection-Storm metrics.
- Allowed terminal statuses only:
  `supported_soft_fixed_point`, `statistics_insufficient`,
  `not_converged_20`, and `physical_failed`.
- The output must be a fresh non-overwriting artifact.  No Nash, MFG, regret,
  price-feedback, or predictor claim is allowed.

## Blocking preflight

The current T4c runner cannot yet satisfy this execution contract without a
production-code change:

1. It emits `soft_fixed_point`, not the frozen
   `supported_soft_fixed_point` status.
2. Its round rows do not contain latency, Replay, Hedge, work, or
   Protection-Storm metrics.
3. It does not normalize simulation/invariant exceptions to the required
   `physical_failed` terminal status.

Because the user explicitly froze this round as execution-only and prohibited
further code changes, the formal run must not start until that contract gap
is separately authorized and fixed.

## Progress log

### Update: 2026-09-08 — Formal T4c execution preflight blocked

Status: blocked

#### Goal

Accept ADR-0025 and preflight the formal 4,096×20 run without changing code
or launching a non-conforming statistical diagnostic.

#### Changed

- Accepted ADR-0025 as confirmed by the user on 2026-09-08.
- Created this execution ticket; ticket 15 remains resolved.
- No production code, test code, or artifact was changed or generated.

#### Verification

- Read-only scan of `token_mfg_t4c_supported_soft_response.py` confirmed the
  missing formal status, per-round metric fields, and `physical_failed`
  normalization listed above.
- Confirmed the requested fresh run path
  `artifacts/token-t4c-supported-soft-response-20260908-r1` does not exist.
- Formal run was not started; therefore no scheduler calls were consumed.

#### Artifacts

None.

#### Decisions and risks

- Starting now would produce an artifact that cannot be interpreted under the
  accepted ADR-0025 contract.
- This is an execution-contract blocker, not evidence that Soft-BR failed or
  that the population iteration did not converge.

#### Next

Authorize a narrow implementation correction for the three output-contract
gaps, then rerun the preflight before starting the long diagnostic.

### Update: 2026-09-08 — T4c execution gate cleared

Status: claimed

#### Goal

Proceed with the accepted formal 4,096 x 20 T4c diagnostic after the narrow
execution-contract correction passed verification.

#### Changed

- Ticket 17 resolved after implementing only the three authorized output and
  fail-closed corrections.
- Removed the blocker and claimed this ticket as the sole active ticket.
- No campaign parameters, sample sizes, mathematical definitions, or ADRs
  were changed.

#### Verification

- Focused T4c suite: `12/12` passed.
- Full suite: `597/597` passed.
- Config check: `status: ok`.
- Fresh formal artifact directory was checked before launch and did not exist.

#### Artifacts

None before the formal run.

#### Decisions and risks

- The only permitted terminal statuses remain
  `supported_soft_fixed_point`, `statistics_insufficient`,
  `not_converged_20`, and `physical_failed`.
- The run remains a zero-price finite Token supported-soft fixed-point
  diagnostic and cannot claim BR, regret, Nash, or MFG.

#### Next

Launch the formal diagnostic and append its terminal result and fresh artifact
path after completion.

### Update: 2026-09-08 — Formal run cancelled by the user

Status: completed

#### Goal

Stop the unexpectedly long 4,096 by 20 run and preserve the absence of a
partial formal result.

#### Changed

- The two confirmed T4c Python processes were terminated at the user's request.
- No source, protocol, or historical artifact was changed by the cancellation.

#### Verification

- Process inspection identified the active T4c parent/child processes; both
  were stopped successfully.
- No `*t4c*` final artifact or staging directory exists under `artifacts/`.
- A 16-identity one-round timing probe used 68 calls in 4.141 seconds, or about
  0.0609 seconds per call on the current host.

#### Artifacts

None. The interrupted runner publishes only at terminal completion, so its
exact consumed call count is unavailable and no partial result is claimed.

#### Decisions and risks

- The accepted 4,096 by 20 protocol remains scientifically unchanged but was
  not completed.
- Its host estimate is roughly seven hours at the measured small-probe rate;
  the earlier 34-hour estimate was overly pessimistic, but still correctly
  indicated an unsuitable interactive run.

#### Next

Run a separately labelled bounded exploratory diagnostic before committing to
another formal campaign.

## Answer

The formal T4c execution was cancelled by explicit user instruction without a
published or partial artifact.

## Comments

- 2026-09-08 correction: ticket 20 found and repaired the T4c NIII-versus-NIIN
  fallback mismatch. The cancelled formal run produced no artifact and must
  not be resumed from its former process state.
