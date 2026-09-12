# Complete the original Token-player model and audit restoration scope

Type: design and read-only engineering review
Status: resolved
Blocked by: none

## Scope

Restore the user's original research subject: Tokens choose protection under
mean-field congestion and a declared resource-price/admission mechanism;
Experts and Replicas are execution resources. Complete a reviewable mathematical
specification and map current code to a bounded migration plan. This ticket
does not change production code, launch experiments, revert historical work,
or claim ownership of the separately claimed Shared-Backup ticket 15.

## Acceptance criteria

- Define player, information, decisions, local state, finite physics, admission,
  resource units, individual and system objectives, forward law, best response,
  common noise, price interpretation, scaling and honest equilibrium labels.
- Explain where exact finite FIFO queues do not yield a closed Token marginal
  law, and make any mean-field approximation an explicit new assumption.
- Resolve ex-ante versus realized pricing, quota suppression, softmax regret,
  Token lifetime/arrival mass, and population-CVaR versus individual preference.
- Inspect actual historical Token and current Expert code entry points; identify
  reusable parts, required modifications, isolated alternatives, and missing work.
- Provide an ordered implementation plan with acceptance checks, without a
  speculative percentage-complete or unchanged historical-result reinterpretation.
- Verify references and document-only mutation boundaries; record exact checks.

## Progress log

### Update: 2026-09-06 — design/audit scope claimed

Status: partial

#### Goal

Complete the Token model and determine engineering restoration scope.

#### Changed

Created this design-only ticket. Found Shared-Backup ticket 15 separately claimed;
its files, campaign and status remain outside this ticket's ownership.

#### Verification

Read CONTEXT, workflow rules, ADR-0006/0008/0012/0015 and current issue status;
inspected the working tree, which contains unrelated parent-workspace changes.

#### Artifacts

This ticket; specification and audit to follow.

#### Decisions and risks

Token-player restoration is the user's explicit direction. New closure/scaling
choices will be identified as proposals, not silently accepted physics changes.

#### Next

Inspect both execution/solver paths and write the complete model and code map.

### Update: 2026-09-06 — Token model and engineering restoration audit completed

Status: completed

#### Goal

Complete the original Token-player model and establish the actual engineering
restoration boundary, preserving the separate Expert game and active pilot.

#### Changed

- Added `spec.md`: finite Token state/information, one-shot N/D/I actions,
  actual admission, individual payoff versus system CVaR, explicit posted
  reservation-price rule, exact finite-law boundary and candidate many-server
  marked-measure HJB/forward scaling with births/physical settlement.
- Added Proposed ADR-0020, separating the user-restored player identity from
  the newly proposed preferences, pricing and scaling dispositions.
- Added `engineering-audit.md`: inspected both old Token and new Expert paths,
  file-level reuse/change map, hidden migration risks and six implementation
  work packages with acceptance checks.
- Added the local read-only `audit_probes.py` and its evidence JSON. No source,
  configuration, production test or historical artifact was edited.

#### Verification

- `.venv\Scripts\python.exe .scratch/token-mfg-restoration/audit_probes.py`:
  exactly two hand-fixture scheduler calls. Switching only Token0 N->I changed
  its latency4->1 and Token1 latency1->1.9 on the original ONE Expert A/B
  engine. Assertions passed; source hashes unchanged during the probe.
- `.venv\Scripts\python.exe -m unittest tests.test_hedge_simulation tests.test_transient_control tests.test_game_workload tests.test_mfg_readiness_fixes -q`:
  75 tests passed in2.292s.
- `.venv\Scripts\python.exe -m mfg_hedge check --config configs/v1_minimal.json`:
  status ok, with its existing F base-load1.4 versus target.9 finding. This
  historical minimal config is not a new-model experiment preflight.
- Python documentation/reference verification: four Markdown files, thirteen
  existing source symbols, local links and fences checked; errors=[];
  source_changed_since_probe=[]. Output in `verification.json`.
- No full suite, Fit, Validation, pilot or MFG campaign was run by this ticket.
  Mathematical well-posedness and new-model empirical validity remain unproved.

#### Artifacts

- `.scratch/token-mfg-restoration/spec.md`
- `.scratch/token-mfg-restoration/engineering-audit.md`
- `.scratch/token-mfg-restoration/audit_probes.py`
- `.scratch/token-mfg-restoration/engineering-evidence.json`
- `.scratch/token-mfg-restoration/verification.json`
- `docs/adr/0020-restore-token-player-model.md`

#### Decisions and risks

The finite Token reference can be restored without deleting the Expert route.
The largest missing work is online prediction/Token deviations/forward closure,
not a Git rollback. Proposed many-server K scaling keeps Token players and
individual service times but changes physical executor count for the limit;
its applicability at K=1 is unverified. Price pacing is not market clearing;
entropy response is not exact unregularized Nash. The separately claimed
Shared-Backup ticket15 and its accepted protocol are untouched.

#### Next

Implement only T1: a causal Token observation/policy and online reservation
entry point on the original A/B engine, with static-rule trajectory equivalence.
Do not resume the old scalar-rho solver or expand Expert Pi256 as restoration.

## Answer

All design/audit acceptance criteria are met. The mathematical contract,
explicit proposed closure, source-level restoration plan and executable
externality evidence are delivered. Production implementation and formal
model/scale validation are future separately scoped work, not this ticket's
completion claim.
