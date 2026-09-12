# Review hidden defects and MFG experiment readiness

Type: code review and diagnostic analysis
Status: resolved
Blocked by: none

## Scope

Review the shared-Backup physics, causal workload/observations, player scorer,
finite solver, Fit/Validation boundaries and inference for silent scientific
errors. Use bounded synthetic/read-only reproductions, at most32 scheduler
calls; no new formal Fit/Validation or changed campaign result. Separate
reproducible defects from modeling limitations and missing MFG components.
Do not alter production behavior in this review ticket.

## Acceptance criteria

1. Trace reviewed dataflow and identify precise code locations for actionable
   findings; reproduce high-confidence defects with synthetic checks.
2. State impact on the retained r2 Fit versus unexecuted Validation/future MFG.
3. Produce a prioritized report and a bounded route to experiment readiness,
   without claiming that passing existing tests proves correctness.
4. Record executed checks and preserve historical code/config/artifact state.

## Progress log

- Claimed after checking current repository state and accepted ADR-0015/0016/
  0018. This is a review-only continuation of the user's MFG experiment work.

### Update: 2026-09-06 — Six hidden issues reproduced and MFG gaps documented

Status: completed

#### Goal

Identify implementation and statistical-contract defects that would affect
future finite-game/MFG experiments, separately from missing MFG functionality.

#### Changed

- Added a bounded synthetic probe script and prioritized readiness report.
- Reproduced Dual destination/attempt CRN mismatch; the one-sided bootstrap
  pivot-direction discrepancy under skew; stateful policy reuse across
  episodes; voided timers observed as pending; missing pseudo-values raising
  at the bound interface; and undercounted partial baseline failures.
- Documented missing source/protocol fingerprints in formal manifests.
- Distinguished existing finite-N best responses from the unimplemented
  conditional forward law, nonatomic response and finite-size validation.
- Production source, configurations, version and scientific protocol unchanged.

#### Verification

- `.venv/Scripts/python.exe .scratch/shared-backup-game/review_readiness_probes.py`
  reproduced all six issues with7 synthetic scheduler invocations, including
  one injected failure. No formal Fit/Validation calls or new random scenarios.
- Probe output verifies all source/config content hashes unchanged.
- Checked bootstrap inversion direction against the R boot source's basic.ci
  and stud.ci; the synthetic skew example demonstrates direction, not coverage.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`:
  467 tests passed in51.173 seconds despite the untested counterexamples.
- `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`:
  status ok with the existing single-domain failure headroom finding.

#### Artifacts

- `.scratch/shared-backup-game/review_readiness_probes.py`
- `.scratch/shared-backup-game/mfg-readiness-review-20260906.md`
- `artifacts/mfg-readiness-review-20260906/probes.json`

#### Decisions and risks

No fixes or new experiment activation in this review ticket. The r2 outcome
remains fit_no_candidate. Dual rules violate the declared paired-attempt
contract; NSNS/NSSS's prior variance diagnosis does not depend on Dual.
The bootstrap issue is in unexecuted Validation, and feedback-policy defects
do not affect stateless Pi256 rules. Existing successful r2 accounting is not
overturned by the injected partial-baseline failure. Review is bounded, not a
guarantee that every other defect has been found.

#### Next

Create a separately scoped repair slice for the high-priority CRN, statistical
interface/direction, and policy-lifecycle findings before designing the
conditional mean-field solver and revised precision experiment.

## Answer

Resolved. The user has a prioritized, reproducible defect report and a clear
separation between finite-game implementation repairs and work still needed
to run a defensible MFG experiment.
