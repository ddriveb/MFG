# Implement bounded transient admission and outcome objective

Type: implementation
Status: resolved
Blocked by: none

## Scope

ADR-0012; transient-model-design.md sections 4-5 only. Build an isolated
metadata-only time/class rule, causal reservation, applied-action episode runner,
and pooled outcome objective. No enumeration, fitting, MFG, CLI, config or artifacts.
Preserve production engine, old quota, old objectives and historical schemas.

## Acceptance criteria

1. Half-open eligibility/window boundaries, cap 2.8125 admits two mean-one
   requests; no refund, final short window, pooling, no future draw inputs.
2. Deterministic prefix decisions under altered future metadata or service draws;
   fresh ledger each episode; requested = applied + quota_suppressed for Hedge
   counts globally/per-window/per-class; engine requests equal applied Hedge.
3. Suppression runs Normal physics including Replay; all-Normal exact regression.
4. Fractional CVaR matches variational hand cases at n=1/20/21, differs from legacy.
5. Hand-computable full score, class weights, SLO excess, Replay, work/waste;
   pooled counts not mean episode ratios; missing cohort total=null; bad/duplicate
   or mismatched results rejected; all executions including drain counted.
6. Real development-only bounded smoke without policy selection; deterministic
   repeats and trace fingerprints. No artifact write or holdout generation.
7. Focused Red then Green, full unittest suite and environment check.

## Progress log

### Update: 2026-09-05 — Implement transient admission and realized objective

Status: completed

#### Goal

Deliver the authorized admission/scoring slice after ticket 09 physical validation.
No old solver implementation, rule enumeration, qualification or holdout run.

#### Changed

- New Accepted ADR-0012 activates only reservation and outcome scoring; whole
  ADR-0011 remains Proposed. Spec contains a bounded activation pointer.
- `src/mfg_hedge/transient_control.py`: immutable fixed time/class rule and
  decisions; metadata-only causal prepass; pooled mean-work reservation; strict
  half-open boundaries, Decimal debit accounting, no refunds, partial-window
  capacity; requested/applied/suppressed audits at global/window/class levels;
  applied-action wrapper around the unmodified physical episode engine.
- `src/mfg_hedge/transient_objective.py`: invariant-checked pooled outcome score;
  explicit fractional-tail CVaR95, class/phase sample counts and cost components;
  total/wasted executed work through drain; missing cohort => incomplete/null;
  duplicate/mismatched episode rejection. Historical metric builders unchanged.
- `tests/test_transient_control.py`: 19 new tests covering the acceptance contract.
- Public exports and package version 0.15.0 -> 0.16.0 for new APIs. Old science
  unchanged; future historical summaries may differ only in version provenance.
- `docs/transient-control.md`: executable usage and limitations; README entry
  and removal of the stale statement that no solver has been implemented.

#### Verification

- Real Red, before implementing either new module:
  `.\.venv\Scripts\python.exe -m unittest tests.test_transient_control -v`
  -> `Ran 1 test ... FAILED (errors=1)`, ModuleNotFoundError for transient_control.
  This is the expected missing implementation, not a claim of an old engine bug.
- Focused final Green, same command: `Ran 19 tests in 0.134s ... OK`.
- Full required regression:
  `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`
  -> `Ran 366 tests in 23.534s ... OK` (347 existing + 19 new).
- Environment:
  `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`
  -> exit 0, status ok, Python 3.10.11; existing single-domain headroom warning.
- Hand score: phase loss 8.4 + D/F tail 6 + work 4 + waste 0 = J 18.4.
- Suppressed I on identical trace: accepted completion 19.5/no Replay vs
  denied completion 23/one Replay. Denied full result equals Normal baseline.
- Running loser fixture: A wins at 199, B runs through 398; executed work
  200.5 and wasted work 200, including post-cutoff service.
- New fractional CVaR n=21: 100.5/1.05; old empirical CVaR stays 55.
- In-memory documented-API smoke executed through project Python stdin, 4
  episodes, macro 10, namespace transient-control:v1:fit, load .45, canonical
  100/200/220/320 protocol, compute_tau0=1.6385447242959017, fixed D/I/D/I rule:
  1133 Tokens, requested 169, applied 32, suppressed 137, reserved work 32,
  engine requests 32. Score complete, J=16.802814750614324, components
  phase=3.6087459120868064, tail=12.126005420705066,
  work=1.046862747225536, waste=0.02120067059691549.
  Two runs exactly equal; sorted-key score JSON byte-equal. No policy comparison,
  selection, efficacy test, or significance claim was made.

#### Artifacts

Only source/tests/docs/ticket changes listed above. No experiment output written;
no historical artifacts, configurations, old engine/solver/quota code modified.

#### Decisions and risks

- Metadata prepass is causal only for this restricted fixed-rule/ledger family;
  it is not queue-feedback control. Fixed TimeClassRule accepts no callback.
- Pooling can deny later Urgent requests; counts expose this. Empty D windows
  omitted with explicit audit coverage label. Defaults admit two mean-one
  requests per full window, preserving unused fractional capacity rather than
  silently increasing the cap. Requested=applied+quota suppression does not
  conflate executor suppression or launch count.
- Reservation is not realized work: fixture admits work 4 under cap 2.8125.
  No expectation bound is asserted for arbitrary handcrafted input traces.
- New score is a finite-data engineering objective, not a capacity certificate,
  safety gate, economic optimum, equilibrium or MFG attribution result.
- Legacy empirical CVaR and all historical schemas stay unchanged. Full
  ADR-0011 adoption and old blocked ticket 03 are not advanced by this slice.

#### Next

Design/implement a bounded development-only complete 81-rule enumeration with
No-Hedge reference and the predeclared safety filters; no holdout or MFG claim.

## Answer

All seven acceptance criteria passed. The bounded applied-action-to-outcome
pipeline is ready as a regression-tested basis for later policy comparisons.
Ticket 10 is resolved; no adjacent implementation ticket was claimed.

## Comments

- User's latest confirmation authorizes the stated next slice, not full ADR-0011.
- Parent workspace contains unrelated tracked/untracked changes; none are touched.
