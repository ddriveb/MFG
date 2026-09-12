# Fix the T4c NIIN fallback mapping

Type: narrow correctness fix

Status: resolved

Blocked by: none

## Goal

Make the T4c initial/unknown-bin fallback exactly equal the frozen NIIN rule
`(N,I,I,N)` without changing the event engine, payoff, Reservation, bins, or
sampling protocol.

## Acceptance criteria

1. A focused Red proves the late-D Urgent mismatch.
2. Early/late by Regular/Urgent maps exactly to N/I/I/N, while non-D and
   Backup-primary observations remain N.
3. Existing T4c focused tests, full suite, and config check pass.
4. The invalid exploratory artifact is not rewritten or reinterpreted; no new
   experiment is launched in this ticket.

## Progress log

### Update: 2026-09-08 — Repair exact NIIN fallback

Status: partial

#### Goal

Correct the concrete baseline bug found during review before any new pricing or
fixed-point experiment.

#### Changed

- Added a direct equivalence test against `token_t3a_execution.policy_factory`.
- Replaced the `urgent OR late` condition with the exact NIIN four-cell mapping.

#### Verification

- Real Red: late-D Urgent returned `I` from T4c and `N` from the frozen NIIN
  source.
- Focused T4c suite after the fix: 13/13 passed.

#### Artifacts

- `src/mfg_hedge/token_mfg_t4c_supported_soft_response.py`
- `tests/test_token_mfg_t4c_supported_soft_response.py`

#### Decisions and risks

- This ticket repairs only the exact mapping.  Bin design, target sampling,
  temperature, and pricing remain separate scientific changes.
- The ticket was materialized immediately after the urgent run cancellation
  and focused Red/Green; no experiment ran during that interval.

#### Next

Run the required full regression and config gate, then resolve this ticket.

### Update: 2026-09-08 — NIIN fallback repair verified

Status: completed

#### Goal

Finish the exact NIIN correction and isolate all invalid prior results.

#### Changed

- `T4CPopulationPolicy.base_action` now implements `N/I/I/N` exactly via the
  late-versus-Urgent four-cell mapping.
- Package version changed from `0.24.0` to `0.25.0` because the correction
  changes scientific policy behavior.
- Tickets 16 and 18 now point to this correction. The old exploratory artifact
  is retained unchanged as historical bug evidence, not a NIIN comparison.
- The interrupted price pilot, its partial source, and its tests were removed;
  ADR-0026 is Proposed/deferred and ticket 19 is blocked.

#### Verification

- Focused Red: late-D Urgent returned `I` in T4c versus `N` in the frozen NIIN
  source.
- Focused T4c suite: 13/13 passed after repair.
- Final full suite after the version change:
  `.venv/Scripts/python.exe -m unittest discover -s tests -v` ->
  `Ran 598 tests in 150.318s`, `OK`.
- `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`
  -> exit 0, `status: ok`; the existing single-domain load warning remains.
- No Python experiment process remains and no T4d artifact directory exists.

#### Artifacts

- `src/mfg_hedge/token_mfg_t4c_supported_soft_response.py`
- `tests/test_token_mfg_t4c_supported_soft_response.py`
- `pyproject.toml`
- `src/mfg_hedge/__init__.py`

#### Decisions and risks

- Existing `token-t4c-exploratory-20260908-r1` numbers are invalid as an NIIN
  baseline comparison and must not be reused for pricing or MFG claims.
- The exact bug is fixed, but the coarse aggregate-queue/phase-age bins and
  first-eligible target selector remain known modeling risks. No new experiment
  should run until that separate observation/sampling revision is frozen.

#### Next

Replace the coarse observation/first-target protocol in one bounded revision,
then run a small fixed-temperature/fixed-price grid before any dynamic price.

## Answer

The T4c initial and fallback policy now matches the frozen NIIN rule exactly,
and the invalid prior comparison has been isolated without rewriting it.
