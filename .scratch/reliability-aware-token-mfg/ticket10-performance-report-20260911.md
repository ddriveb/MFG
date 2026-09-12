# Ticket 10 performance and restart-safety report

Date: 2026-09-11

## Outcome

The real K-scale scheduler now passes both frozen timing gates without changing
the qualification models, samples, namespaces, CRN identities, physical event
semantics, or call budgets.

- Worst measured K=64 scheduler call: `0.4881354 s` (gate: `<= 30 s`).
- Eight-worker K=64 throughput: `0.33936165 calls/s`.
- Frozen 51,072-call qualification projection: `25,611.31 s`, or `7.11 h`
  (gate: `<= 12 h`).
- Isolated preflight calls: `24/32`; formal calls consumed: `0`.
- Parent peak RSS: `27,922,432 bytes`; every worker reported supported RSS.
- Compact worker identity serialization: `155 bytes` per task.

The former K=64 measurement was approximately `137.907 s` per call.  The
optimized scheduler call is therefore roughly 282 times faster on the measured
probe.  Trace construction remains outside scheduler timing and is performed
inside workers so full immutable traces are not transferred through the parent.

## Root cause and correction

The dominant defect was `RoutingTrace.work_for`: each service lookup scanned the
entire work-draw collection.  At K=64 this nested scan dominated the event loop.
The trace now constructs immutable dense lookup offsets once and performs the
canonical lookup in O(1), with a compatibility fallback for legacy noncanonical
fixtures.  Component fault events are indexed once, and active-token snapshots
iterate only currently active token identities instead of all future arrivals.

The optimized and reference paths were compared exactly at K=8, 16, 32, and 64.
The complete `SimultaneousRoutingResult`, trace fingerprints, decisions,
completion records, work accounting, and invariant counters were equal.

## Restart safety

`qualification_checkpoint.py` adds a parent-owned atomic journal:

- plan, source, and per-work input fingerprints are mandatory;
- a unit is durably reserved before dispatch and atomically committed afterward;
- committed outputs can be reused after restart without another scheduler call;
- an unresolved prior dispatch fails closed and is never silently retried;
- completed and reserved call counts are persisted independently;
- changed source, plan, input, or call reservation is rejected.

The qualification source fingerprint covers the routing physics, simultaneous
engine, population estimator, response/deviation layers, topology, campaign
contract, and checkpoint implementation.

## Remaining execution boundary

Formal qualification was not started.  `run_qualification_campaign` is still a
fail-closed entry point: ticket 06 must connect its real forward, continuation,
finite-K deviation, final-confirmation, and holdout work units to the checkpoint
journal and deterministic worker merge before the first formal call.  The timing
gate is passed, but starting before that wiring exists would reintroduce the
partial-run restart risk this ticket is intended to remove.

## Verification

- `.venv\\Scripts\\python.exe -m unittest tests.test_token_mfg_qualification_safety tests.test_token_mfg_response_solver tests.test_token_mfg_qualification_campaign -v` — 19 tests passed.
- affected routing/population/MFG suites — 70 tests passed.
- `.venv\\Scripts\\python.exe -m unittest discover -s tests -v` — 795 tests passed in 275.929 s.
- `.venv\\Scripts\\python.exe -m mfg_hedge check --config configs\\v1_minimal.json` — exit 0, `status: ok`.
- `git diff --check` — exit 0; warnings concerned unrelated parent-worktree line endings only.

No formal qualification, holdout, candidate selection, Nash claim, MFG claim,
or experiment artifact was produced.
