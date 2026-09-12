# Random common noise, nested workloads and causal observations

Type: implementation
Status: resolved
Blocked by: 02

## Scope

Implement only the second Shared-Backup Expert game slice:

- deterministic SHA-256 keyed stochastic common fault paths;
- independent nested per-Expert workload traces with stable destinations and
  attempt-0/1/2/3 streams;
- causal, immutable public-fault observations with no future leakage;
- an online N/D/S/X fixed-rule interface and strict 256-rule enumeration;
- construction and audit of multiple idiosyncratic populations under one
  realized common path;
- spec fixtures 8, 11, 13, and 14 information, trace, and rule-contract
  portions only.

Preserve Stage 1 scheduler semantics and every historical engine, trace,
configuration, artifact, CLI, solver, and scientific result. This ticket does
not implement scoring, CVaR/social objectives, deviations, tariffs, prices,
equilibrium search, conditional particle models, campaigns, qualification,
holdout, or experiment artifacts.

## Acceptance criteria

1. Scheduled and stochastic common-fault paths have validated half-open H/D/F/R
   boundaries; stochastic durations use independent stable SHA-256 child keys,
   exclude Expert/population IDs, and are identical across N and populations.
2. Each Expert has an independent Poisson(.45) nested stream with independent
   class, destination, and all four attempt work streams. First eight local
   streams are byte-identical when nested from N=8 to N=16; global IDs are
   assigned only after merge and never drive randomness or routing.
3. Population identity and fingerprints audit the common path separately from
   local idiosyncrasy, without scientific timestamps.
4. ActionSource receives only an immutable causal public-fault observation and
   current local/public physical fields; sampled durations, future boundaries,
   future arrivals, other queues, and true work requirements are absent.
5. The online four-action rule interface strictly validates and enumerates all
   4^4=256 lexical N/D/S/X rules, with the specified D/A and age-position
   semantics and Stage 1 budget projection.
6. Focused tests prove future-prefix invariance, malicious-observer field
   absence, common-path sharing with independent populations, nested scaling,
   exact trace determinism, full drain at cutoff 360, and unchanged Stage 1 /
   historical behavior.

## Progress log

### Update: 2026-09-05 — Claimed random-noise and causal-observation slice

Status: partial

#### Goal

Claim the sole Stage 2 implementation ticket and establish its bounded
red-green-refactor scope before changing implementation code.

#### Changed

- Created and claimed this ticket as the only active ticket for the random
  common-noise, nested-workload, and causal-observation slice.
- Retained the Stage 1 fixture ownership correction in spec section 11: this
  slice owns fixture 8 and the information/trace/rule portions of fixtures 11,
  13, and 14.

#### Verification

- Re-read `AGENTS.md`, `CONTEXT.md`, issue tracker/update format, ADR-0005,
  ADR-0012, ADR-0014, ADR-0015, the feature spec, ticket 02, Stage 1 source
  and tests, and current workload/domain/common-state/scaled sources before
  code changes.
- No implementation or test command claimed in this preparatory update.

#### Artifacts

- This ticket only; no generated artifact or experiment output.

#### Decisions and risks

The new API will remain isolated in `game_workload.py` with only minimal
observation integration into `shared_backup.py`. Common-fault keys will exclude
population and Expert identity; all idiosyncratic streams will include the
population and local identity. No future fault path will be serialized into an
ActionSource observation.

#### Next

Add the required focused failing tests before implementing the Stage 2 API.

### Update: 2026-09-05 — Random common noise and causal nested traces implemented

Status: completed

#### Goal

Implement only the Stage 2 random common-fault, nested workload, causal
observation, online-rule, and multiple-population construction contracts while
leaving Stage 1 physics and all historical engines unchanged.

#### Changed

- Added isolated `src/mfg_hedge/game_workload.py` with frozen, strictly
  validated `CommonFaultIdentity`/`CommonFaultPath`, scheduled and stochastic
  H/D/F/R paths, independent SHA-256 child streams, and public observations
  containing only history through the current time.
- Added independent per-Expert Poisson(.45) nested streams with independent
  class, destination, and attempt 0/1/2/3 lognormal draws. Local keys include
  namespace, macro seed, common episode, population, Expert, local Token,
  Replica, and attempt; merged global IDs are assigned only after sorting and
  are not used for randomness or destination marks.
- Added explicit merge Expert IDs, population/common/local fingerprints, an
  immutable audit identity mapping, and a lazy adapter to the existing isolated
  `SharedBackupTrace` contract.
- Added the online `ActionRule` interface and stable lexical 256-rule
  enumeration. It returns N outside D Primary-A and selects by early/late D x
  Regular/Urgent; Stage 1 remains the owner of budget projection and copy
  physics.
- Added `PublicFaultObservation` integration to `ActionObservation` with
  observed phase start/transitions and `k_N` aliases, without exposing the
  full timeline, future boundaries, future arrivals, other queues, or true
  service/remaining work.
- Added 15 focused tests in `tests/test_game_workload.py`, including malicious
  observation probing, future-prefix/timer invariance, N=8/N=16 nested traces,
  multiple populations on one common path, rule enumeration, cutoff drain and
  Stage 1 adapter compatibility.
- Exported the new isolated API and changed the package version from 0.19.0
  to 0.20.0 because this is a new public namespace; no historical engine,
  configuration, artifact or scientific result was changed.

#### Verification

- Real Red: before implementation,
  `.venv/Scripts/python.exe -m unittest tests.test_game_workload -v` failed at
  collection with `ModuleNotFoundError: No module named
  'mfg_hedge.game_workload'`.
- Focused green:
  `.venv/Scripts/python.exe -m unittest tests.test_game_workload -v` — 15
  tests passed.
- Historical Stage 1 focused regression:
  `.venv/Scripts/python.exe -m unittest tests.test_shared_backup -v` — 14
  tests passed, including the Stage 1 shared-capacity, lifecycle, budget,
  batch, and conservation fixtures.
- Full regression:
  `.venv/Scripts/python.exe -m unittest discover -s tests -v` — 414 tests
  passed.
- Required config gate:
  `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` — exit 0, `status: ok`; the pre-existing
  single-domain failure-load diagnostic remains `1.400 > 0.900`.
- Syntax check:
  `.venv/Scripts/python.exe -m py_compile
  src/mfg_hedge/game_workload.py src/mfg_hedge/shared_backup.py
  tests/test_game_workload.py` — passed.
- Fault probes: for namespace `stage2:report`, macro seed `20260905`, common
  episode `4`, the realized durations were degraded
  `99.36353723678042` and failed `16.108560115953747`, with boundaries
  `100.0`, `199.36353723678042`, and `215.47209735273415`. At ages
  `49.999` and `50.0`, observations were respectively early-D and late-D;
  only the D transition was visible before failure.
- Future-prefix evidence: two traces with identical history through both
  arrivals had identical observations, actions, balances, and actual delayed
  timer outcomes; duration edits after the prefix did not change prior data.
  The malicious source saw none of `failed_start`, `recovered_start`,
  `timeline`, `trace`, service requirement, remaining work, future event, or
  future-arrival fields.
- Nested evidence: the same `p0` common episode generated 1,316 merged Tokens
  at N=8 and 2,599 at N=16; all first eight local Expert traces and all their
  class/destination/attempt draws were exactly equal, while merged global IDs
  were independently assigned. Rebuilding the same identity was equal and
  byte-stable by the canonical fingerprints.
- Population evidence: `p0` and `p1` shared the same common fault fingerprint
  and had different local fingerprints. The auditable identity includes
  namespace, macro seed, common episode, population, Expert count, both
  fingerprints, key-schema version, and simulator version, with no scientific
  timestamp.
- Rule evidence: 256 unique rules in lexical N<D<S<X> order, beginning NNNN,
  containing NSSN at lexical index 40, with online D/A and age-position
  semantics. D/S charge 1 and X charge 2; Stage 1 whole-action suppression
  remains N with no X-to-S downgrade.
- Cutoff/drain evidence for the N=8 `p0` stochastic population: last arrival
  `358.94019281697257`, drain end `360.9850143941203`, 1,316/1,316 Tokens
  complete, 30 Replays, total executed work `1332.2624955201597`, and
  non-winner work `6.258628295423421`. Every checked shared interval stayed
  within its capacity bound and every checked queue audit conserved work.
- Stage 1 shared-capacity evidence remains the ticket-02 hand trace: the
  mixed fixture's busy B/C work integral is `7.0` against capacity integral
  `7.0`, total executed work is `13.5`, waste is `8.5`, and its maximum
  conservation residual is `0.0`.

#### Artifacts

- Source: `src/mfg_hedge/game_workload.py`
- Minimal observation integration: `src/mfg_hedge/shared_backup.py`
- Focused tests: `tests/test_game_workload.py`
- Public exports/version: `src/mfg_hedge/__init__.py`, `pyproject.toml`
- No experiment, formal config, CLI output, qualification/holdout output,
  dataset, report, or generated artifact was created.

#### Decisions and risks

The common path is deliberately constructed once and reused by population
builders, while all local streams include population identity. The public
observation is a projection, not a view over the source path, so future fault
durations cannot be read by policy code. This slice does not implement a
conditional-law estimator, scorer, unilateral deviation, best response,
solver, tariff/price, MFG/Nash search, campaign, experiment, qualification or
holdout. Random service values are unbounded lognormal draws but are checked
finite and positive before entering the trace contract. No ADR mathematical
semantics were changed.

#### Next

None; the scoped Stage 2 implementation and required verification are
complete. Later scorer/deviation/solver slices require separate tickets.

## Answer

The Stage 2 random common-noise, nested-workload, and causal-observation slice
is complete and resolved. Its ticket scope is satisfied; later
scoring, deviation, solver/MFG, campaign, qualification, holdout and artifact
work remains unimplemented.
