# Implement deterministic replica-routing baseline kernels

Type: implementation
Status: resolved
Blocked by: none

## Scope

Implement the first slice in `../spec.md`: placement, routing, and hedging
kernels plus provenance and focused tests. Preserve all existing engines and
artifacts. Use only the Python standard library.

## Acceptance criteria

1. Every public baseline reports a stable ID, semantic class, source URL,
   source revision, and adaptation kind.
2. EPLB-style replication/packing matches hand-computed upstream examples for
   load splitting, equal-cardinality packing, and stable ties.
3. RR, JSQ, and least-work routing are causal and deterministic.
4. LPLB-style batch routing returns an eligible, integral assignment that
   minimizes maximum token count for the tested topologies and uses stable
   tie-breaking.
5. Failover-only launches no speculative copy; P95 launches only after its
   frozen threshold; LÆDGE never idles a replica while unserved work exists and
   hedges only when no unserved request is waiting.
6. Existing public APIs and scientific behavior remain unchanged.
7. Focused tests, the full unittest suite, and config check pass.

## Progress log

### Update: 2026-09-08 — Implement deterministic baseline kernels

Status: completed

#### Goal

Integrate executable, provenance-labeled placement, routing, and hedging
baseline kernels without modifying the existing event simulators or artifacts.

#### Changed

- Added `src/mfg_hedge/replica_routing_baselines.py` with EPLB-style
  hierarchical/global placement, RR/JSQ/least-work online routing, exact
  integral LPLB token-count min-max assignment, failover-only, frozen-P95
  delayed hedging, and generalized LÆDGE scheduling.
- Added `tests/test_replica_routing_baselines.py` with 15 focused tests,
  including DeepSeek EPLB's published two-layer example and brute-force LPLB
  optimum checks.
- Added `THIRD_PARTY_NOTICES.md`, the feature spec, and ADR-0030. Public APIs
  are exported from `mfg_hedge`; package version changed 0.25.0 to 0.26.0.
- No third-party dependency was added. The attempted shallow Git clones failed
  before checkout because GitHub transport was unavailable; implementation and
  license review used the public GitHub/USENIX web sources named in the notice.

#### Verification

- Red: `.venv/Scripts/python.exe -m unittest tests.test_replica_routing_baselines -v`
  failed with `ModuleNotFoundError: mfg_hedge.replica_routing_baselines` before
  implementation.
- Focused final: the same command ran 15 tests, all `OK`.
- Full: `.venv/Scripts/python.exe -m unittest discover -s tests -v` ran 639
  tests in 158.895 seconds, all `OK`.
- Config: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 and `status: ok`, retaining the
  existing single-domain failure-load audit finding.
- Compile: `.venv/Scripts/python.exe -m compileall -q` for the new source and
  test files returned exit 0.
- Import smoke reported version 0.26.0 and all eight catalog IDs.

#### Artifacts

- `src/mfg_hedge/replica_routing_baselines.py`
- `tests/test_replica_routing_baselines.py`
- `THIRD_PARTY_NOTICES.md`
- `docs/adr/0030-replica-routing-baseline-taxonomy.md`
- `.scratch/replica-routing-baselines/spec.md`

No experiment artifact was generated or overwritten.

#### Decisions and risks

- The baseline taxonomy prevents placement, routing, and hedging results from
  being mislabeled as direct substitutes.
- EPLB matches the inspected algorithm and README example. LPLB is an exact
  integer solver for its token-count min-max objective, not a claim of CUDA/IPM
  runtime equivalence. LÆDGE uses deterministic replica-ID ties instead of the
  paper's random replica choice.
- These are runnable kernels, not yet a common-physics comparison. A shared
  simulation adapter is required before reporting performance rankings.

#### Next

Implement one common replicated-Expert evaluation adapter that runs RR, JSQ,
least-work, LPLB, failover-only, P95, LÆDGE, NIIN, and the proposed controller
on identical Gate output, placement, CRN workload/fault traces, and metrics.

## Answer

All acceptance criteria for the baseline-kernel slice are satisfied. The
ticket is resolved; no experiment was run.

## Comments

- This ticket integrates executable algorithm kernels. It intentionally does
  not produce a cross-baseline experiment; that requires a separate common
  simulator/metrics adapter.
