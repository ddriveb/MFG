# Implement fail-closed two-panel comparison adapters

Type: implementation
Status: resolved
Blocked by: 01

## Scope

Implement the second slice in `../spec.md`: a stable registry, backend
capability contract, panel validation, routing/hedging kernel construction,
and a common metric projection. Do not run the formal comparison and do not
invent missing batch/GPU/LÆDGE physics.

## Acceptance criteria

1. The protection and routing panels contain exactly the frozen comparator
   sets and one shared placement/trace/metrics identity per panel.
2. EPLB is recorded as routing-panel placement and cannot appear as an arm.
3. Every arm declares required backend capabilities; incompatible execution
   fails before a scheduler call.
4. Online router adapters invoke RR/JSQ/least-work; the batch adapter invokes
   LPLB; hedging adapters construct Failover/P95/LÆDGE kernels.
5. Existing NIIN and requested-price candidates are represented by stable
   policy IDs without importing or executing sealed artifacts at plan time.
6. Common metric projection accepts both mapping and dataclass records,
   requires the headline latency/replay/work/storm fields, and rejects partial
   results.
7. Focused, full-suite, and config checks pass; no artifact is written.

## Progress log

### Update: 2026-09-08 — Implement fail-closed comparison adapters

Status: completed

#### Goal

Bind the previously implemented baseline kernels to two scientifically valid
comparison panels, validate backend capabilities before execution, and expose
one strict common metric projection without pretending that missing batch,
GPU-capacity, or idle-release physics already exists.

#### Changed

- Added `src/mfg_hedge/baseline_adapters.py` with stable arm IDs, two frozen
  panel definitions, backend capability declarations, fail-closed validation,
  routing/batch/kernel constructors, and a common headline metric projection.
- The protection panel contains Failover-only, P95 delayed hedging, LÆDGE,
  exact NIIN, and the sealed requested-price candidate. The routing panel
  contains RR, JSQ, least unfinished work, and LPLB under one EPLB placement.
- Existing NIIN and requested-price policies are represented by references at
  plan construction time; the adapter does not open sealed artifacts.
- Added `tests/test_baseline_adapters.py` with 10 focused tests. Exported the
  adapter API from `mfg_hedge`; package version changed 0.26.0 to 0.27.0.
- Updated the feature spec with the adapter slice. No simulator physics,
  experiment configuration, existing artifact, or third-party dependency was
  changed.

#### Verification

- Red: `.venv/Scripts/python.exe -m unittest tests.test_baseline_adapters -v`
  failed with `ModuleNotFoundError: mfg_hedge.baseline_adapters` before the
  implementation existed.
- Focused final: the same command ran 10 tests, all `OK`; combined baseline and
  adapter suites ran 25 tests, all `OK`.
- Full: `.venv/Scripts/python.exe -m unittest discover -s tests -v` ran 649
  tests in 211.192 seconds, all `OK`.
- Config: `.venv/Scripts/python.exe -m mfg_hedge check --config
  configs/v1_minimal.json` returned exit 0 and `status: ok`, retaining the
  existing single-domain failure-load audit finding.
- Compile and import smoke returned exit 0, reported version 0.27.0, and built
  exactly `protection_same_topology` and `routing_after_placement`.

#### Artifacts

- `src/mfg_hedge/baseline_adapters.py`
- `tests/test_baseline_adapters.py`
- `.scratch/replica-routing-baselines/spec.md`

No experiment artifact was generated or overwritten.

#### Decisions and risks

- EPLB is a placement shared by all routing arms and is never reported as a
  measured routing arm.
- The existing two-Replica engine intentionally fails capability validation
  for LÆDGE and LPLB: it lacks idle-release and batch-release/GPU-capacity
  semantics. This prevents an invalid flat comparison.
- The adapters establish executable contracts, not a completed performance
  experiment. The two missing common-physics evaluation backends remain the
  material next implementation work.

#### Next

Implement the protection-panel execution backend first, adding an explicit
idle-release callback so Failover-only, P95, LÆDGE, NIIN, and the requested-
price policy run on identical episode traces and metric definitions.

## Answer

All acceptance criteria for the comparison-adapter slice are satisfied. The
ticket is resolved; no comparison experiment was run.

## Comments

- The executable experiment backend remains a separate ticket because LPLB
  requires batch release semantics and EPLB performance requires a GPU/shared
  resource model that the current two-Replica episode does not contain.
