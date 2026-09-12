# Design the shared-Backup Expert game

Type: mathematical design
Status: resolved
Blocked by: none

## Scope

Design the nondegenerate finite Expert game and its proposed common-noise
mean-field limit. Read ADR-0014 as the implemented baseline. Produce a new
feature spec and Proposed ADR without changing production code, historical
configs, results, or activating experiments.

## Acceptance criteria

1. Specify a dimensionally consistent shared scheduler, finite state and
   observations, lifecycle/event rules, four actions, and admission ledger.
2. Specify individual and social objectives, price semantics, common-noise
   information, population scaling and the precise equilibrium claim boundary.
3. Define a bounded first solver and separate fixed-point, unilateral deviation,
   finite-size and decision-value validation with independent data.
4. Include hand-computable cross-Expert coupling and conservation fixtures,
   implementation slices, and explicit supersession boundaries.
5. Verify required design sections, referenced local paths and analytic fixtures
   with a standard-library Python check; make no unexecuted test claims.

## Progress log

- Claimed after reading project instructions, CONTEXT, ADR-0004/0005/0009,
  ADR-0011/0012/0013/0014 and the attribution/scaled feature specifications.
- Inspected the working tree. Existing unrelated changes and all historical
  run families are preserved. Current work is documentation only.

### Update: 2026-09-05 — Shared-Backup Expert game designed

Status: completed

#### Goal

Define real cross-Expert resource coupling, individual incentives and a
checkable route from the finite queueing game to a conditional MFG candidate.

#### Changed

- Added the full feature specification: equal sharing across B/C active heads,
  N*c_B capacity, exact work accounting, online observations, four actions,
  independent local reservations, and global event rescheduling.
- Defined scheduled/random common-fault scenarios, causal conditional laws,
  nested population scaling, own-Expert CVaR objectives, social welfare and an
  optional occupancy tariff distinct from a shadow price.
- Defined a bounded256-rule pure-response search, nonconvergence outcomes,
  finite unilateral-deviation reruns, conditional size validation, statistical
  qualification,15 physical/information fixtures and implementation slices.
- Added Proposed ADR-0015 and a standard-library design arithmetic checker.
  Production source, historical configuration and experiment artifacts were
  not edited. No new scientific experiment was executed.

#### Verification

- `.venv/Scripts/python.exe .scratch/shared-backup-game/verify_design.py`:
  passed required-section/reference checks and exact rational arithmetic for
  interference, acceleration, dual sharing, capacity, budgets and256 rules.
  These are analytic design checks, not validation of an implemented scheduler.
- `.venv/Scripts/python.exe -m unittest discover -s ./tests -v`:
  385 tests passed in19.501 seconds.
- `.venv/Scripts/python.exe -m mfg_hedge check --config ./configs/v1_minimal.json`:
  status ok; retained existing single-domain failure headroom finding
  (base load1.400 exceeds target.900).

#### Artifacts

- `.scratch/shared-backup-game/spec.md`
- `docs/adr/0015-shared-backup-expert-game.md`
- `.scratch/shared-backup-game/verify_design.py`
- This design ticket.

#### Decisions and risks

ADR-0015 is Proposed; design completion does not activate implementation.
Shared throughput is an explicit new physical assumption. Default c_B=.5,
new360 cutoff, random duration laws and local destination marks require the
new run family. First results can certify only a restricted strategy game;
pure equilibrium existence and MFG approximation are unproved. Complementarity
is inapplicable to a mere occupancy tariff. Finite deviations must recompute
all shared dynamics. Historical paused solver tickets remain unchanged.

#### Next

Implement the bounded shared-capacity scheduler slice under a separate
disposition, beginning with cross-Expert interference and conservation fixtures.

## Answer

The requested game layer has a complete proposed mathematical and engineering
contract. The design ticket is resolved; implementation and equilibrium
validation remain future work, not claimed completed by these checks.
