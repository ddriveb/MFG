# Survey MoE replication and routing methods

Type: research
Status: resolved
Blocked by: none

## Goal

Systematically classify representative 2020--2026 MoE routing, expert
placement/replication, runtime scheduling, memory/offload, and fault-tolerance
work, with special attention to whether a method executes the same Token on
two replicas of the same Logical Expert and keeps the first completion.

## Scope

- Use primary papers, proceedings pages, and official project repositories.
- Keep model-level Top-k routing, replica load sharing, Replay, prefetching,
  and true request hedging separate.
- Produce a reusable Chinese related-work report with a normalized method
  matrix and explicitly bounded counts.
- Do not modify simulator behavior, configurations, or historical artifacts.

## Acceptance criteria

- The report states its sample boundary and counting rule.
- At least 25 representative works are classified by primary mechanism.
- Every included work has a primary-source link.
- The report explicitly answers whether expert replication normally means
  duplicate same-Token racing.
- The report maps the literature taxonomy to the repository's existing and
  proposed baselines without making an unsupported novelty claim.

## Progress log

### Update: 2026-09-09 — Survey MoE replication and routing methods

Status: completed

#### Goal

Classify representative MoE routing, replication, scheduling, memory, and
fault-tolerance methods and determine whether Expert replication normally
means same-Token request hedging.

#### Changed

- Added `docs/moe-replication-routing-method-census-zh.md`.
- Normalized 30 representative 2020--2026 works into five mutually exclusive
  primary-mechanism categories.
- Separated model Top-k activation, different-Token replica load sharing,
  failure Replay, speculative Expert prefetch, and true same-Expert
  same-Token first-completion hedging.
- Mapped the literature taxonomy to the repository's placement, routing,
  protection, and project-specific methods.
- No production code, configuration, test, or historical artifact changed.

#### Verification

- Programmatic report audit: 214 lines, 38 primary/source links, five method
  matrix sections, and an explicit zero-count same-Token Hedge row.
- `\.venv\Scripts\python.exe -m mfg_hedge check --config
  .\configs\v1_minimal.json` returned exit 0 with `status: ok`; the existing
  single-domain failure headroom finding remains unchanged.
- No simulator test run was required because this was a documentation-only
  research update.

#### Artifacts

- `docs/moe-replication-routing-method-census-zh.md`
- `.scratch/replica-routing-baselines/issues/13-survey-moe-replication-routing-methods.md`

#### Decisions and risks

- Counts describe the stated 30-work representative corpus, not the entire
  global literature.
- Exact-keyword searches found no clear MoE paper centered on same-Expert
  same-Token first-completion racing, but this is not sufficient for a
  universal priority or `first` claim.
- The current holdout supports dynamic Primary routing, not a claim that
  positive-budget hedging or MFG has already improved the system.

#### Next

Use the taxonomy to freeze a strong replicated-Expert routing benchmark before
designing another selective-Hedge or MFG experiment.

## Answer

The 30-work corpus contains 6 model-routing/load-balance methods, 7
placement/replication methods, 9 runtime scheduling/communication methods, 5
memory/offload/prefetch methods, and 3 fault-recovery methods. None uses
same-Expert same-Token first-completion hedging as its central MoE mechanism.
Expert replicas are normally used to split different Tokens, improve locality,
or recover from failures. The full method matrix and project mapping are in
`docs/moe-replication-routing-method-census-zh.md`.

## Comments
