# Qualify development and freeze the holdout registry

Type: task
Status: open
Blocked by: 05

## Goal

Use development-only data to verify feasibility and sample sufficiency, then
freeze every input required for a one-shot holdout without running it.

## Scope

- Use burned historical/development namespaces only for end-to-end qualification.
- Verify the load-0.45 nominal A reservation, planned D-cohort B envelope,
  2,000-probe calibration precision, 40 x 50 sample availability, runtime, and
  artifact completeness.
- Freeze/hash code, spec/ADR, schema-5 config, calibration table, analysis,
  policies, quota scales/main point, gates, each cost diagnostic's scheduled or
  `NOT_EVALUABLE` status, and the complete holdout registry.
- Write the immutable freeze bundle to
  `artifacts/attribution-v1-freeze-<manifest-sha256>/` and run the existing final
  CLI in `--verify-only` mode. Do not generate a holdout trace or execute an arm.

## Acceptance criteria

1. Development results cannot change any frozen threshold, arm, main scale,
   endpoint, or conclusion rule; deviations require a new documented design
   version before holdout generation.
2. The registry contains exactly 40 macro-seeds x 50 episode labels in the
   `attribution:v1:holdout` namespace and exposes only identifiers/hashes, no
   generated workload outcomes.
3. The official A2/A3 numerical/feasibility gate passes on frozen calibration or
   the ticket reports blocked without weakening the contract. D1-D3 are each
   frozen as qualified/scheduled or `NOT_EVALUABLE` with a stable solver reason.
4. Repeated freeze construction is byte-identical; tampering with any frozen
   input is detected before simulation.
5. Full tests and environment check pass; no holdout trace or result exists, and
   the frozen manifest proves ticket 07 can execute with no code/config change.

## Progress log

## Answer

## Comments
