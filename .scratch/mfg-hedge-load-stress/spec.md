# MFG-Hedge load 0.6/0.7 stress diagnostics

## Question

Under the predeclared diagnostic loads 0.6 and 0.7, does the frozen production
calibration and ADR-0008 solver produce an official-gate-eligible MFG policy?

## Rules

- Start from the exact schema-3 load-0.5 paired configuration and override
  only `healthy_offered_load` in memory.
- Build a fresh production 70-cell table for each load and run the unchanged
  solver.
- Record every state status, reason, clamp flag, rho, work rates, capacity gap,
  residual, and policy probabilities.
- These are pressure diagnostics, never pooled with load-0.5 and never called
  official A/B results.
- If H/D/F do not satisfy the official gate, do not project actions or run a
  misleading paired simulation.
- Commit one fresh transactional artifact directory; never overwrite.

## Acceptance criteria

1. Unit tests cover gate classification for accepted and rejected solutions.
2. Full tests and environment check pass.
3. A real diagnostic run persists load 0.6 and 0.7 results and explicitly
   states whether a paired simulation was scientifically admissible.
