# ADR-0024: Token-MFG T4b soft-response fixed-point diagnostic

Status: Proposed

Date: 2026-09-08

## Context

T4-v1 used a deterministic hard argmin response table with function damping.
The accepted T4 run produced a period-two response-table cycle.  Continuing
the same hard update would repeat the cycle and would not distinguish a real
population response flip from a Q-estimation fluctuation.

## Decision

Add a separate T4b diagnostic using a finite-temperature conditional response:

\[
 \operatorname{SoftBR}(a\mid b)=
 \frac{\exp[-\beta Q(a\mid b)]}
 {\sum_{a'}\exp[-\beta Q(a'\mid b)]}.
\]

The frozen first implementation uses `beta=1.0`, `eta=0.2`, zero external
price, and at most 20 response rounds.  The initial policy is the exact online
NIIN function.  Every subsequent policy is the function mixture

```text
pi_next = (1 - eta) * pi_current + eta * SoftBR(Q_current)
```

The policy samples N/D/I online from the current per-bin probability vector;
it never replays actions from an earlier round.

## Frozen execution contract

- Environment, runtime ADR-0009 cost model, reservation, slowdown, hedge
  delay, dispatcher, complete drain, bin schema, nine retained bins, and
  32-panel per `(bin, action)` floor remain those of T4-v1.
- A fresh immutable library of 2,048 episode traces is generated once under
  namespace `token-mfg-restoration:t4b:soft-fixed-point:v1:library`, macro seed
  `20260914`, and the T4 attribution protocol.  The same trace object and
  fingerprint are reused in every response round; each physical branch still
  creates a fresh endogenous online engine state.
- Each episode consumes one online target-selection run and, when a target is
  selected, one complete baseline plus N/D/I counterfactual panel.  The
  maximum is 10,240 calls per round and 204,800 calls over 20 rounds.  There
  is no retry, supplement, bin merge, or result-driven sample change.
- Q rows retain mean, ordinary SE, effective n, status, and action-gap
  diagnostics.  For the best and runner-up actions, the gap uses paired
  within-panel differences and reports paired SE.  The gap is descriptive and
  does not select a hard action.
- The response table stores all three probabilities per retained bin.  Unknown
  bins retain the NIIN base function.  N/D/I sampling uses a stable private
  hash of the policy fingerprint and online policy key.
- Convergence requires two comparable rounds with maximum per-observation
  probability L1 residual and selected-bin occupancy L1 residual both at or
  below `0.01`.  Repeated hard-argmin fingerprints are retained only as a
  diagnostic; they cannot by themselves produce a cycle status.
- Allowed final statuses are `soft_fixed_point`, `not_converged_20`, and
  `statistics_insufficient`.  None implies Nash, regret, or MFG validity.
- Price feedback, predictor training, best-response claims, regret claims,
  and MFG forward closure remain outside this ADR.

## Consequences

Using one library removes a major cross-round Monte Carlo source, but it does
not make the finite trace an exact conditional law.  Paired gaps expose when
hard flips are small relative to panel noise.  Soft probabilities may damp a
population oscillation, but convergence remains an empirical diagnostic only.

## Claim boundary

Every T4b artifact must set `claims_best_response`, `claims_regret`,
`claims_nash`, and `claims_mfg` to `false`.
