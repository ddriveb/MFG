# Implement attribution controllers and destination-B quota projection

Type: task
Status: open
Blocked by: 03

## Goal

Implement A0/A1/A2/A3 and diagnostic controller plans with identical causal
observations and ex-ante resource envelopes.

## Scope

- Implement exact No-Hedge, frozen RULE, MYOPIC-11, and mean-field-11 plans.
- Force H/F/R and D/Primary-1 to Normal in every arm, including diagnostics.
- Extend causal deficit/quota projection to policy context and expected lifecycle
  Hedge work sent to B at scales 0.25/0.50/0.75/1.00 using the single frozen
  `g_budget` table; report the solver's separate total B envelope.
- Add no-quota and 00/10/01 cost diagnostic plans without changing the event
  engine's copy lifecycle.

## Acceptance criteria

1. Every arm sees only phase/class/Primary at arrival, never completion/future
   data, and produces deterministic assignments on repeat input.
2. RULE is exactly the spec mapping; every arm's H/F/R and D/Primary-1 requested
   and applied counts are exactly all Normal.
3. Requested/applied/suppressed identities hold per window/context/class and
   globally; planned Hedge charge never exceeds the common cap. D0 reuses the
   identical phase-1 causal requests and skips only phase-2 quota projection;
   class shares are demand-proportional, isolated, reset, and expire as specified.
4. All arms consume identical trace fingerprints and the No-Hedge projection
   regresses against the accepted engine.
5. Full tests, focused controller/quota scenarios, and environment check pass;
   no experiment artifact is generated.

## Progress log

## Answer

## Comments
