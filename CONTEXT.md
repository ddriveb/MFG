# MFG-Hedge v1 Domain Context

## Purpose

MFG-Hedge v1 is a pure-Python mechanism experiment for one question: when a shared degradation or failure affects many MoE Tokens, can a price-mediated population policy coordinate protection work without creating a Protection Storm?

## V1 boundary

The first version is deliberately limited to:

- one MoE layer;
- Top-1 Gate output fixed before protection decisions;
- inference only;
- fixed Active-Active replica placement across failure domains;
- one affected failure domain;
- at most one Replay;
- no training, cross-layer coupling, dynamic placement, or full HJB-FPK system.

Begin with one logical Expert and two Replicas. Expand to eight Experts only after the minimal mechanism is verified.

## Ubiquitous language

### Logical Expert

The Expert selected by the Top-1 Gate. Protection never changes this selection.

### Replica

One physical execution copy of a Logical Expert. Primary and Backup replicas contain the same Expert parameters and must be in different failure domains.

### Primary / Backup

The fixed Dispatcher chooses the Primary. The Backup is a same-Expert replica in another failure domain. The protection policy chooses whether and when the Backup starts; it does not choose another Expert.

### Common State

The shared health state `H` (Healthy), `D` (Degraded), or `F` (Failed).

### Protection Action

`N` (Normal), `D` (Delayed Hedge), or `I` (Immediate Hedge).

### Hedge

A second same-Expert execution started for protection. Hedge work is controllable extra work.

### Replay

A replacement execution after a failed copy. Replay work is extra work even when the original action was Normal.

### Protection Storm

A population-level burst of protection launches that amplifies queueing or overload. Measure it with peak Hedge launch rate, peak/mean ratio, peak utilization, sustained overload duration, and failure-window P99; do not assume it exists from policy names alone.

### Action Stats

The coarse statistics used by a policy solver: mean latency, Replay probability, deadline-miss probability, expected Hedge work, expected Replay work, and (under ADR-0009) expected wasted work from executed non-winning attempts.

### Quota Projector

The deterministic mechanism that converts a mixed population policy into per-window Token assignments while enforcing a hard Hedge-work budget.

## Invariants

1. Protection never changes the Logical Expert selected by the Gate.
2. Solver/calibration statistics and event-simulator evaluation remain separate.
3. Effective utilization includes Primary, Hedge, and Replay work.
4. Replay probability and deadline-miss probability remain separate metrics.
5. A finite price and Softmax distribution do not by themselves guarantee a hard capacity constraint; quota projection is required.
6. All compared policies consume the same pre-generated workload and failure trace.
7. Base-load infeasibility is reported separately from Hedge-budget violation.
8. Array-backed workload traces use unique, ordered Token IDs exactly equal to `0..token_count-1`.

## Avoid these ambiguous terms

- Do not use “Expert rerouting” for Primary-to-Backup execution; call it same-Expert replica protection.
- Do not use “failure rate” when the value is specifically Replay rate or deadline-miss rate.
- Do not use “extra work” without stating whether it includes Hedge work, Replay work, or both.
- Do not call the current conditional finite-action game a complete dynamic common-noise MFG.
