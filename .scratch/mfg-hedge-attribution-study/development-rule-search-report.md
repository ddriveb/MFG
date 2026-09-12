# 81-rule transient-control development result

Date: 2026-09-05

Status: complete development fit; not qualification, holdout, or MFG evidence.

## Frozen run

- 64 independent episodes, macro-seed 20260905, episode indices 0..63.
- Namespace `transient-control:v1:fit`; 18,709 generated/completed Tokens.
- Load 0.45; timeline H[0,100), D[100,200), F[200,220), R[220,320),
  followed by drain; slowdown 2; `tau0=1.6385447242959017`.
- All 81 N/D/I rules over early-Regular, early-Urgent, late-Regular,
  late-Urgent. All reuse identical episode traces and ADR-0012 reservation.
- Main reservation: four 25-time D windows, cap 2.8125 per episode/window,
  mean charge 1, pooled FIFO, no refund.

## Result

Selected rule: **NIIN**

| Context | Requested action |
|---|---|
| Early D, Regular | Normal |
| Early D, Urgent | Immediate Hedge |
| Late D, Regular | Immediate Hedge |
| Late D, Urgent | Normal |

The result is sample-best in this exact bank under the frozen objective and
safety filters. It is not evidence that late-Urgent protection is generally
harmful; finite fit noise and a pooled first-come budget can both affect that arm.

| Pooled metric | NIIN | No-Hedge NNNN | NIIN / NNNN |
|---|---:|---:|---:|
| Objective J | 14.887604 | 20.578092 | 0.723469 |
| D fractional-tail CVaR95 | 10.432193 | 16.373484 | 0.637139 |
| F fractional-tail CVaR95 | 10.229315 | 13.130886 | 0.779027 |
| Total executed work | 19,051.501353 | 18,817.268380 | 1.012448 |
| Non-winner executed work | 342.960114 | 38.563332 | 8.893441 |

Thus the fit point has 27.65% lower declared J, 36.29% lower D tail, 22.10%
lower F tail, and 1.24% more total executed work. Waste rose substantially in
relative terms; its absolute per-Token contribution rose from 0.002061 to
0.018331 and is explicitly charged in J rather than hidden.

Safety ratios were H mean=1.0000, H P95=1.0000, R mean=0.972934, R
P95=0.968298, and total work=1.012448. All 81 rules passed the prospective
filters. Therefore the filters were nonbinding in this low-cap run; the selected
rule was determined by J rather than constraint exclusion.

NIIN made 1,401 eligible Hedge requests; reservation admitted 451 and projected
950 to Normal (67.81% request suppression), reserving 451 mean-work units.
The physical engine received exactly the 451 applied requests. Reservation is
not a realized-work cap.

D Regular mean latency changed 3.571922 -> 2.720733 and D Urgent changed
3.649848 -> 2.442203. D Replay rates changed 3.8018% -> 2.7125% for Regular
and 3.3539% -> 2.3831% for Urgent. F mean latency also improved through
carry-in/spillover only: F arrivals themselves always requested Normal.

## Reproducibility and files

The full 64x81 computation was repeated in memory without writing another
artifact. All 81 rule rows and all 64 trace fingerprints matched the committed
artifact exactly; selection and comparison numbers were identical.

The transactional artifact contains:

- `manifest.json`: frozen configuration, episode registry, protocol, rule-bank
  digest, safety gates and simulator version.
- `rule_results.json`: all 81 complete rows plus every trace fingerprint.
- `summary.json`: selected and NNNN rows and their paired descriptive deltas.

## Interpretation boundary

This is selection on one burned development set. It has no independent sampling
uncertainty estimate and cannot establish generalization, superiority,
mean-field value, equilibrium, or global optimality. The next legitimate step is
to apply the already selected NIIN once to a separately keyed qualification set,
without examining other rules or changing weights/budget/gates. A failed
qualification is retained as a negative result; NIIN is not replaced using that data.
