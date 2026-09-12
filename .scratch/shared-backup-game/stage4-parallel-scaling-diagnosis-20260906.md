# Stage 4 parallel-scaling diagnosis — 2026-09-06

This is a performance diagnosis only. It does not contain Fit, Validation,
a candidate rule, a Nash result, or an MFG result.

## Finding

The previous 59.29-hour projection is not representative of the frozen
campaign task shape. It divided an eight-process batch containing only one
scenario per task by eight calls, so Windows spawn and initialization were
charged almost entirely to those calls. A frozen Fit task processes 32
scenarios consecutively inside one worker, and the process pool handles many
tasks before shutdown.

Using the actual 32-scenario task shape, eight workers sustain
13.8816 scheduler calls/second. The mechanical 921,088-call upper bound is
therefore 18.43 hours, not 59.29 hours. This is still a timing estimate and
does not authorize campaign launch.

## Probe contract

- namespace: `shared-backup-game:v1:stage4-fit`
- macro seed: `20260905`
- 16 common paths x 2 populations = 32 scenarios
- N=8, c_B=.5, slowdown=2, Hedge delay=1.5
- eight `(deviating Expert, NNNN)` tasks
- 256 full scheduler calls per worker-count measurement
- realized Tokens per scenario: 1,226 to 1,360
- Windows spawn; PreparedScenario tuple passed once through the initializer

## Scaling table

| Workers | Startup s | Steady s | Calls/s | Speedup | Scaling efficiency | Median task s | Worst-case hours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.411 | 83.604 | 3.062 | 1.000x | 100.0% | 10.357 | 83.56 |
| 2 | 0.702 | 43.569 | 5.876 | 1.919x | 95.9% | 10.833 | 43.54 |
| 4 | 1.408 | 24.685 | 10.371 | 3.387x | 84.7% | 12.247 | 24.67 |
| 8 | 2.766 | 18.442 | 13.882 | 4.533x | 56.7% | 18.254 | 18.43 |

`Scaling efficiency` is speedup divided by worker count. The near-99% worker
occupancy value reported by the diagnostic is not scaling efficiency: it says
that submitted workers remained busy, while each task itself became slower.

## Transport and parent overhead

- prepared 32-scenario initializer payload: 9,779,555 bytes, sent once per
  worker initialization;
- task payload: 212 bytes;
- returned result: about 26.9 KiB per complete 32-scenario task;
- total returned data for eight tasks: 215,826 bytes;
- eight-worker startup: 2.766 seconds;
- eight-worker parent CPU time over startup, execution and shutdown: .75 seconds;
- worker task CPU/wall ratio: about .95.

These measurements rule out parent merge and per-task trace/result transport
as the dominant steady-state bottleneck. The workers are compute-bound. From
one to eight workers, median task wall time grows from 10.357 to 18.254 seconds
(+76.3%), and median task CPU time grows from 10.148 to 17.383 seconds. This is
consistent with CPU frequency/cache/SMT or host scheduling contention. The
host exposes 12 logical processors; physical topology could not be queried
because the local CIM provider denied access, so no unsupported physical-core
claim is made.

## Remaining single-call hotspot

An in-process cProfile of one complete 32-scenario worker task recorded
46,606,878 function calls. Profiling adds overhead, so its absolute duration
is not used for campaign projection, but the proportions identify the next
safe optimization target:

| Function | Cumulative profiled time |
| --- | ---: |
| `_handle_arrival` | 11.637 s |
| `_observation` | 10.758 s |
| `_audit_queues` | 2.698 s |
| `score_tagged_result` | .209 s |
| aggregate `score_expert` | .073 s |

PreparedTrace changed observation from a scan over all Experts to the current
Expert, but every arrival still reconstructs that Expert's complete historical
winner/timer/failure/attempt tuples. This remains quadratic in the number of
Tokens per Expert. Attempt-key reconstruction alone produced about 3.4 million
calls in the profiled task. Queue audit also visits all 24 queues at every
event batch even though tagged output materializes per-event audit rows for
only one Expert.

## Disposition

The production batcher already uses the intended coarse task, worker-local
PreparedScenario state and compact returned rows. No production bug or CRN /
physics mismatch was found. The earlier projection method was the primary
parallel-scaling diagnosis.

If 18.43 worst-case hours remains outside the launch gate, the next exact
performance work should target single-call CPU cost:

1. preserve the generic full ActionObservation path, but add an explicitly
   verified compact online projection for the frozen Pi256 ActionRule, whose
   decision uses only current state, phase, phase age, class and Primary;
2. in tagged mode, retain incremental ledgers for all queues but materialize
   and check per-event audit rows only for the tagged Expert, followed by one
   terminal conservation verification over all queues;
3. repeat reference equivalence and the full 1/2/4/8 table before changing the
   campaign disposition.

These changes require their own red/green implementation update. The present
diagnosis does not change the scheduler or authorize a campaign.

## Reference-equivalent correction outcome

The follow-up implementation retained the general `ActionObservation` path
and added a private compact request protocol used only by the exact base
Pi256 `ActionRule` inside the campaign worker. Custom rule subclasses and
general policies continue to receive the full observation. Tagged runs now
audit only the tagged Expert at each event and perform one terminal physical
and algebraic conservation check over every Expert/Replica ledger.

The same frozen 32-scenario, eight-task probe was repeated after the change:

| Workers | Startup s | Steady s | Calls/s | Speedup | Scaling efficiency | Median task s | Worst-case hours |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.366 | 28.111 | 9.107 | 1.000x | 100.0% | 3.504 | 28.09 |
| 2 | 0.724 | 15.451 | 16.568 | 1.819x | 90.9% | 3.823 | 15.44 |
| 4 | 1.501 | 8.718 | 29.364 | 3.224x | 80.6% | 4.333 | 8.71 |
| 8 | 3.019 | 5.897 | 43.414 | 4.767x | 59.6% | 5.857 | 5.89 |

At eight workers this is 3.127x the preceding 13.8816 calls/second and reduces
the mechanical 921,088-call upper-bound projection from 18.43 to 5.89 hours
(Fit 5.05 hours plus Validation .84 hours at steady throughput). The number
remains a mechanical estimate, not campaign authorization.

The post-change cProfile fell from about 46.6 million calls / 18.3 profiled
seconds to 12.4 million calls / 5.6 seconds for a complete worker task.
`_observation` no longer appears among the leading cumulative-time rows;
`_audit_queues` fell from 2.698 to .892 profiled seconds. Full regression and
the exact projection tests prove unchanged actions, winners, scores, pool
physics, tagged rows, and stop semantics.
