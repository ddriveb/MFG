# Minimal Healthy + No Hedge Queue Simulation

## Goal

Build the smallest deterministic queue-simulation vertical slice that validates workload generation, fixed dispatch, per-Replica FCFS queueing, latency accounting, metrics, and artifact output under Healthy state with every Token using Normal protection.

## In scope

- One Logical Expert and two healthy Replicas in two failure domains.
- Poisson Token arrivals derived from healthy offered load and aggregate service capacity.
- Regular/Urgent Token classes using the versioned configuration ratios.
- Lognormal service requirements with configured mean and coefficient of variation.
- A deterministic fixed Dispatcher; use round-robin for this slice and document the choice.
- FCFS single-server queueing per Replica.
- Normal action only: exactly one Primary execution, no Backup launch and no Replay.
- Deterministic workload generation from `base_seed`, independent of policy code.
- Summary metrics and a versioned JSON artifact under a unique `artifacts/<run-id>/` directory.
- Unit and integration tests using only the Python standard library.

## Out of scope

- Degraded or Failed state.
- Hedge timers, Immediate Hedge, Backup execution, cancellation, Replay, MFG, prices, quota projection, multiple Experts, plotting, NumPy, Pandas, and GPU code.

## Required design

1. Generate an immutable workload trace before applying a policy. Keep arrival, Token-class, and per-Replica service-time random streams separate so future policy comparisons can reuse common random numbers.
2. Keep workload generation separate from queue evaluation. Do not let `NoHedgePolicy` consume random numbers.
3. Represent each Token result with at least Token ID, class, arrival time, Primary replica, start time, service time, completion time, queue delay, and completion latency.
4. Derive lognormal parameters from configured arithmetic mean and CV:
   - `sigma2 = log(1 + cv**2)`
   - `sigma = sqrt(sigma2)`
   - `mu = log(mean) - sigma2 / 2`
5. Calculate arrival rate as `healthy_offered_load * aggregate_service_capacity`; with two equal Replicas and mean service time 1.0, the default rate is 1.4.
6. Define percentile interpolation explicitly and test it. Do not rely on a third-party statistics package.
7. Write artifacts atomically or fail if the target run directory already exists. Never overwrite a previous run.

## Required outputs

- A reusable Python API for generating the workload and running Healthy + No Hedge.
- CLI command extending the existing module, for example:

  ```powershell
  .\.venv\Scripts\python.exe -m mfg_hedge simulate-healthy-no-hedge --config .\configs\v1_minimal.json --tokens 1000
  ```

- `summary.json` containing configuration identity, seed, Token count, arrival rate, mean latency, P50/P95/P99, mean queue delay, throughput, per-Replica assigned count, busy time and utilization, and invariant counters.
- `invariant counters` must show zero Hedge launches, zero Replay executions, and one Primary execution per completed Token.

## Acceptance criteria

- Same configuration and seed produce byte-identical `summary.json` content except for an explicitly excluded run identifier/path; preferably keep timestamps out of scientific content.
- Arrival times are non-decreasing and service times are positive.
- Replica assignment counts differ by at most one under round-robin dispatch.
- Every start time is at or after arrival; completion equals start plus service time; latency equals completion minus arrival.
- Completed Token count equals generated Token count.
- Hedge launch and Replay counts are both zero; Primary execution count equals Token count.
- All reported utilizations are in `[0, 1]` under their documented observation interval.
- Existing six tests remain green and new focused tests cover determinism, queue equations, dispatch balance, percentiles, invariants, and CLI artifact creation.
- No third-party dependency is added.

## Verification

Run at least:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json
.\.venv\Scripts\python.exe -m mfg_hedge simulate-healthy-no-hedge --config .\configs\v1_minimal.json --tokens 1000
```

