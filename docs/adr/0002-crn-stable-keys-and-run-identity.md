# ADR-0002: Bind Service Draws to Stable Keys and Pin Run Identity in Artifacts

Status: Accepted

Date: 2026-09-03

## Context

The first Healthy + No Hedge slice consumed each Replica's service-time stream by assignment order. Under common random numbers this is fragile: any policy that changes dispatch or starts a Hedge/Backup would shift the same service draws onto different Token copies, so compared policies would no longer face identical per-copy conditions. The slice also wrote `summary.json` with only partial configuration identity, and the artifact writer accepted any `run_id`, allowing a path such as `../escaped` to write outside `artifacts/`.

## Decision

1. Service-time draws are bound to the stable key `(token_id, replica_id, attempt_id)`. The workload trace stores `service_times[replica][token_id]` for attempt 0 (the Primary); Hedge and Replay slices will extend the key with `attempt_id > 0` draws instead of reusing or shifting attempt-0 draws. Simulators index streams by this key and keep no positional consumption counters.
2. Every `summary.json` pins full configuration identity: `config_schema_version`, the complete `resolved_config`, `config_sha256` over the configuration file bytes, plus `simulator_version`, `scenario`, `dispatcher`, and `policy`. The byte-identical determinism criterion still excludes only `run_id`.
3. `run_id` must be a single safe directory name; the artifact writer validates the resolved run directory stays inside the artifacts root and removes a partially created run directory if serialization or writing fails.

## Consequences

- Policies compared on the same trace face identical service draws per Token copy even when dispatch or protection behavior differs.
- A result can be traced to exact configuration content, not just an experiment name, so editing a config file in place cannot silently alias old runs.
- `simulator_version` follows the package version; behavior-affecting changes bump it (this hardening moved 0.1.0 to 0.2.0).
- Artifact writes are confined to `artifacts/<run-id>/` and never leave partial run directories behind.
