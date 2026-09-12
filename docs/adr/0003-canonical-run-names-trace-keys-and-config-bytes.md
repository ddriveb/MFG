# ADR-0003: Canonical Run Names, Dense Trace Keys, and Single-Read Configuration

Status: Accepted

Date: 2026-09-03

Clarifies: ADR-0002

## Context

ADR-0002 confines artifacts to a validated run directory and binds Primary service draws to `(token_id, replica_id, attempt_id)`. Windows aliases names with trailing dots and reserves device names such as `CON` and `NUL`, so an ASCII path-component regex alone does not guarantee a canonical physical directory. Array-backed service draws also require a precise Token-ID domain. Finally, reading a configuration once for parsing and again for hashing permits a concurrent edit to make the parsed configuration and recorded digest disagree.

## Decision

1. A run ID must be one portable, canonical path component: it matches the existing ASCII character set, does not end with a dot, and its stem is not a Windows reserved device name, case-insensitively.
2. Array-backed workload traces use dense Token IDs. Tokens must appear in arrival order with IDs exactly `0..token_count-1`; the simulator rejects any other trace before queue evaluation.
3. CLI configuration provenance comes from one immutable byte string. Those bytes are decoded and parsed into `ExperimentConfig`, and the same bytes are hashed into `config_sha256`.

## Consequences

- Logical run identity matches the physical directory name on supported platforms.
- Stable service-draw indexing cannot silently use Python negative indexes or alias duplicate IDs.
- `resolved_config` and `config_sha256` describe the same file content even if the on-disk file changes after the initial read.
- These are validation/provenance changes only; valid default runs retain their scientific behavior.
