"""Read-only analysis of the r2 checkpoint blobs for ticket 06 protocol redesign.

Decodes the committed per-episode sufficient-statistics blobs of
reliability-aware-token-mfg-qualification-20260911-r2 and recomputes the
per-cell support distribution that drove the `statistics_insufficient` gate.
No trace is generated and no scheduler call is dispatched.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from mfg_hedge.token_mfg_qualification_state_machine import _unpack

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / ".scratch" / "reliability-aware-token-mfg" / "qualification-runs" / "reliability-aware-token-mfg-qualification-20260911-r2"


def main() -> None:
    checkpoint = json.loads((RUN / "checkpoint.json").read_text(encoding="utf-8"))
    records = checkpoint["records"]
    complete = {
        key: rec for key, rec in records.items()
        if rec.get("status") == "complete" and "complete-panel" in key
    }
    print(f"complete panel records: {len(complete)}")

    cont_counts: Counter = Counter()
    finite_counts: Counter = {}
    finite_pairs: dict = {}
    token_state_counts: Counter = Counter()
    total_tokens = 0
    target_states: Counter = Counter()
    per_episode_calls = Counter()

    for key, rec in sorted(complete.items()):
        blob_ref = rec["output"]["__content_addressed_blob__"]
        blob = json.loads((RUN / "blobs" / blob_ref).read_text(encoding="utf-8"))
        per_episode_calls[rec["completed_calls"]] += 1
        target_state = blob["target_state"]
        target_states[target_state] += 1
        for row in _unpack(blob["compact_tokens"]):
            token_state_counts[row[0]] += 1
            total_tokens += 1
        for row in _unpack(blob["continuation_rows"]):
            cont_counts[(target_state, row.action_bucket.fingerprint)] += 1
        for row in _unpack(blob["finite_rows"]):
            key2 = (target_state, row.candidate_bucket.fingerprint)
            finite_counts[key2] = finite_counts.get(key2, 0) + 1
            finite_pairs.setdefault(key2, []).append(
                (-float(row.pathwise_difference), float(row.baseline_cost))
            )

    cells = sorted(set(cont_counts) | set(finite_counts))
    print(f"distinct states (targets): {len(target_states)}")
    print(f"distinct (state, action) cells: {len(cells)}")
    print(f"total tokens counted: {total_tokens}")
    print(f"per-episode completed calls: {dict(per_episode_calls)}")

    cell_counts = [finite_counts.get(c, cont_counts.get(c, 0)) for c in cells]
    cont_only = [cont_counts.get(c, 0) for c in cells]
    mismatch = sum(1 for c in cells if cont_counts.get(c, 0) != finite_counts.get(c, 0))
    print(f"continuation-vs-finite count mismatches: {mismatch}")

    def quantile(values, q):
        values = sorted(values)
        if not values:
            return None
        pos = (len(values) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return values[int(pos)]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    print(f"cell count min={min(cell_counts)} q25={quantile(cell_counts, 0.25):.1f} "
          f"median={quantile(cell_counts, 0.5):.1f} mean={sum(cell_counts)/len(cell_counts):.2f} "
          f"q75={quantile(cell_counts, 0.75):.1f} max={max(cell_counts)}")
    hist = Counter(cell_counts)
    print(f"cell count histogram: {dict(sorted(hist.items()))}")
    below8 = sum(1 for n in cell_counts if n < 8)
    print(f"cells below floor 8: {below8}/{len(cell_counts)}")

    # per-state episode counts and occupancy
    state_eps = sorted(target_states.values())
    print(f"state episode-count min={min(state_eps)} median={quantile(state_eps, 0.5):.1f} "
          f"max={max(state_eps)} sum={sum(state_eps)}")
    state_hist = Counter(state_eps)
    print(f"state episode-count histogram: {dict(sorted(state_hist.items()))}")

    # occupancy from first-batch tokens (uniform start -> forward occupancy proxy)
    occ = {s: n / total_tokens for s, n in token_state_counts.items()}
    occ_sorted = sorted(occ.values())
    print(f"token occupancy min={min(occ_sorted):.5f} median={quantile(occ_sorted, 0.5):.5f} "
          f"max={max(occ_sorted):.5f}")
    below05 = sum(1 for v in occ_sorted if v < 0.005)
    print(f"states with occupancy < 0.5%: {below05}/{len(occ_sorted)}")

    # per-cell normalized upper bound recomputation (critical value 2.0)
    normalized_uppers = []
    for c in cells:
        pairs = finite_pairs.get(c, [])
        if not pairs:
            continue
        gains = [g for g, _ in pairs]
        baselines = [b for _, b in pairs]
        n = len(gains)
        mean = math.fsum(gains) / n
        if n > 1:
            var = math.fsum((g - mean) ** 2 for g in gains) / (n - 1)
            se = math.sqrt(var / n)
        else:
            se = 0.0
        baseline = math.fsum(baselines) / n
        normalized_uppers.append(((mean + 2.0 * se) / baseline, n, c))
    normalized_uppers.sort(reverse=True)
    print(f"recomputed max normalized upper: {normalized_uppers[0][0]:.6f} "
          f"(recorded: 1.190440)")
    print("top-5 cells by normalized upper:")
    for value, n, c in normalized_uppers[:5]:
        print(f"  n={n} upper={value:.4f} state={c[0][:12]} action={c[1][:12]}")

    report = {
        "episodes": len(complete),
        "states": len(target_states),
        "cells": len(cells),
        "cell_counts": cell_counts,
        "state_episode_counts": state_eps,
        "state_occupancy": occ,
        "below_floor_cells": below8,
        "per_episode_calls": dict(per_episode_calls),
        "recomputed_max_normalized_upper": normalized_uppers[0][0],
    }
    out = Path(__file__).with_name("r2_cell_occupancy_report.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report written: {out}")


if __name__ == "__main__":
    main()
