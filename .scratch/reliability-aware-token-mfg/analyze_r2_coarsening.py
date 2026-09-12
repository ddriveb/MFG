"""Read-only coarsening analysis over r2 cells for ticket 06 fork 1.

Reconstructs every observed (state, action) cell's replica-side action bucket
labels from the committed r2 blobs and simulates replica-side bucket merges
(hazard / queue / work / running-age). Token-side bucket fields are NOT present
in the compact blobs (only token_class is recorded per state), so token-side
merges are reported as missing-data items rather than fabricated.
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


def load_cells():
    checkpoint = json.loads((RUN / "checkpoint.json").read_text(encoding="utf-8"))
    records = {
        k: r for k, r in checkpoint["records"].items()
        if r.get("status") == "complete" and "complete-panel" in k
    }
    cells = {}
    state_class = {}
    state_ep = Counter()
    for key, rec in sorted(records.items()):
        blob_ref = rec["output"]["__content_addressed_blob__"]
        blob = json.loads((RUN / "blobs" / blob_ref).read_text(encoding="utf-8"))
        target = blob["target_state"]
        state_ep[target] += 1
        for row in _unpack(blob["compact_tokens"]):
            state_class[row[0]] = row[1]
        for row in _unpack(blob["finite_rows"]):
            ck = (target, row.candidate_bucket.fingerprint)
            entry = cells.setdefault(ck, {"n": 0, "bucket": None})
            entry["n"] += 1
            if entry["bucket"] is None:
                b = row.candidate_bucket
                entry["bucket"] = {
                    "domain_id": b.domain_id,
                    "health": b.health,
                    "queue_bin": b.queue_bin,
                    "work_bin": b.work_bin,
                    "running_age_bin": b.running_age_bin,
                    "hazard_bin": b.hazard_bin,
                    "failure_risk_bin": b.failure_risk_bin,
                    "locality": b.locality,
                }
    return cells, state_class, state_ep


def summarize(cells, state_ep, merge_fields):
    """Group cells by (state, merged action bucket); report cell count + counts."""
    grouped = {}
    for (state, _action), entry in cells.items():
        b = entry["bucket"]
        merged = tuple(sorted((f, b[f]) for f in merge_fields))
        grouped.setdefault((state, merged), []).append(entry["n"])
    counts = [sum(ns) for ns in grouped.values()]
    below8 = sum(1 for n in counts if n < 8)
    return {
        "cells": len(grouped),
        "count_min": min(counts),
        "count_median": sorted(counts)[len(counts) // 2],
        "count_max": max(counts),
        "below_floor_8": below8,
    }


def main() -> None:
    cells, state_class, state_ep = load_cells()
    print(f"observed cells: {len(cells)}, target states: {len(state_ep)}")

    # bucket label diversity across the 63 cells
    for field in ("queue_bin", "work_bin", "running_age_bin", "hazard_bin",
                  "failure_risk_bin", "health", "locality"):
        values = Counter(entry["bucket"][field] for entry in cells.values())
        print(f"{field}: {dict(sorted(values.items()))}")

    # state x class table
    cls = Counter(state_class.get(s, "?") for s in state_ep)
    print(f"target-state token classes: {dict(cls)}")

    # floor reach under replica-side-only merges; episodes needed for the
    # rarest merged cell under the frozen panel (target draw per episode).
    base = summarize(cells, state_ep,
                     ("domain_id", "health", "queue_bin", "work_bin",
                      "running_age_bin", "hazard_bin", "failure_risk_bin", "locality"))
    print("baseline (no merge):", base)

    schemes = {
        "A_merge_hazard_pair": ("domain_id", "health", "queue_bin", "work_bin",
                                 "running_age_bin", "locality"),
        "B_merge_queue_top": ("domain_id", "health", "work_bin",
                              "running_age_bin", "hazard_bin", "failure_risk_bin", "locality"),
        "C_merge_hazard_and_queue": ("domain_id", "health", "work_bin",
                                     "running_age_bin", "locality"),
    }
    for name, fields in schemes.items():
        print(name, summarize(cells, state_ep, fields))

    # how many episodes would the dominant/rare structure need?
    # per-state episode counts are unchanged by action-side merges; report them.
    print("state episode counts:", dict(sorted(state_ep.items(), key=lambda kv: -kv[1])))

    out = Path(__file__).with_name("r2_cell_bucket_table.json")
    table = [
        {"state": s[:16], "token_class": state_class.get(s, "?"),
         "state_episodes": state_ep.get(s, 0), "n": e["n"], **e["bucket"]}
        for (s, _a), e in sorted(cells.items())
    ]
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"table written: {out}")


if __name__ == "__main__":
    main()
