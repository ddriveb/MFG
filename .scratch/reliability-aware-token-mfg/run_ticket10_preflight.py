from __future__ import annotations

import json

from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationPlan,
    run_k_scale_timing_preflight,
)


if __name__ == "__main__":
    report = run_k_scale_timing_preflight(QualificationPlan())
    print(json.dumps({
        "scheduler_calls": report.scheduler_calls,
        "worst_k64_call_seconds": report.worst_k64_call_seconds,
        "projected_eight_worker_seconds": report.projected_eight_worker_seconds,
        "parent_wait_seconds": report.parent_wait_seconds,
        "k64_parallel_elapsed_seconds": report.k64_parallel_elapsed_seconds,
        "k64_throughput_calls_per_second": report.k64_throughput_calls_per_second,
        "parent_peak_rss_bytes": report.parent_peak_rss_bytes,
        "worker_peak_rss_bytes": [row.peak_rss_bytes for row in report.throughput_rows],
        "rss_supported": report.rss_supported,
        "complete": report.complete,
        "rows": [
            {
                "k": row.k,
                "scenario": row.scenario,
                "wall_seconds": row.wall_seconds,
                "scheduler_wall_seconds": row.scheduler_wall_seconds,
                "preparation_seconds": row.preparation_seconds,
                "cpu_seconds": row.cpu_seconds,
                "serialization_bytes": row.serialization_bytes,
                "peak_rss_bytes": row.peak_rss_bytes,
            }
            for row in report.rows
        ],
    }, sort_keys=True, indent=2))
