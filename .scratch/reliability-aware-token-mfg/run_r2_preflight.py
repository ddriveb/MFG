from __future__ import annotations

import json

from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationPlan,
    run_k_scale_timing_preflight,
)


if __name__ == "__main__":
    plan = QualificationPlan.r2()
    report = run_k_scale_timing_preflight(plan)
    plan.require_formal_start(report)
    print(json.dumps({
        "plan_fingerprint": report.plan_fingerprint,
        "scheduler_calls": report.scheduler_calls,
        "formal_calls_consumed": report.formal_calls_consumed,
        "worst_k64_call_seconds": report.worst_k64_call_seconds,
        "projected_eight_worker_seconds": report.projected_eight_worker_seconds,
        "complete": report.complete,
        "rss_supported": report.rss_supported,
        "worker_count": report.worker_count,
    }, sort_keys=True))
