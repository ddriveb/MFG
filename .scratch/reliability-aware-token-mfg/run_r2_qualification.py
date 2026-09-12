from __future__ import annotations

from dataclasses import asdict
import json

from mfg_hedge.artifacts import write_run_directory
from mfg_hedge.token_mfg_qualification_campaign import (
    QualificationPlan,
    R2_RUN_ID,
    qualification_source_fingerprint,
    run_k_scale_timing_preflight,
)
from mfg_hedge.token_mfg_streaming_formal import StreamingFormalQualificationRunner


def main() -> None:
    plan = QualificationPlan.r2()
    preflight = run_k_scale_timing_preflight(plan)
    plan.require_formal_start(preflight)
    runner = StreamingFormalQualificationRunner(
        plan=plan,
        preflight=preflight,
        checkpoint_root=".scratch/reliability-aware-token-mfg/qualification-runs",
        run_id=R2_RUN_ID,
        worker_count=8,
    )
    status, rows = runner.run_qualification()
    summary = {
        "run_id": R2_RUN_ID,
        "status": status,
        "claim_boundary": "finite_regularized_mfg_qualification",
        "claims_exact_nash": False,
        "claims_unique_mfg": False,
        "plan_fingerprint": plan.protocol_fingerprint,
        "source_fingerprint": qualification_source_fingerprint(),
        "preflight": asdict(preflight),
        "checkpoint": {
            "reserved_calls": runner.checkpoint.reserved_calls,
            "completed_calls": runner.checkpoint.completed_calls,
        },
        "model_runs": [asdict(row) for row in rows],
        "holdout_started": False,
    }
    path = write_run_directory(
        "artifacts",
        R2_RUN_ID,
        {
            "summary.json": summary,
            "protocol.json": asdict(plan),
            "manifest.json": {
                "run_id": R2_RUN_ID,
                "plan_fingerprint": plan.protocol_fingerprint,
                "source_fingerprint": qualification_source_fingerprint(),
            },
        },
    )
    print(json.dumps({
        "status": status,
        "artifact": str(path),
        "reserved_calls": runner.checkpoint.reserved_calls,
        "completed_calls": runner.checkpoint.completed_calls,
        "model_runs": len(rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
