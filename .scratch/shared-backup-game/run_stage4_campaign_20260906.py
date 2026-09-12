"""Guarded entry point for the one authorized frozen Stage 4 campaign run."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

from mfg_hedge.stage4_campaign import run_frozen_stage4_campaign


if __name__ == "__main__":
    multiprocessing.freeze_support()
    result = run_frozen_stage4_campaign(
        artifacts_root=Path("artifacts"),
        run_id="stage4-pi256-fit-validation-20260906-r2",
        max_workers=8,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True), flush=True)
