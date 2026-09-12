"""Read-only Token restoration evidence; no campaign or production mutation."""
from pathlib import Path
import ast
import hashlib
import json

from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction as A, TokenClass as K
from mfg_hedge.hedge_simulation import simulate_hedge_common_state
from mfg_hedge.workload import TokenSpec, WorkloadTrace

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("engineering-evidence.json")
paths = sorted((ROOT / "src/mfg_hedge").glob("*.py"))
before = {str(p.relative_to(ROOT)).replace("\\", "/"):
          hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
modules = {}
for p in paths:
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    modules[p.name] = {
        "lines": len(p.read_text(encoding="utf-8-sig").splitlines()),
        "definitions": {n.name: n.lineno for n in tree.body
                        if isinstance(n, (ast.ClassDef, ast.FunctionDef))},
    }

# Both Tokens belong to ONE Expert. Only Token 0 changes its own action.
# Its extra B work delays Token 1; no cross-Expert pool is present.
tokens = (TokenSpec(0, 11.0, K.REGULAR), TokenSpec(1, 11.1, K.URGENT))
trace = WorkloadTrace(
    base_seed=1, arrival_rate=0.9, tokens=tokens,
    service_times=((2.0, 9.0), (9.0, 1.0)),
    replay_service_times=((1.0, 1.0), (1.0, 1.0)),
    hedge_service_times=((1.0, 1.0), (1.0, 1.0)),
)
timeline = CommonStateTimeline(10.0, 20.0, 30.0)
runs = {}
for action in (A.NORMAL, A.IMMEDIATE_HEDGE):
    result = simulate_hedge_common_state(
        trace, timeline=timeline, degraded_slowdown=2.0,
        actions={} if action is A.NORMAL else {0: action},
    )
    runs[action.value] = {
        "latencies": [t.latency for t in result.tokens],
        "completion_times": [t.completion_time for t in result.tokens],
        "executed_work": [sum(a.executed_work for a in t.attempts)
                          for t in result.tokens],
    }
assert abs(runs["N"]["latencies"][0] - 4.0) < 1e-10
assert abs(runs["I"]["latencies"][0] - 1.0) < 1e-10
assert abs(runs["N"]["latencies"][1] - 1.0) < 1e-10
assert abs(runs["I"]["latencies"][1] - 1.9) < 1e-10
after = {str(p.relative_to(ROOT)).replace("\\", "/"):
         hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
output = {
    "kind": "read_only_design_audit", "scheduler_calls": 2,
    "claims_equilibrium": False,
    "single_expert_token_externality": runs,
    "source_hashes_at_inspection": before,
    "source_changed_during_probe": [p for p in before if before[p] != after[p]],
    "modules": modules,
}
OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in output.items()
                  if k not in ("source_hashes_at_inspection", "modules")}, indent=2))
