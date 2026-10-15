"""Exact uneven flow input for the isolated SimAI ns-3 replay driver.

This is a barrier-separated single MoE stage, not full training or default
AICB uniform all-to-all. Measured rank times are converted from ms to ns.
"""
import math

PHASES = ("forward_dispatch", "forward_combine", "backward_combine_grad", "backward_dispatch_grad")

def encode_replay(workload: dict) -> str:
    g = workload["gpu_count"]
    if not isinstance(g, int) or not 1 <= g <= 512:
        raise ValueError("invalid GPU count")
    profiles = workload["existing_local_compute_profiles"]
    if len(profiles) != g or sorted(p["gpu"] for p in profiles) != list(range(g)):
        raise ValueError("one exact compute profile per GPU required")
    lines = ["MFG_MATRIX_REPLAY_V1", str(g)]
    for p in sorted(profiles, key=lambda p: p["gpu"]):
        times = [p["local_profile_forward_ms"], p["local_profile_backward_ms"]]
        if any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError("invalid compute duration")
        lines.append(" ".join(str(round(t * 1_000_000)) for t in times))
    for phase in PHASES:
        matrix = workload["payload_bytes_by_sender_receiver"][phase]
        if len(matrix) != g or any(len(row) != g for row in matrix):
            raise ValueError("matrix shape mismatch")
        flows = []
        for src, row in enumerate(matrix):
            for dst, size in enumerate(row):
                if not isinstance(size, int) or isinstance(size, bool) or size < 0 or size >= 2**63:
                    raise ValueError("payload must be a nonnegative integer")
                if src == dst and size:
                    raise ValueError("self routes must not enter network")
                if size:
                    flows.append(f"{src} {dst} {size}")
        lines.extend([f"{phase} {len(flows)}", *flows])
    return "\n".join(lines) + "\n"
