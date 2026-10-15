"""Training-only smooth upper bound of the unchanged physical duality gap.

Inputs must be feasible model outputs on the graph's simplexes. Model softmax
heads enforce that contract; independent certificate/feasibility checks belong
in evaluation. Nothing here performs an inference-time correction or changes
the physical objective. Torch is imported only when explicitly called.
"""
import math


def direct_gap_loss(graph, shares, dual, *, smoothing_ms):
    """Return raw bounds and a differentiable gap with bias at most smoothing_ms.

    Half the bias budget is divided among resource maxima and half among row
    minima. Padded route slots have zero softmin mass; singletons are exact.
    The optimized scalar is the smooth gap divided by the fixed uniform F.
    """
    if not math.isfinite(smoothing_ms) or smoothing_ms <= 0:
        raise ValueError('smoothing_ms must be finite and positive')
    import torch
    loads = graph.apply(shares)
    upper = shares.new_tensor(graph.problem.fixed_startup_ms)
    smooth_upper = upper.clone()
    for load in loads:
        upper = upper + load.max()
        if load.numel() == 1:
            smooth_upper = smooth_upper + load[0]
        else:
            tau = smoothing_ms / (2 * len(loads) * math.log(load.numel()))
            smooth_upper = smooth_upper + tau * torch.logsumexp(load / tau, dim=0)
    gradient = graph.adjoint(dual)
    padded = gradient[graph.row_slot_indices]
    masked = padded.masked_fill(~graph.row_slot_mask, torch.inf)
    lower = shares.new_tensor(graph.problem.fixed_startup_ms) + masked.min(-1).values.sum()
    widths = graph.row_slot_mask.sum(-1).to(shares.dtype)
    tau_row = smoothing_ms / (2 * graph.row_count * widths.clamp_min(2).log())
    smooth_lower = shares.new_tensor(graph.problem.fixed_startup_ms) - (
        tau_row * torch.logsumexp(-masked / tau_row[:, None], dim=-1)
    ).sum()
    smooth_gap = smooth_upper - smooth_lower
    return dict(upper_ms=upper, lower_ms=lower, gap_ms=upper-lower,
                smooth_upper_ms=smooth_upper, smooth_lower_ms=smooth_lower,
                smooth_gap_ms=smooth_gap, total=smooth_gap / graph.f_reference)
