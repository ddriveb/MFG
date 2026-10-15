"""Independent physical objectives for separately parameterized networks.

Splitting loss scalars alone does not isolate shared parameters. Callers must
use disjoint allocation/price parameters and separate optimizers. Acceptance
uses the original unsmoothed certificate, never these training surrogates.
"""
from .moe_gpu_direct_gap_training_v1 import direct_gap_loss


def separated_physical_losses(graph, shares, dual, *, smoothing_ms):
    """Expose allocation descent and price ascent without cross-output gradients.

    The allocation branch minimizes its execution upper bound; the price
    branch maximizes a valid conservative lower bound. Their normalized sum
    is ADR0195's smooth gap with its unchanged physical-ms bias guarantee.
    """
    result = direct_gap_loss(graph, shares, dual, smoothing_ms=smoothing_ms)
    return dict(result, primal=result['smooth_upper_ms'] / graph.f_reference,
                dual=-result['smooth_lower_ms'] / graph.f_reference)
