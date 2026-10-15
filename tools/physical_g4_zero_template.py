"""CPU structural check for G4 fixed-support reconstruction with zero rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "src"))


def check_zero_template() -> dict:
    import torch

    from mfg_hedge.granite_communication_social_v2_g4_training_policy_v1 import canonical_ring_rows
    from mfg_hedge.granite_g4_feedback_planner_v1 import default_model_config, dense_reference_problem
    from mfg_hedge.moe_communication_social_v2 import SparseSocialProblem
    from mfg_hedge.moe_communication_social_v2_g256_sparse_warmstart_v1 import SparseSocialTorchGraph

    config = default_model_config()
    template = SparseSocialProblem.from_rows(canonical_ring_rows(), config)
    counts = [0 if row % 5 == 0 else row % 7 + 1 for row in range(128)]
    problem = dense_reference_problem(counts, config)
    template_graph = SparseSocialTorchGraph.from_problem(template, device="cpu")
    graph = SparseSocialTorchGraph.from_problem(problem, device="cpu")

    if tuple(problem.row_offsets) != tuple(template.row_offsets):
        raise AssertionError("zero counts changed source-to-destination row support")
    if tuple(problem.variable_rows) != tuple(template.variable_rows):
        raise AssertionError("zero counts changed variable-to-row support")
    if tuple(problem.variable_gpus) != tuple(template.variable_gpus):
        raise AssertionError("zero counts changed the fixed replica placement")
    if not torch.equal(graph.incidence_resource, template_graph.incidence_resource):
        raise AssertionError("zero counts changed sparse resource incidence")
    if not torch.equal(graph.incidence_variable, template_graph.incidence_variable):
        raise AssertionError("zero counts changed sparse variable incidence")
    if any(a.resource_names != b.resource_names for a, b in zip(problem.blocks, template.blocks)):
        raise AssertionError("zero counts changed the fixed resource template")

    zero_rows = [row for row, demand in enumerate(counts) if demand == 0]
    zero_vars = [variable for variable, row in enumerate(problem.variable_rows) if row in zero_rows]
    if graph.row_features[zero_rows][:, [0, 1, 6]].abs().max().item() != 0.0:
        raise AssertionError("zero-count row features were not refreshed")
    if graph.edge_features[zero_vars, 7:].abs().max().item() != 0.0:
        raise AssertionError("zero-count edge resource features were not refreshed")
    zero_var_set = set(zero_vars)
    for incidence, variable in enumerate(graph.incidence_variable.tolist()):
        if variable in zero_var_set and graph.incidence_coefficients[incidence].item() != 0.0:
            raise AssertionError("zero-count allocation coefficient is nonzero")

    return {
        "passed": True,
        "rows": len(problem.rows),
        "variables": problem.variable_count,
        "zero_rows": len(zero_rows),
        "fixed_support_preserved": True,
        "zero_coefficients_and_features_refreshed": True,
        "execution": "CPU structural characterization; no physical timing claim",
    }


if __name__ == "__main__":
    print(json.dumps(check_zero_template(), sort_keys=True))
