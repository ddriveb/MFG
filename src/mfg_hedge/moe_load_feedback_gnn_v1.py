"""Optional fixed-R2 GNN with shared differentiable load-feedback rounds.

Price features are detached; the allocation objective trains both allocation
rounds. No solver iterations, reference labels or production default changes.
"""


def build_load_feedback_gnn(*, width=16, price_width=64,
                            feedback_rounds=1, pressure_features=False):
    if type(feedback_rounds) is not int or feedback_rounds < 1:
        raise ValueError('feedback_rounds must be a positive integer')
    if type(pressure_features) is not bool:
        raise ValueError('pressure_features must be boolean')
    import torch
    from torch import nn
    from .moe_light_resource_price_v1 import build_resource_price_gnn
    template = build_resource_price_gnn(width=width, price_width=price_width)

    class LoadFeedbackGNN(type(template)):
        def __init__(self):
            super().__init__()
            self.allocation['feedback_resource'] = nn.Sequential(
                nn.Linear(width + 3, width), nn.SiLU()).double()
            self.allocation['feedback_readout'] = nn.Sequential(
                nn.Linear(3 * width + 1 + 2 * int(pressure_features), width), nn.SiLU(),
                nn.Linear(width, 1)).double()
            nn.init.zeros_(self.allocation['feedback_readout'][-1].weight)
            nn.init.zeros_(self.allocation['feedback_readout'][-1].bias)

        def allocation_with_context(self, g):
            if g is not self.prepared_graph:
                raise ValueError('prepare this graph before inference')
            edge = self.allocation['route'](torch.cat(
                (g.row_features[g.row_index], g.edge_features), -1))
            relative = self.relative_load(g.resource_features[:, 6])
            messages = edge.new_zeros((g.resource_features.shape[0], self.width)).index_add(
                0, g.incidence_resource,
                edge[g.incidence_variable] * g.resource_to_variable_weight[:, None])
            resource = self.allocation['resource'](torch.cat(
                (g.resource_features, relative[:, None], messages), -1))

            def return_messages(state):
                returned = torch.zeros_like(edge).index_add(
                    0, g.incidence_variable,
                    state[g.incidence_resource] * g.variable_to_resource_weight[:, None])
                sibling = returned.reshape(-1, 2, self.width).mean(1)[g.row_index]
                return returned, sibling

            returned, sibling = return_messages(resource)
            logits = self.allocation['readout'](torch.cat((edge, returned, sibling), -1))
            # Resource features contain total incident demand, twice uniform load for R2.
            uniform = g.resource_features[:, 6] / 2
            for _ in range(feedback_rounds):
                initial = logits.reshape(-1, 2).softmax(-1).reshape(-1)
                blocks = g.apply(initial)
                loads = torch.cat(blocks)
                normalized = loads / g.f_reference
                resource = self.allocation['feedback_resource'](torch.cat((
                    resource, normalized[:, None], self.relative_load(loads)[:, None],
                    (normalized - uniform)[:, None]), -1))
                returned, sibling = return_messages(resource)
                inputs = [edge, returned, sibling, initial[:, None]]
                if pressure_features:
                    pressures = tuple((v / (.05 * v.max().clamp_min(1e-12))).softmax(0)
                                      for v in blocks)
                    cost = g.adjoint(pressures) / g.f_reference
                    scale = cost.reshape(-1, 2).mean(1).clamp_min(1e-12)
                    relative_cost = cost / scale[g.row_index]
                    difference = relative_cost - relative_cost.reshape(-1, 2).flip(1).reshape(-1)
                    inputs.extend((relative_cost[:, None], difference[:, None]))
                logits = logits + self.allocation['feedback_readout'](torch.cat(inputs, -1))
            shares = logits.reshape(-1, 2).softmax(-1).reshape(-1)
            return shares, resource

        def allocation_only(self, graph):
            return self.allocation_with_context(graph)[0]

        def encode(self, graph):
            shares, resource = self.allocation_with_context(graph)
            loads = torch.cat(graph.apply(shares.detach()))
            inputs = torch.cat((graph.resource_features,
                                self.relative_load(graph.resource_features[:, 6])[:, None],
                                (loads / graph.f_reference)[:, None],
                                self.relative_load(loads)[:, None], resource), -1).detach()
            return shares, inputs

    return LoadFeedbackGNN()
