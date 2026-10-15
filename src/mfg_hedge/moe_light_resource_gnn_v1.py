"""FP64 one-round resource GNN for fixed R2, with isolated terminal prices.

All dynamic features and incidence weights are read at each forward. Static
topology is prepared once. No initializer GNN or iterative correction.
"""


def build_light_resource_gnn(*,width=16):
    if type(width) is not int or width<=0:
        raise ValueError('width must be a positive integer')
    import torch
    from torch import nn

    class LightResourceGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.width=width
            self.allocation=nn.ModuleDict({
                'route':nn.Sequential(nn.Linear(19,width),nn.SiLU()),
                'resource':nn.Sequential(nn.Linear(width+8,width),nn.SiLU()),
                'readout':nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,1)),
            })
            self.price=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.Linear(width,1))
            self.double();self.prepared_graph=None

        def prepare(self,graph):
            if any(b-a!=2 for a,b in zip(graph.problem.row_offsets,graph.problem.row_offsets[1:])):
                raise ValueError('fixed two replicas per row required')
            if next(self.parameters()).device!=graph.device or next(self.parameters()).dtype!=torch.float64:
                raise ValueError('graph/model require same device and FP64')
            self.offsets=tuple(graph.block_resource_offsets)
            self.prepared_graph=graph
            return self

        def relative_load(self,load):
            return torch.cat(tuple(load[a:b]/load[a:b].max().clamp_min(1e-12) for a,b in zip(self.offsets,self.offsets[1:])))

        def forward(self,graph):
            if graph is not self.prepared_graph:
                raise ValueError('prepare this graph before inference')
            g=graph
            inputs=torch.cat((g.row_features[g.row_index],g.edge_features),-1)
            edge=self.allocation['route'](inputs)
            uniform_relative=self.relative_load(g.resource_features[:,6])
            messages=torch.zeros((g.resource_features.shape[0],self.width),device=g.device,dtype=torch.float64).index_add(
                0,g.incidence_resource,edge[g.incidence_variable]*g.resource_to_variable_weight[:,None])
            resource=self.allocation['resource'](torch.cat((g.resource_features,uniform_relative[:,None],messages),-1))
            returned=torch.zeros_like(edge).index_add(
                0,g.incidence_variable,resource[g.incidence_resource]*g.variable_to_resource_weight[:,None])
            sibling=returned.reshape(-1,2,self.width).mean(1)[g.row_index]
            logits=self.allocation['readout'](torch.cat((edge,returned,sibling),-1)).reshape(-1,2)
            shares=logits.softmax(-1).reshape(-1)
            # Price optimization must not backpropagate into allocation.
            loads=torch.cat(g.apply(shares.detach()))
            price_inputs=torch.cat((g.resource_features,uniform_relative[:,None],
                                    (loads/g.f_reference)[:,None],self.relative_load(loads)[:,None]),-1).detach()
            price_logits=self.price(price_inputs).squeeze(-1)
            dual=tuple(price_logits[a:b].softmax(0) for a,b in zip(self.offsets,self.offsets[1:]))
            return shares,dual

    return LightResourceGNN()
