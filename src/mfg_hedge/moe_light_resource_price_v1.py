"""Optional richer terminal prices using detached current GNN resource states."""


def build_resource_price_gnn(*,width=16,price_width=64):
    if type(price_width) is not int or price_width<=0:
        raise ValueError('price_width must be a positive integer')
    import torch
    from torch import nn
    from .moe_light_resource_gnn_v1 import build_light_resource_gnn
    template=build_light_resource_gnn(width=width)

    class ResourcePriceGNN(type(template)):
        def __init__(self):
            super().__init__()
            self.price=nn.Sequential(nn.Linear(width+10,price_width),nn.SiLU(),
                                     nn.Linear(price_width,price_width),nn.SiLU(),
                                     nn.Linear(price_width,1)).double()

        def encode(self,graph):
            if graph is not self.prepared_graph:
                raise ValueError('prepare this graph before inference')
            g=graph
            edge=self.allocation['route'](torch.cat((g.row_features[g.row_index],g.edge_features),-1))
            uniform_relative=self.relative_load(g.resource_features[:,6])
            messages=torch.zeros((g.resource_features.shape[0],self.width),device=g.device,dtype=torch.float64).index_add(
                0,g.incidence_resource,edge[g.incidence_variable]*g.resource_to_variable_weight[:,None])
            resource=self.allocation['resource'](torch.cat((g.resource_features,uniform_relative[:,None],messages),-1))
            returned=torch.zeros_like(edge).index_add(
                0,g.incidence_variable,resource[g.incidence_resource]*g.variable_to_resource_weight[:,None])
            sibling=returned.reshape(-1,2,self.width).mean(1)[g.row_index]
            shares=self.allocation['readout'](torch.cat((edge,returned,sibling),-1)).reshape(-1,2).softmax(-1).reshape(-1)
            loads=torch.cat(g.apply(shares.detach()))
            inputs=torch.cat((g.resource_features,uniform_relative[:,None],
                              (loads/g.f_reference)[:,None],self.relative_load(loads)[:,None],resource),-1).detach()
            return shares,inputs

        def prices_from_features(self,inputs):
            logits=self.price(inputs).squeeze(-1)
            return tuple(logits[a:b].softmax(0) for a,b in zip(self.offsets,self.offsets[1:]))

        def forward(self,graph):
            shares,inputs=self.encode(graph)
            return shares,self.prices_from_features(inputs)

    return ResourcePriceGNN()
