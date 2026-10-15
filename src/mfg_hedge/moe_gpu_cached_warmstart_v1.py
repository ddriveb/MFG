"""Opt-in cached G256 FP64 graph features and GPU neural warm start.

Fixed positive-demand row identities only. Numeric features are refreshed for
current counts; graph.problem retains immutable setup metadata and must not be
used to independently score a later batch. Use a fresh problem for that audit.
Torch remains lazy on ordinary import. Defaults/frozen checkpoints unchanged.
"""
from .moe_gpu_sparse_planner_v2 import SparseGpuPlanner


class CachedGpuWarmStart:
    def __init__(self, problem, model, *, device='cuda:0', projection='active'):
        if any(row.demand <= 0 for row in problem.rows):
            raise ValueError('cached first slice requires all setup row demands positive')
        import torch
        from .moe_communication_social_v2_g256_sparse_warmstart_v1 import SparseSocialTorchGraph
        self.torch = torch
        self.runtime = SparseGpuPlanner(problem, device=device, projection=projection)
        self.graph = SparseSocialTorchGraph.from_problem(problem, device=device)
        self.model = model
        param = next(model.parameters())
        if param.device != self.runtime.device or param.dtype != torch.float64:
            raise ValueError('model must already reside on the selected device in FP64')
        self.model.eval()
        g = self.graph
        if not torch.equal(g.incidence_resource, self.runtime.ir) or not torch.equal(g.incidence_variable, self.runtime.iv):
            raise ValueError('network and solver resource indexing differ')
        if g.block_resource_offsets != tuple(self.runtime.block_offsets):
            raise ValueError('network and solver block indexing differ')
        self.block_ids = torch.cat([
            torch.full((b-a,), i, dtype=torch.int64, device=self.runtime.device)
            for i, (a,b) in enumerate(zip(g.block_incidence_offsets, g.block_incidence_offsets[1:]))
        ])
        self.mass_slots = g.incidence_variable * 5 + self.block_ids
        self.row_width = g.row_slot_mask.sum(-1).to(torch.float64)
        self.resource_total = torch.zeros_like(self.runtime.dual)
        self.variable_total = torch.zeros_like(self.runtime.p)
        self.variable_mass = torch.zeros((g.variable_count,5), dtype=torch.float64, device=self.runtime.device)
        self.row_mass = torch.zeros(g.row_count, dtype=torch.float64, device=self.runtime.device)

    def update_features(self, demands):
        """Update numeric buffers without reconstructing topology or Python rows."""
        t = self.torch
        if isinstance(demands, t.Tensor):
            if demands.dtype != t.int64 or demands.device != self.runtime.device or demands.shape != self.runtime.counts.shape:
                raise ValueError('demands must be a matching resident int64 vector')
            if bool((demands <= 0).any().item()):
                raise ValueError('positive demands required; zero/active-row rebinding unsupported')
        elif len(demands) != self.graph.row_count or any(type(x) is not int or x <= 0 for x in demands):
            raise ValueError('matching positive integer demands required')
        self.runtime.set_demands(demands)
        return self._refresh_features()

    def _refresh_features(self):
        """Refresh from validated runtime counts; capture-safe device operations."""
        t = self.torch
        g = self.graph
        counts = self.runtime.counts.to(t.float64)
        coefficient = self.runtime.coefficient
        reference = self.runtime.normalization
        g.incidence_coefficients.copy_(coefficient)
        self.resource_total.zero_().scatter_add_(0, g.incidence_resource, coefficient)
        self.variable_total.zero_().scatter_add_(0, g.incidence_variable, coefficient)
        self.variable_mass.zero_().view(-1).scatter_add_(0, self.mass_slots, coefficient / reference)
        self.row_mass.zero_().scatter_add_(0, g.row_index, self.variable_mass.sum(-1))
        g.row_features[:,0].copy_(counts.log1p() / counts.max().log1p().clamp_min(1.))
        g.row_features[:,1].copy_(counts / counts.mean().clamp_min(1.))
        g.row_features[:,6].copy_(self.row_mass / self.row_width)
        g.edge_features[:,7:].copy_(self.variable_mass)
        g.resource_features[:,6].copy_(self.resource_total / reference)
        rt = self.resource_total[g.incidence_resource]
        vt = self.variable_total[g.incidence_variable]
        # Positive row restriction preserves setup resource/variable incidence.
        g.resource_to_variable_weight.copy_(t.where(rt > 0, coefficient / rt.clamp_min(1e-300), 0.))
        g.variable_to_resource_weight.copy_(t.where(vt > 0, coefficient / vt.clamp_min(1e-300), 0.))
        g.f_reference.copy_(reference)
        g.step_scale.copy_(self.runtime.initial_gamma / reference)
        return g

    def predict(self):
        """Fresh prediction from current features; never reuse prior outputs."""
        with self.torch.no_grad():
            primal, dual_blocks = self.model(self.graph)
            return primal, self.torch.cat(dual_blocks)

    def solve(self, demands, **solver_options):
        """Same high-precision solver and quality threshold after prediction.

        The existing solver currently refreshes demand coefficients again at its
        entry. This redundant update is included in timings, not hidden.
        Returned solver complete_C_ms excludes this adapter's feature/network
        work; callers must time this entire method for complete warm-start C.
        """
        self.update_features(demands)
        initial = self.predict()
        return self.runtime.solve_high_precision(self.runtime.counts, initial=initial, **solver_options)
