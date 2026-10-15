"""Isolated FP64 device planners; CUDA Graphs are not fused CUDA kernels.

No cloud/dependency side effects. Existing CPU objectives and entry points stay
unchanged. Dense prepared operators target G4/G6; an explicit memory bound
rejects larger layouts instead of silently moving work to the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Sequence

from .moe_communication_social_v2 import (
    BLOCK_NAMES, CommunicationConfig, RouteRow, SparseSocialProblem,
)
from .moe_communication_social_v2_reduction_v1 import reduce_equal_source_masks


@dataclass(frozen=True)
class PreparedLayout:
    original_rows: tuple
    rows: tuple
    row_groups: tuple
    slot_map: tuple
    unit_problem: SparseSocialProblem
    config: CommunicationConfig
    block_names: tuple = BLOCK_NAMES

    @classmethod
    def build(cls, rows, config, *, reduce_rows=True):
        original = SparseSocialProblem.from_rows(tuple(rows), config)
        reduction = reduce_equal_source_masks(original) if reduce_rows else None
        if reduction is None:
            reduced = original
            groups = tuple(range(len(original.rows)))
            maps = tuple(tuple(range(len(r.allowed_gpus))) for r in original.rows)
        else:
            reduced = reduction.problem
            groups = reduction.row_groups
            maps = reduction.original_slot_to_group_slot
        unit_rows = tuple(replace(row, demand=1) for row in reduced.rows)
        unit = SparseSocialProblem.from_rows(unit_rows, config)
        slots = tuple(tuple(unit.row_offsets[group] + slot for slot in mapping)
                      for group, mapping in zip(groups, maps))
        return cls(original.rows, unit_rows, groups, slots, unit, config)

    def validate_demands(self, demands):
        values = tuple(demands)
        if len(values) != len(self.original_rows):
            raise ValueError('demand vector does not match prepared source rows')
        if any(type(n) is not int or n < 0 for n in values) or sum(values) > 2**53 - 1:
            raise ValueError('demands must be nonnegative exact-count integers')
        return values

    def expand_reference(self, shares):
        self.unit_problem.validate_shares(shares)
        return tuple(shares[slot] for row in self.slot_map for slot in row)


@dataclass(frozen=True)
class GpuPlan:
    shares: object
    dual: object
    original_shares: object
    quotas: object
    status: str
    iterations: int
    gap_ms: float | None
    wall_ms: float
    extra: dict


def simplex_projection(values):
    """Euclidean last-axis projection without any host synchronization."""
    import torch
    if values.shape[-1] == 1:
        return torch.ones_like(values)
    ordered = torch.sort(values, dim=-1, descending=True).values
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    candidates = (ordered.cumsum(-1) - 1) / ranks
    count = (ordered > candidates).sum(-1, keepdim=True).clamp_min(1)
    threshold = candidates.gather(-1, count - 1)
    result = (values - threshold).clamp_min(0)
    return result / result.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)


class GpuThreePlannerRuntime:
    """Cached homogeneous five-block model, original-row quotas and GPU OT.

    A runtime is mutable, single-owner and not thread safe. Returned plans own
    their tensors. Graph capture occurs once per requested iteration chunk;
    cold capture is included in wall_ms and reported separately.
    """
    def __init__(self, rows, config, *, device='cuda:0', use_graphs=True,
                 reduce_rows=True, max_dense_bytes=256 * 1024 * 1024):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('only explicit CPU or CUDA execution is supported')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; CPU fallback is forbidden')
        if use_graphs and self.device.type != 'cuda':
            raise ValueError('CUDA Graph mode requires an explicit CUDA device')
        self.use_graphs = use_graphs
        self.layout = PreparedLayout.build(rows, config, reduce_rows=reduce_rows)
        self.groups = len(self.layout.rows)
        self.original_count = len(self.layout.original_rows)
        self.gpus = len(config.gpu_nodes)
        self.width = max(len(row.allowed_gpus) for row in self.layout.original_rows)
        self.offsets = self.layout.unit_problem.row_offsets
        self.variables = self.layout.unit_problem.variable_count
        self.block_widths = tuple(max(1, len(b.resource_names)) for b in self.layout.unit_problem.blocks)
        self.block_offsets = [0]
        for width in self.block_widths:
            self.block_offsets.append(self.block_offsets[-1] + width)
        elements = (self.block_offsets[-1] * self.variables + self.original_count * self.gpus
                    + self.variables * self.original_count * self.width)
        if elements * 8 > max_dense_bytes:
            raise ValueError('prepared dense GPU layout exceeds memory bound; use a separately validated sparse backend')
        self.graphs = {}
        self.graph_capture_ms = 0.0
        self.counts = self.tensor([0] * self.original_count, integer=True)
        aggregate = [[0.] * self.original_count for _ in range(self.groups)]
        for i, group in enumerate(self.layout.row_groups):
            aggregate[group][i] = 1.
        self.aggregate = self.tensor(aggregate)
        self.reduced_counts = self.tensor([0.] * self.groups)
        self.variable_rows = self.tensor(self.layout.unit_problem.variable_rows, integer=True)
        unit = [[0.] * self.variables for _ in range(self.block_offsets[-1])]
        for bi, block in enumerate(self.layout.unit_problem.blocks):
            for column, terms in enumerate(block.columns):
                for resource, coefficient in terms:
                    unit[self.block_offsets[bi] + resource][column] = coefficient
        self.unit = self.tensor(unit)
        self.k = torch.zeros_like(self.unit)
        self.reference = self.tensor(1.)
        self.norm = self.tensor(0.)
        self.step_scale = self.tensor(0.)
        self.uniform = self.tensor([1 / len(r.allowed_gpus) for r in self.layout.rows for _ in r.allowed_gpus])
        self.p = torch.zeros_like(self.uniform)
        self.dual = self.tensor([0.] * self.block_offsets[-1])
        self.p_sum = torch.zeros_like(self.p)
        self.d_sum = torch.zeros_like(self.dual)
        self.trial_p = torch.zeros_like(self.p)
        self.trial_d = torch.zeros_like(self.dual)
        self.iteration_tensor = self.tensor(0.)
        self.primal_groups = {}
        for width in sorted(set(len(r.allowed_gpus) for r in self.layout.rows)):
            indices = [list(range(self.offsets[i], self.offsets[i + 1]))
                       for i, row in enumerate(self.layout.rows) if len(row.allowed_gpus) == width]
            self.primal_groups[width] = self.tensor(indices, integer=True)
        self.original_slots = self.tensor([list(slots) + [0] * (self.width - len(slots))
                                         for slots in self.layout.slot_map], integer=True)
        self.original_valid = self.tensor([[j < len(r.allowed_gpus) for j in range(self.width)]
                                          for r in self.layout.original_rows], boolean=True)
        expansion = [[0.] * (self.original_count * self.width) for _ in range(self.variables)]
        for i, slots in enumerate(self.layout.slot_map):
            for j, slot in enumerate(slots):
                expansion[slot][i * self.width + j] = 1.
        self.slot_aggregate = self.tensor(expansion)
        self.destinations = self.tensor([list(r.allowed_gpus) + [0] * (self.width - len(r.allowed_gpus))
                                        for r in self.layout.original_rows], integer=True)
        self.slot_order = self.tensor(list(range(self.width)), integer=True)
        mask = [[g in r.allowed_gpus for g in range(self.gpus)] for r in self.layout.original_rows]
        self.ot_mask = self.tensor(mask, boolean=True)
        # The cost proxy is linear incidence WORK, not a v2 bottleneck cost.
        unit_cost = self.unit.clone()
        start, end = self.block_offsets[2:4]
        unit_cost[start:end].zero_()
        flat_cost = unit_cost.sum(0)
        self.route_cost = torch.zeros((self.original_count, self.gpus), dtype=torch.float64, device=self.device)
        for i, row in enumerate(self.layout.original_rows):
            self.route_cost[i, list(row.allowed_gpus)] = flat_cost[list(self.layout.slot_map[i])]
        self.log_kernel = torch.zeros_like(self.route_cost)
        self.log_v = self.tensor([0.] * self.gpus)
        self.log_a = self.tensor([0.] * self.original_count)
        self.active_columns = self.tensor([False] * self.gpus, boolean=True)
        self.gauge = torch.zeros((self.gpus, self.gpus), dtype=torch.float64, device=self.device)
        self.ot_done = self.tensor(False, boolean=True)
        self.ot_iterations = self.tensor(0, integer=True)
        self.ot_residual = self.tensor(0.)
        self.ot_tolerance = self.tensor(1e-8)
        self.ot_theta = self.tensor(100 / 101)
        self._states = (self.p, self.dual, self.p_sum, self.d_sum, self.trial_p,
                        self.trial_d, self.iteration_tensor, self.log_v, self.ot_done,
                        self.ot_iterations, self.ot_residual)

    def tensor(self, values, *, integer=False, boolean=False):
        t = self.torch
        return t.tensor(values, device=self.device,
                        dtype=t.bool if boolean else t.int64 if integer else t.float64)

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def set_demands(self, demands):
        values = self.layout.validate_demands(demands)
        self.demands_host = values
        self.counts.copy_(self.tensor(values, integer=True))
        self.reduced_counts.copy_(self.aggregate @ self.counts.double())
        self.k.copy_(self.unit * self.reduced_counts[self.variable_rows].unsqueeze(0))
        uniform_loads = self.apply(self.uniform)
        reference = self.layout.unit_problem.fixed_startup_ms + sum(
            uniform_loads[a:b].max() for a, b in zip(self.block_offsets, self.block_offsets[1:]))
        self.reference.copy_(self.torch.where(reference > 0, reference, self.tensor(1.)))
        self.norm.copy_((self.k.sum(0).max() * self.k.sum(1).max()).sqrt() / self.reference)
        self.step_scale.copy_(self.torch.where(self.norm > 0,
            .5 / self.norm / self.reference, self.tensor(0.)))

    def apply(self, shares):
        return self.k @ shares

    def adjoint(self, dual):
        return self.k.transpose(0, 1) @ dual

    def project_primal(self, values):
        out = self.torch.empty_like(values)
        for width, indices in self.primal_groups.items():
            selected = values[indices]
            if width == 2:
                left = ((selected[:, 0] - selected[:, 1] + 1) * .5).clamp(0, 1)
                projected = self.torch.stack((left, 1 - left), -1)
            else:
                projected = simplex_projection(selected)
            out[indices.reshape(-1)] = projected.reshape(-1)
        return out

    def project_dual(self, values):
        return self.torch.cat([simplex_projection(values[a:b])
            for a, b in zip(self.block_offsets, self.block_offsets[1:])])

    def initialize(self, initial=None):
        t = self.torch
        if initial is None:
            self.p.copy_(self.uniform)
            loads = self.apply(self.p)
            self.dual.zero_()
            for a, b in zip(self.block_offsets, self.block_offsets[1:]):
                self.dual[a:b].scatter_(0, loads[a:b].argmax().reshape(1), self.tensor([1.]))
        else:
            p, dual = initial
            if p.shape != self.p.shape or dual.shape != self.dual.shape:
                raise ValueError('initial primal/dual dimensions do not match layout')
            if p.dtype != t.float64 or dual.dtype != t.float64 or p.device != self.device or dual.device != self.device:
                raise ValueError('initial primal/dual must be FP64 on the planner device')
            invalid = ~t.isfinite(p).all() | ~t.isfinite(dual).all()
            for indices in self.primal_groups.values():
                v = p[indices]
                invalid |= (v < -1e-12).any() | ((v.sum(-1) - 1).abs() > 1e-9).any()
            for a, b in zip(self.block_offsets, self.block_offsets[1:]):
                v = dual[a:b]
                invalid |= (v < -1e-12).any() | ((v.sum() - 1).abs() > 1e-9)
            if invalid.item():
                raise ValueError('initial primal/dual is not finite and feasible')
            self.p.copy_(p); self.dual.copy_(dual)
        self.p_sum.zero_(); self.d_sum.zero_(); self.iteration_tensor.zero_()

    def certificate(self, p, dual):
        loads = self.apply(p)
        upper = self.layout.unit_problem.fixed_startup_ms + sum(
            loads[a:b].max() for a, b in zip(self.block_offsets, self.block_offsets[1:]))
        gradient = self.adjoint(dual)
        lower = self.layout.unit_problem.fixed_startup_ms + sum(
            gradient[indices].min(-1).values.sum() for indices in self.primal_groups.values())
        return upper, lower

    def _mp_chunk(self, count):
        for _ in range(count):
            pt = self.project_primal(self.p - self.step_scale * self.adjoint(self.dual))
            dt = self.project_dual(self.dual + self.step_scale * self.apply(self.p))
            pn = self.project_primal(self.p - self.step_scale * self.adjoint(dt))
            dn = self.project_dual(self.dual + self.step_scale * self.apply(pt))
            self.p_sum.add_(pt); self.d_sum.add_(dt)
            self.trial_p.copy_(pt); self.trial_d.copy_(dt)
            self.p.copy_(pn); self.dual.copy_(dn); self.iteration_tensor.add_(1)

    def _execute(self, kind, count):
        fn = (lambda: self._mp_chunk(count)) if kind == 'mp' else (lambda: self._ot_chunk(count))
        if not self.use_graphs:
            fn(); return
        key = kind, count
        if key not in self.graphs:
            start = time.perf_counter()
            saved = [x.clone() for x in self._states]
            stream = self.torch.cuda.Stream(device=self.device)
            stream.wait_stream(self.torch.cuda.current_stream(self.device))
            with self.torch.cuda.stream(stream), self.torch.no_grad():
                for _ in range(3):
                    fn()
            self.torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = self.torch.cuda.CUDAGraph()
            with self.torch.cuda.graph(graph), self.torch.no_grad():
                fn()
            for target, value in zip(self._states, saved):
                target.copy_(value)
            self.synchronize()
            self.graphs[key] = graph
            self.graph_capture_ms += (time.perf_counter() - start) * 1000
        self.graphs[key].replay()

    def original_shares(self, shares):
        return self.torch.where(self.original_valid, shares[self.original_slots], 0.)

    def round_shares(self, shares, *, original=None):
        t = self.torch
        p = self.original_shares(shares) if original is None else original
        exact = p.clamp_min(0) * self.counts.unsqueeze(-1)
        floors = exact.floor().long()
        remaining = self.counts - floors.sum(-1)
        fractions = t.where(self.original_valid, exact - floors, -t.inf)
        order = t.argsort(fractions, dim=-1, descending=True, stable=True)
        winners = self.slot_order.unsqueeze(0) < remaining.unsqueeze(-1)
        additions = t.zeros_like(floors).scatter(-1, order, winners.long())
        quotas = floors + additions
        bad = (remaining < 0).any() | (remaining > self.original_valid.sum(-1)).any()
        bad |= (quotas.sum(-1) != self.counts).any() | (quotas < 0).any()
        bad |= (quotas.masked_select(~self.original_valid) != 0).any()
        if bad.item():
            raise ArithmeticError('device rounding violated original-row conservation')
        return quotas

    def _finish(self, p, dual, status, iterations, start, extra=None, original=None):
        p = p.clone(); dual = None if dual is None else dual.clone()
        original = self.original_shares(p) if original is None else original.clone()
        invalid = (~self.torch.isfinite(original).all() | (original < -1e-12).any()
                   | ((original.sum(-1) - 1).abs() > 1e-9).any())
        if invalid.item():
            raise ArithmeticError('device shares are not finite and row feasible')
        quotas = self.round_shares(p, original=original)
        gap = None if dual is None else float((lambda v: v[0] - v[1])(self.certificate(p, dual)).item())
        self.synchronize()
        data = dict(extra or {}, reference_ms=self.reference.item(), operator_norm_bound=self.norm.item(),
                    graph_capture_total_ms=self.graph_capture_ms, cuda_graph=self.use_graphs,
                    fused_single_kernel=False, planner_device=str(self.device), planner_dtype='float64')
        return GpuPlan(p, dual, original, quotas, status, iterations, gap,
                       (time.perf_counter() - start) * 1000, data)

    def solve_v2(self, demands, *, steps=36, initial=None, warm_start=None):
        if type(steps) is not int or not 0 <= steps <= 1000000:
            raise ValueError('steps must be an integer in 0..1000000')
        self.synchronize(); start = time.perf_counter()
        with self.torch.no_grad():
            self.set_demands(demands)
            if warm_start is not None:
                if initial is not None:
                    raise ValueError('choose initial or warm_start, not both')
                initial = warm_start.predict(self)
            self.initialize(initial)
            if steps:
                self._execute('mp', steps)
                p, d = self.p_sum / steps, self.d_sum / steps
            else:
                p, d = self.p, self.dual
            return self._finish(p, d, 'fixed_budget', steps, start)

    def solve_high_precision(self, demands, *, max_iterations=100000,
            absolute_gap_ms=1e-5, relative_gap=1e-7, check_interval=50, initial=None):
        if type(max_iterations) is not int or max_iterations <= 0:
            raise ValueError('max_iterations must be positive')
        if type(check_interval) is not int or not 1 <= check_interval <= 1000:
            raise ValueError('check_interval must be in 1..1000')
        if any(not math.isfinite(x) or x < 0 for x in (absolute_gap_ms, relative_gap)):
            raise ValueError('gap tolerances must be finite and nonnegative')
        self.synchronize(); start = time.perf_counter()
        with self.torch.no_grad():
            self.set_demands(demands); self.initialize(initial)
            best_p, best_d = self.p.clone(), self.dual.clone()
            ub, lb = self.certificate(best_p, best_d)
            iterations = 0; status = 'iteration_cap'
            while True:
                u, l = ub.item(), lb.item()
                if not math.isfinite(u - l) or u - l < -1e-10 * max(1., abs(u), abs(l)):
                    status = 'certificate_error'; break
                if u - l <= absolute_gap_ms + relative_gap * max(abs(u), abs(l)):
                    status = 'converged'; break
                if iterations == max_iterations:
                    break
                count = min(check_interval, max_iterations - iterations)
                self._execute('mp', count); iterations += count
                for p, d in ((self.p, self.dual), (self.trial_p, self.trial_d),
                             (self.p_sum / iterations, self.d_sum / iterations)):
                    candidate_u, candidate_l = self.certificate(p, d)
                    improve_u, improve_l = candidate_u < ub, candidate_l > lb
                    best_p = self.torch.where(improve_u, p, best_p)
                    best_d = self.torch.where(improve_l, d, best_d)
                    ub = self.torch.minimum(ub, candidate_u); lb = self.torch.maximum(lb, candidate_l)
            return self._finish(best_p, best_d, status, iterations, start,
                {'absolute_gap_ms': absolute_gap_ms, 'relative_gap': relative_gap,
                 'check_interval': check_interval, 'solver': 'fixed_safe_step_mirror_prox'})

    def _ot_setup(self, epsilon, rho, tolerance):
        t = self.torch
        total = self.counts.sum().double()
        costs = self.route_cost * (total / self.reference)
        positive = self.counts > 0
        costs = t.where(positive[:, None], costs, 0.)
        self.log_kernel.copy_(t.where(self.ot_mask, -costs / epsilon, -t.inf))
        self.log_a.copy_(t.where(positive, self.counts.double().log() - total.clamp_min(1).log(), -t.inf))
        self.active_columns.copy_((self.ot_mask & positive[:, None]).any(0))
        support = (self.ot_mask & positive[:, None]).double()
        adjacency = support.T @ support > 0
        labels = t.arange(self.gpus, device=self.device)
        for _ in range(self.gpus):
            labels = t.where(adjacency, labels[None, :], self.gpus).min(-1).values.minimum(labels)
        same = (labels[:, None] == labels[None, :]) & self.active_columns[:, None] & self.active_columns[None, :]
        self.gauge.copy_(same.double() / same.sum(-1, keepdim=True).clamp_min(1))
        self.log_v.zero_(); self.ot_iterations.zero_(); self.ot_residual.zero_(); self.ot_done.fill_(False)
        self.ot_tolerance.fill_(tolerance); self.ot_theta.fill_(rho / (rho + epsilon))

    def _ot_chunk(self, count):
        t = self.torch
        for _ in range(count):
            u = self.log_a - t.logsumexp(self.log_kernel + self.log_v[None, :], -1)
            column = t.logsumexp(u[:, None] + self.log_kernel, 0)
            proposal = t.where(self.active_columns,
                self.ot_theta * (-math.log(self.gpus) - column), 0.)
            proposal = proposal - self.gauge @ proposal
            residual = (proposal - self.log_v).abs().max()
            self.ot_residual.copy_(t.where(self.ot_done, self.ot_residual, residual))
            self.log_v.copy_(t.where(self.ot_done, self.log_v, proposal))
            self.ot_iterations.add_((~self.ot_done).long())
            self.ot_done.logical_or_(residual <= self.ot_tolerance)

    def solve_sinkhorn(self, demands, *, epsilon=1., rho=100., tolerance=1e-8,
                       max_iterations=2000, check_interval=32):
        if any(not math.isfinite(x) or x <= 0 for x in (epsilon, rho, tolerance)):
            raise ValueError('epsilon, rho and tolerance must be positive')
        if type(max_iterations) is not int or max_iterations <= 0 or type(check_interval) is not int or check_interval <= 0:
            raise ValueError('positive integer iteration bounds required')
        self.synchronize(); start = time.perf_counter()
        with self.torch.no_grad():
            self.set_demands(demands); self._ot_setup(epsilon, rho, tolerance)
            executed = 0
            if sum(self.demands_host):
                while executed < max_iterations:
                    count = min(check_interval, max_iterations - executed)
                    self._execute('ot', count); executed += count
                    if self.ot_done.item():
                        break
            else:
                self.ot_done.fill_(True)
            logs = self.log_kernel + self.log_v[None, :]
            log_share = logs - self.torch.logsumexp(logs, -1, keepdim=True)
            dense = log_share.exp()
            dense = self.torch.where(self.counts[:, None] > 0, dense,
                self.ot_mask.double() / self.ot_mask.sum(-1, keepdim=True))
            dense = dense / dense.sum(-1, keepdim=True)
            original = dense.gather(-1, self.destinations) * self.original_valid
            # Collapse is only for scoring the common v2 model; never optimize
            # OT after reduction, which would change its entropy objective.
            weights = original * self.counts[:, None]
            numerator = self.slot_aggregate @ weights.reshape(-1)
            denominator = self.reduced_counts[self.variable_rows]
            p = self.torch.where(denominator > 0, numerator / denominator.clamp_min(1), self.uniform)
            mass_logs = self.log_a[:, None] + log_share
            log_load = self.torch.logsumexp(mass_logs, 0)
            score = (-epsilon * self.log_kernel + epsilon * mass_logs +
                     rho * (log_load[None, :] + math.log(self.gpus)))
            high = self.torch.where(self.ot_mask, score, -self.torch.inf).max(-1).values
            low = self.torch.where(self.ot_mask, score, self.torch.inf).min(-1).values
            kkt = self.torch.where(self.counts > 0, high - low, 0.).max().item()
            return self._finish(p, None, 'converged' if self.ot_done.item() else 'iteration_cap',
                int(self.ot_iterations.item()), start, original=original,
                extra={'epsilon': epsilon, 'rho': rho, 'tolerance': tolerance,
                    'kkt_residual': kkt, 'update_residual': self.ot_residual.item(),
                    'executed_rounds_including_frozen_padding': executed,
                    'target': 'uniform', 'row_constraint': 'hard', 'column_constraint': 'KL soft'})

    def audit(self, shares, dual=None, quotas=None):
        """Independent stdlib replay; caller reports audit separately from C."""
        from .moe_communication_social_v2 import integerize_rows
        original_rows = tuple(replace(row, demand=n) for row, n in zip(self.layout.original_rows, self.demands_host))
        problem = SparseSocialProblem.from_rows(original_rows, self.layout.config)
        vector = self.layout.expand_reference(shares.detach().cpu().tolist())
        problem.validate_shares(vector)
        result = {'feasible': True, 'upper_bound_ms': problem.objective_value(vector)}
        if dual is not None:
            flat = dual.detach().cpu().tolist()
            prices = [0.] * self.variables
            for bi, block in enumerate(self.layout.unit_problem.blocks):
                values = flat[self.block_offsets[bi]:self.block_offsets[bi+1]]
                if any(not math.isfinite(x) or x < -1e-12 for x in values) or abs(sum(values) - 1) > 1e-9:
                    raise ArithmeticError('independent dual feasibility audit failed')
                for column, incidences in enumerate(block.columns):
                    prices[column] += math.fsum(c * values[r] for r, c in incidences)
            # Retain zero-load resources from the static layout: deleting and
            # redistributing their dual mass could produce an invalid bound.
            result['lower_bound_ms'] = problem.fixed_startup_ms + math.fsum(
                row.demand * min(prices[slot] for slot in mapping)
                for row, mapping in zip(original_rows, self.layout.slot_map))
            result['gap_ms'] = result['upper_bound_ms'] - result['lower_bound_ms']
        if quotas is not None:
            values = quotas.detach().cpu().tolist()
            for row, q in zip(original_rows, values):
                if sum(q) != row.demand or any(x < 0 for x in q) or any(q[len(row.allowed_gpus):]):
                    raise ArithmeticError('independent original-row quota audit failed')
            qshares = tuple(q[j] / row.demand if row.demand else 1 / len(row.allowed_gpus)
                for row, q in zip(original_rows, values) for j in range(len(row.allowed_gpus)))
            result['integer_F_ms'] = problem.objective_value(qshares)
        return result
