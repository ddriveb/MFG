"""Deterministic baselines for routing among fixed same-Expert replicas.

The module deliberately separates placement, single-execution routing, and
request hedging.  It contains independent standard-library adaptations for
simulation; it does not vendor CUDA, P4, network, or PyTorch runtime code.

Upstream semantics and licenses are recorded in ``THIRD_PARTY_NOTICES.md`` and
``docs/adr/0030-replica-routing-baseline-taxonomy.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence


def _true_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


@dataclass(frozen=True)
class BaselineProvenance:
    baseline_id: str
    semantic_class: str
    source_url: str
    source_revision: str
    adaptation_kind: str
    notes: str


_TAIL_AT_SCALE = "https://research.google/pubs/the-tail-at-scale/"
_LAEDGE = "https://www.usenix.org/conference/nsdi21/presentation/primorac"
_EPLB = "https://github.com/deepseek-ai/EPLB"
_LPLB = "https://github.com/deepseek-ai/LPLB"


BASELINE_CATALOG = {
    "failover_only": BaselineProvenance(
        "failover_only", "hedging", _TAIL_AT_SCALE, "CACM-2013",
        "native", "One execution until failure; Replay is reactive, not speculative.",
    ),
    "uniform_round_robin": BaselineProvenance(
        "uniform_round_robin", "routing", _LPLB, "main@0490f79452f7ef277e814449600b1b1dd4c663b3",
        "native", "Counter-free deterministic uniform assignment among eligible replicas.",
    ),
    "join_shortest_queue": BaselineProvenance(
        "join_shortest_queue", "routing", _LAEDGE, "NSDI-2021",
        "native", "Choose the available replica with minimum observable queue depth.",
    ),
    "least_unfinished_work": BaselineProvenance(
        "least_unfinished_work", "routing", _LAEDGE, "NSDI-2021",
        "native", "Choose the available replica with minimum observable unfinished work.",
    ),
    "eplb_style": BaselineProvenance(
        "eplb_style", "placement", _EPLB, "main@retrieved-2026-09-08",
        "independent_stdlib_adaptation",
        "DeepSeek replication plus equal-cardinality greedy hierarchical packing.",
    ),
    "lplb_token_count_minmax": BaselineProvenance(
        "lplb_token_count_minmax", "routing", _LPLB,
        "main@0490f79452f7ef277e814449600b1b1dd4c663b3",
        "independent_stdlib_adaptation",
        "Exact integral min-max token-count assignment; no CUDA/IPM timing model.",
    ),
    "p95_delayed_hedge": BaselineProvenance(
        "p95_delayed_hedge", "hedging", _TAIL_AT_SCALE, "CACM-2013",
        "independent_stdlib_adaptation", "Frozen empirical P95 timer.",
    ),
    "laedge_work_conserving": BaselineProvenance(
        "laedge_work_conserving", "hedging", _LAEDGE, "NSDI-2021",
        "independent_stdlib_adaptation",
        "Generalized LÆDGE ordering with deterministic replica tie-breaking.",
    ),
}


@dataclass(frozen=True)
class ReplicaLoad:
    replica_id: int
    queue_depth: int
    unfinished_work: float
    available: bool = True

    def __post_init__(self) -> None:
        _true_int(self.replica_id, "replica_id")
        _true_int(self.queue_depth, "queue_depth")
        _finite_nonnegative(self.unfinished_work, "unfinished_work")
        if type(self.available) is not bool:
            raise ValueError("available must be bool")


def _available(replicas: Sequence[ReplicaLoad]) -> tuple[ReplicaLoad, ...]:
    if not isinstance(replicas, Sequence) or isinstance(replicas, (str, bytes)):
        raise ValueError("replicas must be a sequence")
    rows = tuple(replicas)
    if not rows or any(not isinstance(row, ReplicaLoad) for row in rows):
        raise ValueError("replicas must contain ReplicaLoad values")
    ids = [row.replica_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("replica IDs must be unique")
    live = tuple(sorted((row for row in rows if row.available), key=lambda row: row.replica_id))
    if not live:
        raise ValueError("at least one replica must be available")
    return live


class RoundRobinRouter:
    baseline_id = "uniform_round_robin"

    def choose(self, token_id: int, replicas: Sequence[ReplicaLoad]) -> int:
        token_id = _true_int(token_id, "token_id")
        live = _available(replicas)
        return live[token_id % len(live)].replica_id


class JoinShortestQueueRouter:
    baseline_id = "join_shortest_queue"

    def choose(self, token_id: int, replicas: Sequence[ReplicaLoad]) -> int:
        _true_int(token_id, "token_id")
        return min(_available(replicas), key=lambda row: (row.queue_depth, row.replica_id)).replica_id


class LeastUnfinishedWorkRouter:
    baseline_id = "least_unfinished_work"

    def choose(self, token_id: int, replicas: Sequence[ReplicaLoad]) -> int:
        _true_int(token_id, "token_id")
        return min(
            _available(replicas),
            key=lambda row: (row.unfinished_work, row.queue_depth, row.replica_id),
        ).replica_id


def _weights(values: Sequence[object], name: str = "loads") -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence")
    return tuple(_finite_nonnegative(value, f"{name}[{index}]") for index, value in enumerate(values))


def _balanced_pack(weights: Sequence[object], num_packs: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pure-Python adaptation of DeepSeek EPLB ``balanced_packing``."""

    rows = _weights(weights, "weights")
    num_packs = _true_int(num_packs, "num_packs", minimum=1)
    if len(rows) % num_packs:
        raise ValueError("item count must be divisible by num_packs")
    per_pack = len(rows) // num_packs
    if per_pack == 1:
        return tuple(range(len(rows))), (0,) * len(rows)
    pack_index = [-1] * len(rows)
    rank_in_pack = [-1] * len(rows)
    pack_weights = [0.0] * num_packs
    pack_items = [0] * num_packs
    for item in sorted(range(len(rows)), key=lambda index: (-rows[index], index)):
        pack = min(
            (candidate for candidate in range(num_packs) if pack_items[candidate] < per_pack),
            key=lambda candidate: (pack_weights[candidate], candidate),
        )
        pack_index[item] = pack
        rank_in_pack[item] = pack_items[pack]
        pack_weights[pack] += rows[item]
        pack_items[pack] += 1
    return tuple(pack_index), tuple(rank_in_pack)


def replicate_experts_eplb(
    loads: Sequence[object], num_physical_experts: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Replicate the largest current ``load / replica_count`` Expert."""

    values = _weights(loads)
    num_physical_experts = _true_int(
        num_physical_experts, "num_physical_experts", minimum=len(values)
    )
    physical_to_logical = list(range(len(values)))
    ranks = [0] * len(values)
    counts = [1] * len(values)
    while len(physical_to_logical) < num_physical_experts:
        logical = min(
            range(len(values)),
            key=lambda index: (-(values[index] / counts[index]), index),
        )
        physical_to_logical.append(logical)
        ranks.append(counts[logical])
        counts[logical] += 1
    return tuple(physical_to_logical), tuple(ranks), tuple(counts)


def _inverse(permutation: Sequence[int]) -> tuple[int, ...]:
    result = [-1] * len(permutation)
    for source, destination in enumerate(permutation):
        result[destination] = source
    if any(value < 0 for value in result):
        raise AssertionError("invalid permutation")
    return tuple(result)


@dataclass(frozen=True)
class EPLBPlan:
    physical_to_logical: tuple[tuple[int, ...], ...]
    logical_to_physical: tuple[tuple[tuple[int, ...], ...], ...]
    logical_counts: tuple[tuple[int, ...], ...]
    mode: str


def _hierarchical_layer(
    weights: tuple[float, ...], num_physical: int, num_groups: int,
    num_nodes: int, num_gpus: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    num_logical = len(weights)
    group_size = num_logical // num_groups
    groups_per_node = num_groups // num_nodes
    physical_per_node = num_physical // num_nodes
    logical_per_node = num_logical // num_nodes
    gpus_per_node = num_gpus // num_nodes
    physical_per_gpu = num_physical // num_gpus

    group_weights = tuple(
        math.fsum(weights[group * group_size:(group + 1) * group_size])
        for group in range(num_groups)
    )
    group_pack, group_rank = _balanced_pack(group_weights, num_nodes)
    logical_to_middle = tuple(
        (group_pack[logical // group_size] * groups_per_node
         + group_rank[logical // group_size]) * group_size
        + logical % group_size
        for logical in range(num_logical)
    )
    middle_to_logical = _inverse(logical_to_middle)
    middle_weights = tuple(weights[index] for index in middle_to_logical)

    final_physical_to_logical: list[int] = []
    final_ranks: list[int] = []
    middle_counts_all: list[int] = []
    for node in range(num_nodes):
        start = node * logical_per_node
        local_weights = middle_weights[start:start + logical_per_node]
        local_phy, local_rank, local_counts = replicate_experts_eplb(
            local_weights, physical_per_node
        )
        physical_weights = tuple(
            local_weights[logical] / local_counts[logical] for logical in local_phy
        )
        pack, rank_in_pack = _balanced_pack(physical_weights, gpus_per_node)
        generated_to_position = tuple(
            pack[index] * physical_per_gpu + rank_in_pack[index]
            for index in range(physical_per_node)
        )
        position_to_generated = _inverse(generated_to_position)
        for generated in position_to_generated:
            middle = start + local_phy[generated]
            final_physical_to_logical.append(middle_to_logical[middle])
            final_ranks.append(local_rank[generated])
        middle_counts_all.extend(local_counts)

    logical_counts = tuple(middle_counts_all[logical_to_middle[i]] for i in range(num_logical))
    return tuple(final_physical_to_logical), tuple(final_ranks), logical_counts


def rebalance_experts_eplb(
    loads_by_layer: Sequence[Sequence[object]],
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> EPLBPlan:
    """Adapt DeepSeek EPLB placement without importing PyTorch.

    The global mode follows upstream by invoking the hierarchical algorithm as
    one group on one node when ``num_groups`` is not divisible by ``num_nodes``.
    """

    if not isinstance(loads_by_layer, Sequence) or not loads_by_layer:
        raise ValueError("loads_by_layer must be non-empty")
    layers = tuple(_weights(row, "layer") for row in loads_by_layer)
    width = len(layers[0])
    if any(len(row) != width for row in layers):
        raise ValueError("all layers must have the same logical Expert count")
    num_physical_experts = _true_int(
        num_physical_experts, "num_physical_experts", minimum=width
    )
    num_groups = _true_int(num_groups, "num_groups", minimum=1)
    num_nodes = _true_int(num_nodes, "num_nodes", minimum=1)
    num_gpus = _true_int(num_gpus, "num_gpus", minimum=1)
    if width % num_groups:
        raise ValueError("logical Expert count must be divisible by num_groups")
    if num_gpus % num_nodes:
        raise ValueError("num_gpus must be divisible by num_nodes")
    if num_physical_experts % num_gpus:
        raise ValueError("physical Expert count must be divisible by num_gpus")

    hierarchical = num_groups % num_nodes == 0
    effective = (num_groups, num_nodes) if hierarchical else (1, 1)
    rows = tuple(
        _hierarchical_layer(row, num_physical_experts, effective[0], effective[1], num_gpus)
        for row in layers
    )
    physical_to_logical = tuple(row[0] for row in rows)
    ranks = tuple(row[1] for row in rows)
    counts = tuple(row[2] for row in rows)
    inverse_rows = []
    for phy2log, rank, layer_counts in zip(physical_to_logical, ranks, counts):
        width_rank = max(layer_counts)
        inverse = [[-1] * width_rank for _ in range(width)]
        for physical, (logical, replica_rank) in enumerate(zip(phy2log, rank)):
            inverse[logical][replica_rank] = physical
        inverse_rows.append(tuple(tuple(row) for row in inverse))
    return EPLBPlan(
        physical_to_logical=physical_to_logical,
        logical_to_physical=tuple(inverse_rows),
        logical_counts=counts,
        mode="hierarchical" if hierarchical else "global",
    )


@dataclass(frozen=True)
class LPLBAssignment:
    token_to_replica: tuple[int, ...]
    final_loads: tuple[tuple[int, int], ...]
    maximum_load: int


class _FlowEdge:
    __slots__ = ("to", "reverse", "capacity", "original")

    def __init__(self, to: int, reverse: int, capacity: int):
        self.to = to
        self.reverse = reverse
        self.capacity = capacity
        self.original = capacity


def _add_edge(graph: list[list[_FlowEdge]], source: int, target: int, capacity: int) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity)
    reverse = _FlowEdge(source, len(graph[source]), 0)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _flow_counts(
    counts: Mapping[int, int], eligible: Mapping[int, tuple[int, ...]],
    initial: Mapping[int, int], cap: int,
) -> dict[tuple[int, int], int] | None:
    experts = tuple(sorted(counts))
    replicas = tuple(sorted(initial))
    source = 0
    expert_node = {expert: 1 + index for index, expert in enumerate(experts)}
    replica_node = {
        replica: 1 + len(experts) + index for index, replica in enumerate(replicas)
    }
    sink = 1 + len(experts) + len(replicas)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    tracked: dict[tuple[int, int], _FlowEdge] = {}
    for expert in experts:
        _add_edge(graph, source, expert_node[expert], counts[expert])
        for replica in eligible[expert]:
            tracked[(expert, replica)] = _add_edge(
                graph, expert_node[expert], replica_node[replica], counts[expert]
            )
    for replica in replicas:
        residual = cap - initial[replica]
        if residual < 0:
            return None
        _add_edge(graph, replica_node[replica], sink, residual)

    total_flow = 0
    needed = sum(counts.values())
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for edge in graph[node]:
                if edge.capacity > 0 and levels[edge.to] < 0:
                    levels[edge.to] = levels[node] + 1
                    queue.append(edge.to)
        if levels[sink] < 0:
            break
        cursor = [0] * len(graph)

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                if edge.capacity > 0 and levels[edge.to] == levels[node] + 1:
                    pushed = send(edge.to, min(amount, edge.capacity))
                    if pushed:
                        edge.capacity -= pushed
                        graph[edge.to][edge.reverse].capacity += pushed
                        return pushed
                cursor[node] += 1
            return 0

        while True:
            pushed = send(source, needed - total_flow)
            if not pushed:
                break
            total_flow += pushed
        if total_flow == needed:
            break
    if total_flow != needed:
        return None
    return {key: edge.original - edge.capacity for key, edge in tracked.items()}


def lplb_token_count_minmax(
    token_experts: Sequence[int],
    eligible_replicas: Mapping[int, Sequence[int]],
    initial_loads: Mapping[int, int] | None = None,
) -> LPLBAssignment:
    """Minimize the maximum integral token count over eligible replicas.

    This is a deterministic simulation reference for LPLB's batch token-count
    objective, solved as a capacitated bipartite flow.  It is not the upstream
    CUDA interior-point implementation and does not model GPU communication.
    """

    if not isinstance(token_experts, Sequence) or isinstance(token_experts, (str, bytes)):
        raise ValueError("token_experts must be a sequence")
    tokens = tuple(_true_int(value, "expert_id") for value in token_experts)
    if not isinstance(eligible_replicas, Mapping) or not eligible_replicas:
        raise ValueError("eligible_replicas must be a non-empty mapping")
    eligible: dict[int, tuple[int, ...]] = {}
    for expert, replica_values in eligible_replicas.items():
        expert = _true_int(expert, "expert_id")
        if not isinstance(replica_values, Sequence) or isinstance(replica_values, (str, bytes)):
            raise ValueError("eligible replica values must be sequences")
        replicas = tuple(sorted({_true_int(value, "replica_id") for value in replica_values}))
        if not replicas:
            raise ValueError("each Expert needs at least one eligible replica")
        eligible[expert] = replicas
    missing = sorted(set(tokens) - set(eligible))
    if missing:
        raise ValueError(f"missing eligible replicas for Experts {missing}")

    initial: dict[int, int] = {
        replica: 0 for replicas in eligible.values() for replica in replicas
    }
    if initial_loads is not None:
        if not isinstance(initial_loads, Mapping):
            raise ValueError("initial_loads must be a mapping")
        for replica, load in initial_loads.items():
            replica = _true_int(replica, "replica_id")
            if replica not in initial:
                raise ValueError("initial_loads contains an ineligible replica")
            initial[replica] = _true_int(load, "initial_load")
    if not tokens:
        rows = tuple(sorted(initial.items()))
        return LPLBAssignment((), rows, max((load for _, load in rows), default=0))

    counts = {expert: tokens.count(expert) for expert in sorted(set(tokens))}
    lower = max(
        max(initial.values(), default=0),
        math.ceil((sum(initial.values()) + len(tokens)) / len(initial)),
    )
    upper = max(initial.values(), default=0) + len(tokens)
    while lower < upper:
        middle = (lower + upper) // 2
        if _flow_counts(counts, eligible, initial, middle) is None:
            lower = middle + 1
        else:
            upper = middle
    flows = _flow_counts(counts, eligible, initial, lower)
    if flows is None:
        raise AssertionError("minimum feasible LPLB cap disappeared")

    remaining = dict(flows)
    assignments = []
    for expert in tokens:
        replica = next(
            replica for replica in eligible[expert]
            if remaining[(expert, replica)] > 0
        )
        remaining[(expert, replica)] -= 1
        assignments.append(replica)
    final = dict(initial)
    for replica in assignments:
        final[replica] += 1
    return LPLBAssignment(tuple(assignments), tuple(sorted(final.items())), lower)


class FailoverOnly:
    baseline_id = "failover_only"

    @staticmethod
    def launch_speculative(*, primary_complete: bool) -> bool:
        if type(primary_complete) is not bool:
            raise ValueError("primary_complete must be bool")
        return False

    @staticmethod
    def launch_replay(*, primary_failed: bool, token_complete: bool) -> bool:
        if type(primary_failed) is not bool or type(token_complete) is not bool:
            raise ValueError("failure and completion flags must be bool")
        return primary_failed and not token_complete


@dataclass(frozen=True)
class P95DelayedHedge:
    delay: float
    baseline_id: str = "p95_delayed_hedge"

    def __post_init__(self) -> None:
        if _finite_nonnegative(self.delay, "delay") <= 0.0:
            raise ValueError("delay must be positive")

    @classmethod
    def from_latency_samples(cls, samples: Sequence[object]) -> "P95DelayedHedge":
        values = sorted(_weights(samples, "latency_samples"))
        if any(value <= 0.0 for value in values):
            raise ValueError("latency samples must be positive")
        rank = (len(values) - 1) * 0.95
        lower = math.floor(rank)
        upper = math.ceil(rank)
        fraction = rank - lower
        return cls(values[lower] + fraction * (values[upper] - values[lower]))

    def timer_at(self, arrival_time: object) -> float:
        return _finite_nonnegative(arrival_time, "arrival_time") + self.delay

    @staticmethod
    def launch_at_timer(*, primary_complete: bool, backup_available: bool) -> bool:
        if type(primary_complete) is not bool or type(backup_available) is not bool:
            raise ValueError("timer state flags must be bool")
        return not primary_complete and backup_available


@dataclass(frozen=True)
class LAEdgeLaunch:
    request_id: int
    replica_id: int
    kind: str


class LAEdgeScheduler:
    """Deterministic generalized LÆDGE scheduling kernel.

    Arrival uses two idle replicas when possible, one when only one is idle,
    and otherwise queues the unserved request.  Released replicas first serve
    oldest unserved work; only remaining idle replicas hedge oldest singly
    served work.  The paper's random replica choice is replaced by increasing
    replica ID so simulation runs are reproducible.
    """

    baseline_id = "laedge_work_conserving"

    def __init__(self, replica_ids: Sequence[int]):
        if not isinstance(replica_ids, Sequence) or isinstance(replica_ids, (str, bytes)):
            raise ValueError("replica_ids must be a sequence")
        values = tuple(sorted(_true_int(value, "replica_id") for value in replica_ids))
        if len(values) < 2 or len(values) != len(set(values)):
            raise ValueError("LÆDGE requires at least two unique replicas")
        self._replicas = values
        self._running: dict[int, int] = {}
        self._copies: dict[int, set[int]] = {}
        self._arrival_order: dict[int, int] = {}
        self._unserved: deque[int] = deque()
        self._sequence = 0

    @property
    def idle_replicas(self) -> tuple[int, ...]:
        return tuple(replica for replica in self._replicas if replica not in self._running)

    @property
    def unserved_requests(self) -> tuple[int, ...]:
        return tuple(self._unserved)

    def _launch(self, request: int, replica: int, kind: str) -> LAEdgeLaunch:
        if replica in self._running:
            raise AssertionError("LÆDGE attempted to use a busy replica")
        self._running[replica] = request
        self._copies.setdefault(request, set()).add(replica)
        return LAEdgeLaunch(request, replica, kind)

    def arrive(self, request_id: int) -> tuple[LAEdgeLaunch, ...]:
        request_id = _true_int(request_id, "request_id")
        if request_id in self._arrival_order:
            raise ValueError("request_id must be unique")
        self._arrival_order[request_id] = self._sequence
        self._sequence += 1
        idle = self.idle_replicas
        if len(idle) >= 2:
            return (
                self._launch(request_id, idle[0], "primary"),
                self._launch(request_id, idle[1], "hedge"),
            )
        if len(idle) == 1:
            return (self._launch(request_id, idle[0], "primary"),)
        self._unserved.append(request_id)
        return ()

    def complete(self, request_id: int, replica_id: int) -> tuple[LAEdgeLaunch, ...]:
        request_id = _true_int(request_id, "request_id")
        replica_id = _true_int(replica_id, "replica_id")
        if self._running.get(replica_id) != request_id:
            raise ValueError("completion does not match a running request copy")
        for replica in tuple(self._copies.pop(request_id, ())):
            self._running.pop(replica, None)

        launches: list[LAEdgeLaunch] = []
        while self.idle_replicas and self._unserved:
            pending = self._unserved.popleft()
            launches.append(self._launch(pending, self.idle_replicas[0], "primary"))
        while self.idle_replicas:
            candidates = sorted(
                (request for request, replicas in self._copies.items() if len(replicas) == 1),
                key=lambda request: (self._arrival_order[request], request),
            )
            if not candidates:
                break
            launches.append(self._launch(candidates[0], self.idle_replicas[0], "hedge"))
        return tuple(launches)


__all__ = [
    "BASELINE_CATALOG",
    "BaselineProvenance",
    "EPLBPlan",
    "FailoverOnly",
    "JoinShortestQueueRouter",
    "LAEdgeLaunch",
    "LAEdgeScheduler",
    "LPLBAssignment",
    "LeastUnfinishedWorkRouter",
    "P95DelayedHedge",
    "ReplicaLoad",
    "RoundRobinRouter",
    "lplb_token_count_minmax",
    "rebalance_experts_eplb",
    "replicate_experts_eplb",
]
