"""Exact continuous reduction for the frozen equal-work communication model.

Rows with the same source and destination mask have identical per-route
resource costs. Merging their demands preserves every attainable resource-load
vector; expansion uses the common group share for each original Expert row.
This relies on equal Expert work and no activation/entropy/Expert-specific cost.
It changes numerical conditioning, not the continuous optimum. Integer rounding
is still performed on ORIGINAL rows: no integer-optimality claim is made.
"""
from dataclasses import dataclass
from typing import Sequence
from .moe_communication_social_v2 import SparseSocialProblem, RouteRow

@dataclass(frozen=True)
class EqualSourceMaskReduction:
    original: SparseSocialProblem
    problem: SparseSocialProblem
    row_groups: tuple[int, ...]
    original_slot_to_group_slot: tuple[tuple[int, ...], ...]

    def expand(self, shares: Sequence[float]) -> tuple[float, ...]:
        q = self.problem.validate_shares(shares)
        result = []
        for row_index, group_index in enumerate(self.row_groups):
            start = self.problem.row_offsets[group_index]
            result.extend(q[start + slot] for slot in self.original_slot_to_group_slot[row_index])
        return self.original.validate_shares(result)

    def collapse(self, shares: Sequence[float]) -> tuple[float, ...]:
        p = self.original.validate_shares(shares)
        sums = [0.0] * self.problem.variable_count
        for row_index, row in enumerate(self.original.rows):
            group_index = self.row_groups[row_index]
            group_start = self.problem.row_offsets[group_index]
            original_start = self.original.row_offsets[row_index]
            for slot, group_slot in enumerate(self.original_slot_to_group_slot[row_index]):
                sums[group_start + group_slot] += row.demand * p[original_start + slot]
        result = []
        for group_index, row in enumerate(self.problem.rows):
            start, end = self.problem.row_offsets[group_index:group_index+2]
            result.extend(value / row.demand if row.demand else 1.0 / (end-start) for value in sums[start:end])
        return self.problem.validate_shares(result)

def reduce_equal_source_masks(problem: SparseSocialProblem) -> EqualSourceMaskReduction:
    """Merge only (source, unordered legal mask), then retain slot remapping."""
    if not isinstance(problem, SparseSocialProblem):
        raise TypeError("problem must be SparseSocialProblem")
    keys = {}; demands = []; sources = []; masks = []; groups = []; slot_maps = []
    for row in problem.rows:
        mask = tuple(sorted(row.allowed_gpus))
        key = (row.source_gpu, mask)
        if key not in keys:
            keys[key] = len(demands)
            demands.append(0); sources.append(row.source_gpu); masks.append(mask)
        group_index = keys[key]
        demands[group_index] += row.demand
        groups.append(group_index)
        slot_maps.append(tuple(mask.index(gpu) for gpu in row.allowed_gpus))
    # Avoid rebuilding the same sparse operator when no rows can be merged.
    if len(demands) == len(problem.rows):
        return EqualSourceMaskReduction(
            problem, problem, tuple(range(len(problem.rows))),
            tuple(tuple(range(len(row.allowed_gpus))) for row in problem.rows),
        )
    rows = tuple(RouteRow(sources[i],i,demands[i],masks[i]) for i in range(len(demands)))
    reduced = SparseSocialProblem.from_rows(rows,problem.config)
    return EqualSourceMaskReduction(problem,reduced,tuple(groups),tuple(slot_maps))