"""Bounded unpriced/priced Token-MFG response diagnostics.

This module is deliberately an isolated response layer.  It consumes
immutable forward summaries and fresh provider outputs; it does not replace
the finite routing engine and it does not select an action or certify a
best-response, Nash, or MFG solution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence


MODEL_IDS = ("unpriced_mfg", "priced_mfg")
MFG_FORWARD_NAMESPACE = "reliability-aware-token-mfg:v1:mfg-forward-smoke"
MFG_FORWARD_MACRO_SEED = 20260914
MFG_CONTINUATION_NAMESPACE = "reliability-aware-token-mfg:v1:mean-field-continuation"
MFG_CONTINUATION_MACRO_SEED = 20260912
MFG_FINITE_K_NAMESPACE = "reliability-aware-token-mfg:v1:finite-k-deviation"
MFG_FINITE_K_MACRO_SEED = 20260913
MFG_FORWARD_EPISODES = 16
MFG_CONTINUATION_EPISODES = 16
MFG_FINITE_K_EPISODES = 16
MFG_MIN_COMPLETE_SAMPLES = 2
MFG_MAX_ACTION_BUCKETS = 8
MFG_MAX_ITERATIONS = 10
MFG_BETA = 2.0
MFG_TEMPERATURE = 0.5
MFG_DAMPING = 0.2
MFG_RESIDUAL_TOLERANCE = 0.01
MFG_FORWARD_CALLS = MFG_FORWARD_EPISODES
MFG_CONTINUATION_CALLS = MFG_CONTINUATION_EPISODES * (1 + MFG_MAX_ACTION_BUCKETS)
MFG_FINITE_K_CALLS = MFG_FINITE_K_EPISODES * (1 + MFG_MAX_ACTION_BUCKETS)
MFG_MAX_CALLS = 2 * MFG_MAX_ITERATIONS * (
    MFG_FORWARD_CALLS + MFG_CONTINUATION_CALLS + MFG_FINITE_K_CALLS
)
CLAIM_BOUNDARY = "bounded_mfg_implementation_smoke"


class MFGSolverError(ValueError):
    """Raised for an invalid solver contract or accounting violation."""


def _strict_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MFGSolverError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MFGSolverError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise MFGSolverError(f"{name} must be finite")
    return result


def _fingerprint(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MFGSolverError("value is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _check_fingerprint(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise MFGSolverError(f"{name} must be a SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as error:
        raise MFGSolverError(f"{name} must be a SHA-256 fingerprint") from error
    return value


def _normalise_distribution(
    rows: Sequence[tuple[str, float]], name: str,
) -> tuple[tuple[str, float], ...]:
    if not isinstance(rows, tuple):
        raise MFGSolverError(f"{name} must be an immutable tuple")
    if not rows:
        raise MFGSolverError(f"{name} must not be empty")
    seen: set[str] = set()
    clean: list[tuple[str, float]] = []
    for action_id, value in rows:
        if not isinstance(action_id, str) or not action_id:
            raise MFGSolverError(f"{name} has an invalid action ID")
        if action_id in seen:
            raise MFGSolverError(f"{name} has duplicate action IDs")
        seen.add(action_id)
        probability = _finite(value, f"{name} probability", nonnegative=True)
        clean.append((action_id, probability))
    if tuple(action_id for action_id, _ in clean) != tuple(sorted(seen)):
        raise MFGSolverError(f"{name} action IDs must be in canonical order")
    if not math.isclose(
        math.fsum(value for _, value in clean), 1.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        raise MFGSolverError(f"{name} probabilities must sum to one")
    return tuple(clean)


def _l1(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return math.fsum(
        abs(left.get(key, 0.0) - right.get(key, 0.0))
        for key in set(left) | set(right)
    )


@dataclass(frozen=True)
class ActionCostEstimate:
    """Mean sufficient data for one state/action paired panel."""

    state_id: str
    action_id: str
    token_class: str
    mean_latency: float
    deadline_miss_rate: float
    replay_count: float
    lost_work: float
    observable_queue_load_index: float
    effective_n: int
    paired_standard_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise MFGSolverError("cost state_id must be non-empty")
        if not isinstance(self.action_id, str) or not self.action_id:
            raise MFGSolverError("cost action_id must be non-empty")
        if self.token_class not in {"Regular", "Urgent"}:
            raise MFGSolverError("cost token_class must be Regular or Urgent")
        _finite(self.mean_latency, "mean_latency", nonnegative=True)
        _finite(self.deadline_miss_rate, "deadline_miss_rate", nonnegative=True)
        _finite(self.replay_count, "replay_count", nonnegative=True)
        _finite(self.lost_work, "lost_work", nonnegative=True)
        _finite(
            self.observable_queue_load_index,
            "observable_queue_load_index",
            nonnegative=True,
        )
        _strict_int(self.effective_n, "effective_n", MFG_MIN_COMPLETE_SAMPLES)
        _finite(self.paired_standard_error, "paired_standard_error", nonnegative=True)


@dataclass(frozen=True)
class MFGCostModel:
    """Frozen private-cost model with an optional observable congestion charge."""

    model_id: str
    alpha_regular: float = 1.0
    alpha_urgent: float = 5.0
    gamma_regular: float = 4.0
    gamma_urgent: float = 6.0
    c_lost: float = 1.0
    price_per_queue_unit: float = 0.0
    price_basis: str = "observable_queue_load_index"

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise MFGSolverError("unsupported MFG cost model")
        for name in (
            "alpha_regular", "alpha_urgent", "gamma_regular", "gamma_urgent",
            "c_lost", "price_per_queue_unit",
        ):
            _finite(getattr(self, name), name, nonnegative=True)
        if self.price_basis != "observable_queue_load_index":
            raise MFGSolverError("unsupported congestion price basis")
        if self.model_id == "unpriced_mfg" and self.price_per_queue_unit != 0.0:
            raise MFGSolverError("unpriced model must have zero congestion charge")
        if self.model_id == "priced_mfg" and self.price_per_queue_unit != 0.25:
            raise MFGSolverError("priced model must use the frozen price 0.25")

    @classmethod
    def unpriced(cls) -> "MFGCostModel":
        return cls("unpriced_mfg")

    @classmethod
    def priced(cls) -> "MFGCostModel":
        return cls("priced_mfg", price_per_queue_unit=0.25)

    def congestion_charge(self, estimate: ActionCostEstimate) -> float:
        if not isinstance(estimate, ActionCostEstimate):
            raise MFGSolverError("cost estimate is invalid")
        return self.price_per_queue_unit * estimate.observable_queue_load_index

    def score(self, estimate: ActionCostEstimate) -> float:
        if estimate.token_class == "Regular":
            alpha, gamma = self.alpha_regular, self.gamma_regular
        else:
            alpha, gamma = self.alpha_urgent, self.gamma_urgent
        return (
            estimate.mean_latency
            + alpha * estimate.deadline_miss_rate
            + gamma * estimate.replay_count
            + self.c_lost * estimate.lost_work
            + self.congestion_charge(estimate)
        )


@dataclass(frozen=True)
class ForwardStateRow:
    state_id: str
    token_class: str
    occupancy_mass: float
    predicted_share: tuple[tuple[str, float], ...]
    realized_share: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise MFGSolverError("forward state_id must be non-empty")
        if self.token_class not in {"Regular", "Urgent"}:
            raise MFGSolverError("forward token_class must be Regular or Urgent")
        occupancy = _finite(self.occupancy_mass, "occupancy_mass", nonnegative=True)
        if occupancy <= 0.0:
            raise MFGSolverError("forward occupancy must be positive")
        predicted = _normalise_distribution(self.predicted_share, "predicted_share")
        realized = _normalise_distribution(self.realized_share, "realized_share")
        if tuple(key for key, _ in predicted) != tuple(key for key, _ in realized):
            raise MFGSolverError("predicted and realized action support differs")


@dataclass(frozen=True)
class ForwardSnapshot:
    """Complete immutable population summary for one solver iteration."""

    model_id: str
    iteration: int
    policy_fingerprint: str
    environment_fingerprint: str
    trace_library_fingerprint: str
    mu_fingerprint: str
    nu_fingerprint: str
    eta_fingerprint: str
    x_fingerprint: str
    states: tuple[ForwardStateRow, ...]
    sample_count: int
    complete: bool
    scheduler_calls: int = MFG_FORWARD_CALLS
    scale_fallback_count: int = 0

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise MFGSolverError("forward model_id is unsupported")
        _strict_int(self.iteration, "forward iteration")
        for name in (
            "policy_fingerprint", "environment_fingerprint",
            "trace_library_fingerprint", "mu_fingerprint", "nu_fingerprint",
            "eta_fingerprint", "x_fingerprint",
        ):
            _check_fingerprint(getattr(self, name), name)
        if not isinstance(self.states, tuple) or not self.states:
            raise MFGSolverError("forward states must be a non-empty tuple")
        if any(not isinstance(row, ForwardStateRow) for row in self.states):
            raise MFGSolverError("forward states contain an invalid row")
        if tuple(row.state_id for row in self.states) != tuple(
            sorted(row.state_id for row in self.states)
        ):
            raise MFGSolverError("forward states must be in canonical order")
        if len({row.state_id for row in self.states}) != len(self.states):
            raise MFGSolverError("forward states contain duplicate state IDs")
        if not math.isclose(
            math.fsum(row.occupancy_mass for row in self.states),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MFGSolverError("forward occupancies must sum to one")
        _strict_int(self.sample_count, "forward sample_count", 1)
        if type(self.complete) is not bool:
            raise MFGSolverError("forward complete must be bool")
        _strict_int(self.scheduler_calls, "forward scheduler_calls")
        if self.scheduler_calls < 1:
            raise MFGSolverError("forward scheduler call count must be positive")
        _strict_int(self.scale_fallback_count, "scale_fallback_count")

    @property
    def share_residual(self) -> float:
        return max(
            _l1(dict(row.predicted_share), dict(row.realized_share))
            for row in self.states
        )

    @property
    def population_fingerprint(self) -> str:
        return _fingerprint({
            "mu": self.mu_fingerprint,
            "nu": self.nu_fingerprint,
            "eta": self.eta_fingerprint,
            "x": self.x_fingerprint,
        })


@dataclass(frozen=True)
class FiniteKDeviationDiagnostic:
    """Sufficient diagnostics from a complete finite-K deviation panel."""

    complete: bool
    effective_n: int
    max_abs_pathwise_difference: float
    trace_library_fingerprint: str
    scheduler_calls: int = MFG_FINITE_K_CALLS
    simultaneous_ucb: float | None = None
    simultaneous_ucb_status: str = "not_computed"
    normalized_simultaneous_ucb: float | None = None

    def __post_init__(self) -> None:
        if type(self.complete) is not bool:
            raise MFGSolverError("finite-K complete must be bool")
        _strict_int(self.effective_n, "finite-K effective_n")
        _finite(
            self.max_abs_pathwise_difference,
            "finite-K max_abs_pathwise_difference",
            nonnegative=True,
        )
        _check_fingerprint(self.trace_library_fingerprint, "finite-K trace fingerprint")
        _strict_int(self.scheduler_calls, "finite-K scheduler_calls")
        if self.scheduler_calls < 1:
            raise MFGSolverError("finite-K scheduler call count must be positive")
        if self.simultaneous_ucb is not None:
            _finite(self.simultaneous_ucb, "finite-K simultaneous UCB", nonnegative=True)
        if self.simultaneous_ucb_status not in {
            "not_computed", "complete", "statistics_insufficient",
        }:
            raise MFGSolverError("invalid finite-K simultaneous UCB status")
        if self.simultaneous_ucb_status == "complete" and self.simultaneous_ucb is None:
            raise MFGSolverError("complete finite-K UCB requires a value")
        if self.normalized_simultaneous_ucb is not None:
            _finite(
                self.normalized_simultaneous_ucb,
                "normalized finite-K simultaneous UCB",
            )


@dataclass(frozen=True)
class PolicyRow:
    state_id: str
    action_ids: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise MFGSolverError("policy state_id must be non-empty")
        if not isinstance(self.action_ids, tuple) or not self.action_ids:
            raise MFGSolverError("policy action IDs must be a non-empty tuple")
        if tuple(self.action_ids) != tuple(sorted(self.action_ids)):
            raise MFGSolverError("policy action IDs must be canonical")
        if len(set(self.action_ids)) != len(self.action_ids) or any(
            not isinstance(item, str) or not item for item in self.action_ids
        ):
            raise MFGSolverError("policy action IDs are invalid or duplicated")
        if not isinstance(self.probabilities, tuple) or len(self.probabilities) != len(self.action_ids):
            raise MFGSolverError("policy probabilities do not match actions")
        clean = tuple(
            _finite(value, "policy probability", nonnegative=True)
            for value in self.probabilities
        )
        if not math.isclose(math.fsum(clean), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise MFGSolverError("policy probabilities must sum to one")


@dataclass(frozen=True)
class PolicyState:
    rows: tuple[PolicyRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise MFGSolverError("policy rows must be a non-empty tuple")
        if any(not isinstance(row, PolicyRow) for row in self.rows):
            raise MFGSolverError("policy rows contain an invalid row")
        if tuple(row.state_id for row in self.rows) != tuple(
            sorted(row.state_id for row in self.rows)
        ):
            raise MFGSolverError("policy rows must be in canonical order")
        if len({row.state_id for row in self.rows}) != len(self.rows):
            raise MFGSolverError("policy rows contain duplicate state IDs")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    @classmethod
    def uniform(cls, states: Sequence[ForwardStateRow]) -> "PolicyState":
        rows = tuple(
            PolicyRow(
                state.state_id,
                tuple(action_id for action_id, _ in state.predicted_share),
                tuple(1.0 / len(state.predicted_share) for _ in state.predicted_share),
            )
            for state in states
        )
        return cls(tuple(sorted(rows, key=lambda row: row.state_id)))

    def _row_map(self) -> dict[str, PolicyRow]:
        return {row.state_id: row for row in self.rows}

    def logit_response(
        self,
        costs_by_state: Mapping[str, Mapping[str, float]],
        *,
        beta: float = MFG_BETA,
    ) -> dict[str, tuple[float, ...]]:
        beta = _finite(beta, "beta", nonnegative=True)
        if beta <= 0.0:
            raise MFGSolverError("beta must be positive")
        if set(costs_by_state) != {row.state_id for row in self.rows}:
            raise MFGSolverError("Logit costs do not cover exactly the policy states")
        result: dict[str, tuple[float, ...]] = {}
        for row in self.rows:
            costs = costs_by_state[row.state_id]
            if tuple(costs) != row.action_ids:
                raise MFGSolverError("Logit costs do not cover canonical actions")
            values = tuple(_finite(costs[action], "Q cost") for action in row.action_ids)
            logits = tuple(-beta * value for value in values)
            peak = max(logits)
            weights = tuple(math.exp(value - peak) for value in logits)
            total = math.fsum(weights)
            result[row.state_id] = tuple(value / total for value in weights)
        return result

    def damped(
        self,
        response: Mapping[str, Sequence[float]],
        *,
        eta: float = MFG_DAMPING,
    ) -> "PolicyState":
        eta = _finite(eta, "damping", nonnegative=True)
        if not 0.0 < eta <= 1.0:
            raise MFGSolverError("damping must lie in (0, 1]")
        if set(response) != {row.state_id for row in self.rows}:
            raise MFGSolverError("response does not cover exactly the policy states")
        rows = []
        for row in self.rows:
            values = tuple(
                _finite(value, "response probability", nonnegative=True)
                for value in response[row.state_id]
            )
            if len(values) != len(row.action_ids) or not math.isclose(
                math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12,
            ):
                raise MFGSolverError("response probabilities are invalid")
            updated = tuple(
                (1.0 - eta) * old + eta * new
                for old, new in zip(row.probabilities, values)
            )
            rows.append(PolicyRow(row.state_id, row.action_ids, updated))
        return PolicyState(tuple(rows))

    def residual(self, other: "PolicyState") -> float:
        if not isinstance(other, PolicyState):
            raise MFGSolverError("policy residual requires PolicyState")
        left, right = self._row_map(), other._row_map()
        if set(left) != set(right):
            raise MFGSolverError("policy states differ")
        return max(
            math.fsum(abs(a - b) for a, b in zip(
                left[state_id].probabilities, right[state_id].probabilities,
            ))
            for state_id in left
        )


@dataclass(frozen=True)
class ScoredQRow:
    state_id: str
    action_id: str
    model_id: str
    mean_cost: float
    effective_n: int
    paired_standard_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise MFGSolverError("Q state_id must be non-empty")
        if not isinstance(self.action_id, str) or not self.action_id:
            raise MFGSolverError("Q action_id must be non-empty")
        if self.model_id not in MODEL_IDS:
            raise MFGSolverError("Q model_id is unsupported")
        _finite(self.mean_cost, "Q mean cost")
        _strict_int(self.effective_n, "Q effective_n", MFG_MIN_COMPLETE_SAMPLES)
        _finite(self.paired_standard_error, "Q paired standard error", nonnegative=True)


@dataclass(frozen=True)
class MFGSolverConfig:
    beta: float = MFG_BETA
    damping: float = MFG_DAMPING
    residual_tolerance: float = MFG_RESIDUAL_TOLERANCE
    max_iterations: int = MFG_MAX_ITERATIONS
    min_complete_samples: int = MFG_MIN_COMPLETE_SAMPLES
    call_budget: int = MFG_MAX_CALLS
    forward_calls: int = MFG_FORWARD_CALLS
    continuation_calls: int = MFG_CONTINUATION_CALLS
    finite_k_calls: int = MFG_FINITE_K_CALLS

    def __post_init__(self) -> None:
        _finite(self.beta, "config beta", nonnegative=True)
        if self.beta <= 0.0:
            raise MFGSolverError("config beta must be positive")
        _finite(self.damping, "config damping", nonnegative=True)
        if not 0.0 < self.damping <= 1.0:
            raise MFGSolverError("config damping must lie in (0, 1]")
        _finite(self.residual_tolerance, "config residual_tolerance", nonnegative=True)
        if self.residual_tolerance <= 0.0:
            raise MFGSolverError("config residual_tolerance must be positive")
        _strict_int(self.max_iterations, "config max_iterations", 1)
        _strict_int(self.min_complete_samples, "config min_complete_samples", 1)
        _strict_int(self.call_budget, "config call_budget", 1)
        _strict_int(self.forward_calls, "config forward_calls", 1)
        _strict_int(self.continuation_calls, "config continuation_calls", 1)
        _strict_int(self.finite_k_calls, "config finite_k_calls", 1)


@dataclass(frozen=True)
class CallReservation:
    reservation_id: int
    count: int


class CallLedger:
    """Parent-owned monotone reservation/settlement accounting."""

    def __init__(self, limit: int) -> None:
        _strict_int(limit, "call budget", 1)
        self.limit = limit
        self.reserved_calls = 0
        self.completed_calls = 0
        self.failed_calls = 0
        self._next_reservation_id = 0
        self._open_reservations: dict[int, int] = {}

    @property
    def attempted_calls(self) -> int:
        return self.completed_calls + self.failed_calls

    def reserve(self, count: int) -> CallReservation:
        _strict_int(count, "reserved call count", 1)
        if self.reserved_calls + count > self.limit:
            raise MFGSolverError("scheduler call budget exhausted before dispatch")
        self.reserved_calls += count
        reservation = CallReservation(self._next_reservation_id, count)
        self._next_reservation_id += 1
        self._open_reservations[reservation.reservation_id] = count
        return reservation

    def _consume(self, reservation: CallReservation) -> int:
        if not isinstance(reservation, CallReservation):
            raise MFGSolverError("invalid call reservation")
        count = self._open_reservations.pop(reservation.reservation_id, None)
        if count is None or count != reservation.count:
            raise MFGSolverError("call reservation is unknown or already settled")
        return count

    def settle(self, reservation: CallReservation, *, completed: int) -> None:
        _strict_int(completed, "completed call count")
        if completed > reservation.count:
            raise MFGSolverError("completed calls exceed reservation")
        self._consume(reservation)
        self.completed_calls += completed

    def fail(self, reservation: CallReservation, *, completed: int = 0) -> None:
        _strict_int(completed, "completed call count")
        if completed > reservation.count:
            raise MFGSolverError("completed calls exceed reservation")
        count = self._consume(reservation)
        self.completed_calls += completed
        self.failed_calls += count - completed


@dataclass(frozen=True)
class MFGIterationRecord:
    iteration: int
    model_id: str
    status: str
    policy_rows: tuple[PolicyRow, ...]
    q_rows: tuple[ScoredQRow, ...]
    policy_before_fingerprint: str
    policy_after_fingerprint: str
    environment_fingerprint: str
    mu_fingerprint: str
    nu_fingerprint: str
    eta_fingerprint: str
    x_fingerprint: str
    q_fingerprint: str
    policy_residual: float | None
    population_residual: float | None
    share_residual: float | None
    finite_k_status: str
    finite_k_max_abs_difference: float | None
    scheduler_calls: int
    stop_reason: str


@dataclass(frozen=True)
class MFGModelResult:
    model_id: str
    status: str
    iterations: tuple[MFGIterationRecord, ...]
    final_policy: PolicyState
    attempted_calls: int
    reserved_calls: int
    claims_br: bool = False
    claims_regret: bool = False
    claims_nash: bool = False
    claims_mfg: bool = False
    claim_boundary: str = CLAIM_BOUNDARY
    failure_type: str | None = None
    failure_message: str | None = None
    failure_iteration: int | None = None
    cycle_start: int | None = None
    cycle_period: int | None = None


@dataclass(frozen=True)
class MFGSolverBatchResult:
    model_results: tuple[MFGModelResult, ...]
    call_budget: int
    attempted_calls: int
    reserved_calls: int
    model_ids: tuple[str, ...] = MODEL_IDS
    claim_boundary: str = CLAIM_BOUNDARY


@dataclass(frozen=True)
class FinalPolicyConfirmation:
    """A final policy and its freshly recomputed environment/Q tuple.

    ``confirmed`` is reserved for a policy whose forward snapshot, Q panel,
    response residual, population residual, share residual, and finite-K
    diagnostic all pass the supplied readiness tolerances.  This helper is
    separate from the bounded Stage-5 iteration so the historical smoke
    result remains unchanged.
    """

    model_id: str
    status: str
    policy: PolicyState
    snapshot: ForwardSnapshot | None
    q_rows: tuple[ScoredQRow, ...]
    response_residual: float | None
    population_residual: float | None
    share_residual: float | None
    finite_k_status: str
    finite_k_max_abs_difference: float | None
    attempted_calls: int
    failure_type: str | None = None
    failure_message: str | None = None
    failure_iteration: int | None = None
    claim_boundary: str = CLAIM_BOUNDARY
    finite_k_simultaneous_ucb: float | None = None
    finite_k_simultaneous_ucb_status: str = "not_computed"
    finite_k_normalized_simultaneous_ucb: float | None = None

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_IDS:
            raise MFGSolverError("confirmation model_id is unsupported")
        if self.status not in {
            "confirmed", "not_converged", "statistics_insufficient", "physical_failed",
        }:
            raise MFGSolverError("invalid final confirmation status")
        if not isinstance(self.policy, PolicyState):
            raise MFGSolverError("confirmation policy is invalid")
        if self.snapshot is not None and not isinstance(self.snapshot, ForwardSnapshot):
            raise MFGSolverError("confirmation snapshot is invalid")
        if not isinstance(self.q_rows, tuple):
            raise MFGSolverError("confirmation Q rows must be immutable")
        _strict_int(self.attempted_calls, "confirmation attempted_calls")
        if self.response_residual is not None:
            _finite(self.response_residual, "confirmation response residual", nonnegative=True)
        if self.population_residual is not None:
            _finite(self.population_residual, "confirmation population residual", nonnegative=True)
        if self.share_residual is not None:
            _finite(self.share_residual, "confirmation share residual", nonnegative=True)
        if self.finite_k_max_abs_difference is not None:
            _finite(
                self.finite_k_max_abs_difference,
                "confirmation finite-K residual",
                nonnegative=True,
            )
        if self.finite_k_simultaneous_ucb is not None:
            _finite(
                self.finite_k_simultaneous_ucb,
                "confirmation finite-K simultaneous UCB",
                nonnegative=True,
            )
        if self.finite_k_simultaneous_ucb_status not in {
            "not_computed", "complete", "statistics_insufficient",
        }:
            raise MFGSolverError("invalid confirmation finite-K UCB status")
        if self.finite_k_normalized_simultaneous_ucb is not None:
            _finite(
                self.finite_k_normalized_simultaneous_ucb,
                "confirmation normalized finite-K simultaneous UCB",
            )
        if self.status == "confirmed":
            if self.snapshot is None or not self.q_rows:
                raise MFGSolverError("confirmed result must contain aligned snapshot and Q")
            if self.snapshot.policy_fingerprint != self.policy.fingerprint:
                raise MFGSolverError("confirmed snapshot used a stale policy")


def _validate_snapshot(
    snapshot: ForwardSnapshot,
    policy: PolicyState,
    model_id: str,
    iteration: int,
    *,
    minimum_samples: int = MFG_FORWARD_EPISODES,
    expected_scheduler_calls: int = MFG_FORWARD_CALLS,
) -> None:
    if not isinstance(snapshot, ForwardSnapshot):
        raise MFGSolverError("forward provider did not return ForwardSnapshot")
    if snapshot.model_id != model_id or snapshot.iteration != iteration:
        raise MFGSolverError("forward snapshot identity does not match request")
    if snapshot.policy_fingerprint != policy.fingerprint:
        raise MFGSolverError("forward snapshot used a stale policy")
    if not snapshot.complete or snapshot.sample_count < minimum_samples:
        raise MFGSolverError("forward statistics are incomplete")
    if snapshot.scheduler_calls != expected_scheduler_calls:
        raise MFGSolverError("forward scheduler call count differs from execution contract")


def _score_q_rows(
    snapshot: ForwardSnapshot,
    estimates: Sequence[ActionCostEstimate],
    model: MFGCostModel,
    minimum_n: int,
) -> tuple[ScoredQRow, ...]:
    if not isinstance(estimates, tuple):
        estimates = tuple(estimates)
    if not estimates:
        raise MFGSolverError("continuation provider returned no Q rows")
    expected: dict[tuple[str, str], str] = {}
    for state in snapshot.states:
        for action_id, _ in state.predicted_share:
            expected[(state.state_id, action_id)] = state.token_class
    actual: set[tuple[str, str]] = set()
    scored: list[ScoredQRow] = []
    for estimate in estimates:
        if not isinstance(estimate, ActionCostEstimate):
            raise MFGSolverError("continuation returned an invalid Q row")
        key = (estimate.state_id, estimate.action_id)
        if key in actual:
            raise MFGSolverError("continuation returned duplicate Q row")
        if key not in expected or expected[key] != estimate.token_class:
            raise MFGSolverError("continuation Q support does not match forward state")
        if estimate.effective_n < minimum_n:
            raise MFGSolverError("continuation panel is below the sample floor")
        actual.add(key)
        scored.append(ScoredQRow(
            estimate.state_id,
            estimate.action_id,
            model.model_id,
            model.score(estimate),
            estimate.effective_n,
            estimate.paired_standard_error,
        ))
    if actual != set(expected):
        raise MFGSolverError("continuation Q panel is incomplete")
    return tuple(sorted(scored, key=lambda row: (row.state_id, row.action_id)))


def _q_mapping(rows: Sequence[ScoredQRow]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        result.setdefault(row.state_id, {})[row.action_id] = row.mean_cost
    return result


def _population_residual(
    previous: ForwardSnapshot | None,
    current: ForwardSnapshot,
) -> float | None:
    if previous is None:
        return None
    left = {row.state_id: row.occupancy_mass for row in previous.states}
    right = {row.state_id: row.occupancy_mass for row in current.states}
    return _l1(left, right)


def confirm_final_policy(
    policy: PolicyState,
    model: MFGCostModel,
    forward_provider: Callable[[PolicyState, str, int], ForwardSnapshot],
    continuation_provider: Callable[
        [ForwardSnapshot, PolicyState, str, int], Sequence[ActionCostEstimate]
    ],
    deviation_provider: Callable[
        [ForwardSnapshot, PolicyState, str, int], FiniteKDeviationDiagnostic
    ],
    *,
    previous_snapshot: ForwardSnapshot | None = None,
    config: MFGSolverConfig = MFGSolverConfig(),
) -> FinalPolicyConfirmation:
    """Recompute the final policy/environment/Q tuple on a fixed library.

    The existing iterative solver intentionally remains a Stage-5 bounded
    diagnostic.  Qualification callers use this explicit confirmation pass,
    which never treats the pre-update snapshot as the final environment.
    """

    if not isinstance(policy, PolicyState):
        raise MFGSolverError("confirmation policy must be PolicyState")
    if not isinstance(model, MFGCostModel):
        raise MFGSolverError("confirmation model must be MFGCostModel")
    if not callable(forward_provider) or not callable(continuation_provider):
        raise MFGSolverError("confirmation providers must be callable")
    if not callable(deviation_provider):
        raise MFGSolverError("confirmation deviation provider must be callable")
    if not isinstance(config, MFGSolverConfig):
        raise MFGSolverError("confirmation config must be MFGSolverConfig")
    if previous_snapshot is not None and not isinstance(previous_snapshot, ForwardSnapshot):
        raise MFGSolverError("previous_snapshot must be ForwardSnapshot")
    iteration = previous_snapshot.iteration + 1 if previous_snapshot is not None else 0
    attempted = 0

    def failure(
        status: str,
        error: Exception,
        *,
        snapshot: ForwardSnapshot | None = None,
        q_rows: tuple[ScoredQRow, ...] = (),
        response_residual: float | None = None,
        population_residual: float | None = None,
        share_residual: float | None = None,
        finite_k_status: str = "not_run",
        finite_k_max_abs_difference: float | None = None,
        finite_k_simultaneous_ucb: float | None = None,
        finite_k_simultaneous_ucb_status: str = "not_computed",
        finite_k_normalized_simultaneous_ucb: float | None = None,
    ) -> FinalPolicyConfirmation:
        return FinalPolicyConfirmation(
            model.model_id,
            status,
            policy,
            snapshot,
            q_rows,
            response_residual,
            population_residual,
            share_residual,
            finite_k_status,
            finite_k_max_abs_difference,
            attempted,
            type(error).__name__,
            str(error),
            iteration,
            CLAIM_BOUNDARY,
            finite_k_simultaneous_ucb,
            finite_k_simultaneous_ucb_status,
            finite_k_normalized_simultaneous_ucb,
        )

    try:
        attempted += config.forward_calls
        snapshot = forward_provider(policy, model.model_id, iteration)
        _validate_snapshot(
            snapshot, policy, model.model_id, iteration,
            minimum_samples=config.min_complete_samples,
            expected_scheduler_calls=config.forward_calls,
        )
        if (
            previous_snapshot is not None
            and snapshot.trace_library_fingerprint
            != previous_snapshot.trace_library_fingerprint
        ):
            raise MFGSolverError("final confirmation changed the fixed trace library")
    except Exception as error:
        if "statistics are incomplete" in str(error):
            return failure("statistics_insufficient", error)
        return failure("physical_failed", error)

    population_residual = _population_residual(previous_snapshot, snapshot)
    share_residual = snapshot.share_residual
    try:
        attempted += config.continuation_calls
        estimates = continuation_provider(snapshot, policy, model.model_id, iteration)
        q_rows = _score_q_rows(
            snapshot, estimates, model, config.min_complete_samples,
        )
    except Exception as error:
        if "incomplete" in str(error) or "below the sample floor" in str(error):
            return failure(
                "statistics_insufficient", error,
                snapshot=snapshot,
                population_residual=population_residual,
                share_residual=share_residual,
            )
        return failure(
            "physical_failed", error,
            snapshot=snapshot,
            population_residual=population_residual,
            share_residual=share_residual,
        )

    try:
        response = policy.logit_response(_q_mapping(q_rows), beta=config.beta)
        response_residual = max(
            math.fsum(
                abs(old - new)
                for old, new in zip(
                    row.probabilities, response[row.state_id],
                )
            )
            for row in policy.rows
        )
    except Exception as error:
        return failure(
            "physical_failed", error,
            snapshot=snapshot,
            q_rows=q_rows,
            population_residual=population_residual,
            share_residual=share_residual,
        )

    finite_k_status = "not_run"
    finite_k_max_abs_difference: float | None = None
    finite_k_simultaneous_ucb: float | None = None
    finite_k_simultaneous_ucb_status = "not_computed"
    finite_k_normalized_simultaneous_ucb: float | None = None
    try:
        attempted += config.finite_k_calls
        diagnostic = deviation_provider(snapshot, policy, model.model_id, iteration)
        if not isinstance(diagnostic, FiniteKDeviationDiagnostic):
            raise MFGSolverError("finite-K provider did not return its diagnostic type")
        if diagnostic.scheduler_calls != config.finite_k_calls:
            raise MFGSolverError(
                "finite-K scheduler call count differs from execution contract"
            )
        if (
            not diagnostic.complete
            or diagnostic.effective_n < config.min_complete_samples
        ):
            raise MFGSolverError("finite-K deviation panel is below the sample floor")
        if diagnostic.simultaneous_ucb_status == "statistics_insufficient":
            raise MFGSolverError("finite-K simultaneous UCB is below the sample floor")
        finite_k_status = "complete"
        finite_k_max_abs_difference = diagnostic.max_abs_pathwise_difference
        finite_k_simultaneous_ucb = diagnostic.simultaneous_ucb
        finite_k_simultaneous_ucb_status = diagnostic.simultaneous_ucb_status
        finite_k_normalized_simultaneous_ucb = diagnostic.normalized_simultaneous_ucb
    except Exception as error:
        if "below the sample floor" in str(error) or "incomplete" in str(error):
            return failure(
                "statistics_insufficient", error,
                snapshot=snapshot,
                q_rows=q_rows,
                response_residual=response_residual,
                population_residual=population_residual,
                share_residual=share_residual,
                finite_k_status="statistics_insufficient",
            )
        return failure(
            "physical_failed", error,
            snapshot=snapshot,
            q_rows=q_rows,
            response_residual=response_residual,
            population_residual=population_residual,
            share_residual=share_residual,
            finite_k_status="physical_failed",
        )

    if (
        population_residual is None
        or response_residual > config.residual_tolerance
        or population_residual > config.residual_tolerance
        or share_residual > config.residual_tolerance
    ):
        error = MFGSolverError("final policy/environment residuals exceed tolerance")
        return failure(
            "not_converged", error,
            snapshot=snapshot,
            q_rows=q_rows,
            response_residual=response_residual,
            population_residual=population_residual,
            share_residual=share_residual,
            finite_k_status=finite_k_status,
            finite_k_max_abs_difference=finite_k_max_abs_difference,
            finite_k_simultaneous_ucb=finite_k_simultaneous_ucb,
            finite_k_simultaneous_ucb_status=finite_k_simultaneous_ucb_status,
            finite_k_normalized_simultaneous_ucb=(
                finite_k_normalized_simultaneous_ucb
            ),
        )
    return FinalPolicyConfirmation(
        model.model_id,
        "confirmed",
        policy,
        snapshot,
        q_rows,
        response_residual,
        population_residual,
        share_residual,
        finite_k_status,
        finite_k_max_abs_difference,
        attempted,
        None,
        None,
        None,
        CLAIM_BOUNDARY,
        finite_k_simultaneous_ucb,
        finite_k_simultaneous_ucb_status,
        finite_k_normalized_simultaneous_ucb,
    )


def _state_fingerprint(policy: PolicyState, snapshot: ForwardSnapshot) -> str:
    return _fingerprint({
        "policy": policy.fingerprint,
        "population": snapshot.population_fingerprint,
        "share": snapshot.x_fingerprint,
    })


def _failed_result(
    model_id: str,
    current_policy: PolicyState,
    iterations: Sequence[MFGIterationRecord],
    ledger: CallLedger,
    *,
    iteration: int,
    error: Exception,
) -> MFGModelResult:
    return MFGModelResult(
        model_id,
        "physical_failed",
        tuple(iterations),
        current_policy,
        ledger.attempted_calls,
        ledger.reserved_calls,
        failure_type=type(error).__name__,
        failure_message=str(error),
        failure_iteration=iteration,
    )


def _solve_model(
    model: MFGCostModel,
    initial_policy: PolicyState,
    forward_provider: Callable[[PolicyState, str, int], ForwardSnapshot],
    continuation_provider: Callable[
        [ForwardSnapshot, PolicyState, str, int], Sequence[ActionCostEstimate]
    ],
    deviation_provider: Callable[
        [ForwardSnapshot, PolicyState, str, int], FiniteKDeviationDiagnostic
    ],
    config: MFGSolverConfig,
    ledger: CallLedger,
) -> MFGModelResult:
    policy = initial_policy
    previous_snapshot: ForwardSnapshot | None = None
    iterations: list[MFGIterationRecord] = []
    seen: dict[str, int] = {}

    for iteration in range(config.max_iterations):
        before = policy
        forward_reservation = ledger.reserve(config.forward_calls)
        try:
            snapshot = forward_provider(policy, model.model_id, iteration)
        except Exception as error:
            ledger.fail(forward_reservation)
            return _failed_result(
                model.model_id, policy, iterations, ledger,
                iteration=iteration, error=error,
            )
        try:
            _validate_snapshot(
                snapshot, policy, model.model_id, iteration,
                minimum_samples=config.min_complete_samples,
                expected_scheduler_calls=config.forward_calls,
            )
        except Exception as error:
            ledger.fail(forward_reservation)
            if "statistics are incomplete" in str(error):
                iterations.append(MFGIterationRecord(
                    iteration, model.model_id, "statistics_insufficient",
                    tuple(policy.rows), (), policy.fingerprint, policy.fingerprint,
                    snapshot.environment_fingerprint, snapshot.mu_fingerprint,
                    snapshot.nu_fingerprint, snapshot.eta_fingerprint,
                    snapshot.x_fingerprint, _fingerprint(()), None,
                    _population_residual(previous_snapshot, snapshot),
                    snapshot.share_residual, "not_run", None,
                    ledger.attempted_calls, "forward_statistics_insufficient",
                ))
                return MFGModelResult(
                    model.model_id, "statistics_insufficient", tuple(iterations), policy,
                    ledger.attempted_calls, ledger.reserved_calls,
                )
            return _failed_result(
                model.model_id, policy, iterations, ledger,
                iteration=iteration, error=error,
            )
        ledger.settle(forward_reservation, completed=config.forward_calls)
        try:
            continuation_reservation = ledger.reserve(config.continuation_calls)
        except Exception as error:
            return _failed_result(
                model.model_id, initial_policy, iterations, ledger,
                iteration=iteration, error=error,
            )
        try:
            estimates = continuation_provider(snapshot, policy, model.model_id, iteration)
        except Exception as error:
            ledger.fail(continuation_reservation)
            return _failed_result(
                model.model_id, policy, iterations, ledger,
                iteration=iteration, error=error,
            )
        try:
            q_rows = _score_q_rows(snapshot, estimates, model, config.min_complete_samples)
        except Exception as error:
            ledger.fail(continuation_reservation)
            record = MFGIterationRecord(
                iteration, model.model_id, "statistics_insufficient", (), (),
                before.fingerprint, before.fingerprint,
                snapshot.environment_fingerprint, snapshot.mu_fingerprint,
                snapshot.nu_fingerprint, snapshot.eta_fingerprint, snapshot.x_fingerprint,
                _fingerprint(()), None, _population_residual(previous_snapshot, snapshot),
                snapshot.share_residual, "not_run", None,
                ledger.attempted_calls, "continuation_statistics_insufficient",
            )
            iterations.append(record)
            return MFGModelResult(
                model.model_id, "statistics_insufficient", tuple(iterations), before,
                ledger.attempted_calls, ledger.reserved_calls,
            )
        ledger.settle(continuation_reservation, completed=config.continuation_calls)
        finite_reservation = ledger.reserve(config.finite_k_calls)
        try:
            diagnostic = deviation_provider(snapshot, policy, model.model_id, iteration)
        except Exception as error:
            ledger.fail(finite_reservation)
            return _failed_result(
                model.model_id, policy, iterations, ledger,
                iteration=iteration, error=error,
            )
        try:
            if not isinstance(diagnostic, FiniteKDeviationDiagnostic):
                raise MFGSolverError("finite-K provider did not return its diagnostic type")
            if diagnostic.scheduler_calls != config.finite_k_calls:
                raise MFGSolverError(
                    "finite-K scheduler call count differs from execution contract"
                )
            if (
                not diagnostic.complete
                or diagnostic.effective_n < config.min_complete_samples
            ):
                raise MFGSolverError("finite-K deviation panel is below the sample floor")
        except Exception as error:
            ledger.fail(finite_reservation)
            record = MFGIterationRecord(
                iteration, model.model_id, "statistics_insufficient", tuple(before.rows),
                q_rows, before.fingerprint, before.fingerprint,
                snapshot.environment_fingerprint, snapshot.mu_fingerprint,
                snapshot.nu_fingerprint, snapshot.eta_fingerprint, snapshot.x_fingerprint,
                _fingerprint(tuple(asdict(row) for row in q_rows)), None,
                _population_residual(previous_snapshot, snapshot), snapshot.share_residual,
                "statistics_insufficient", None, ledger.attempted_calls,
                "finite_k_statistics_insufficient",
            )
            iterations.append(record)
            return MFGModelResult(
                model.model_id, "statistics_insufficient", tuple(iterations), before,
                ledger.attempted_calls, ledger.reserved_calls,
            )
        ledger.settle(finite_reservation, completed=config.finite_k_calls)

        response = before.logit_response(_q_mapping(q_rows), beta=config.beta)
        after = before.damped(response, eta=config.damping)
        policy_residual = before.residual(after)
        population_residual = _population_residual(previous_snapshot, snapshot)
        share_residual = snapshot.share_residual
        state_fp = _state_fingerprint(after, snapshot)
        q_fp = _fingerprint(tuple(asdict(row) for row in q_rows))
        converged = (
            policy_residual <= config.residual_tolerance
            and population_residual is not None
            and population_residual <= config.residual_tolerance
            and share_residual <= config.residual_tolerance
        )
        cycle_start = seen.get(state_fp)
        if converged:
            status, reason = "fixed_point", "all_frozen_residuals_within_tolerance"
        elif cycle_start is not None:
            status, reason = "cycle", "repeated_composite_policy_population_share_state"
        else:
            status, reason = "running", "continue_iteration"
        record = MFGIterationRecord(
            iteration, model.model_id, status, tuple(after.rows), q_rows,
            before.fingerprint, after.fingerprint,
            snapshot.environment_fingerprint, snapshot.mu_fingerprint,
            snapshot.nu_fingerprint, snapshot.eta_fingerprint, snapshot.x_fingerprint,
            q_fp, policy_residual, population_residual, share_residual,
            "complete", diagnostic.max_abs_pathwise_difference,
            ledger.attempted_calls, reason,
        )
        iterations.append(record)
        if status == "fixed_point":
            return MFGModelResult(
                model.model_id, status, tuple(iterations), after,
                ledger.attempted_calls, ledger.reserved_calls,
            )
        if status == "cycle":
            return MFGModelResult(
                model.model_id, status, tuple(iterations), after,
                ledger.attempted_calls, ledger.reserved_calls,
                cycle_start=cycle_start,
                cycle_period=iteration - cycle_start,
            )
        seen[state_fp] = iteration
        policy = after
        previous_snapshot = snapshot

    return MFGModelResult(
        model.model_id, "not_converged", tuple(iterations), policy,
        ledger.attempted_calls, ledger.reserved_calls,
    )


def solve_unpriced_priced_mfg(
    forward_provider: Callable[[PolicyState, str, int], ForwardSnapshot],
    continuation_provider: Callable[
        [ForwardSnapshot, PolicyState, str, int], Sequence[ActionCostEstimate]
    ],
    deviation_provider: Callable[
        [ForwardSnapshot, PolicyState, str, int], FiniteKDeviationDiagnostic
    ],
    *,
    initial_policy: PolicyState,
    config: MFGSolverConfig = MFGSolverConfig(),
    models: Sequence[MFGCostModel] = (
        MFGCostModel.unpriced(), MFGCostModel.priced(),
    ),
) -> MFGSolverBatchResult:
    """Run the two frozen model diagnostics with a parent-owned call budget."""

    if not callable(forward_provider) or not callable(continuation_provider):
        raise MFGSolverError("forward and continuation providers must be callable")
    if not callable(deviation_provider):
        raise MFGSolverError("deviation provider must be callable")
    if not isinstance(initial_policy, PolicyState):
        raise MFGSolverError("initial_policy must be PolicyState")
    if not isinstance(config, MFGSolverConfig):
        raise MFGSolverError("config must be MFGSolverConfig")
    model_rows = tuple(models)
    model_ids = tuple(model.model_id for model in model_rows)
    if not model_ids or any(model_id not in MODEL_IDS for model_id in model_ids):
        raise MFGSolverError("models must contain supported model IDs")
    if len(set(model_ids)) != len(model_ids):
        raise MFGSolverError("models must not contain duplicate model IDs")
    required_budget = len(model_rows) * config.max_iterations * (
        config.forward_calls + config.continuation_calls + config.finite_k_calls
    )
    if config.call_budget < required_budget:
        raise MFGSolverError("solver call budget is below the declared execution contract")
    ledger = CallLedger(config.call_budget)
    results = tuple(
        _solve_model(
            model, initial_policy, forward_provider, continuation_provider,
            deviation_provider, config, ledger,
        )
        for model in model_rows
    )
    return MFGSolverBatchResult(
        results, config.call_budget, ledger.attempted_calls, ledger.reserved_calls,
        model_ids=model_ids,
    )


__all__ = [
    "ActionCostEstimate",
    "CallLedger",
    "CLAIM_BOUNDARY",
    "FiniteKDeviationDiagnostic",
    "FinalPolicyConfirmation",
    "ForwardSnapshot",
    "ForwardStateRow",
    "MFGCostModel",
    "MFGSolverBatchResult",
    "MFGSolverConfig",
    "MFGSolverError",
    "MFGIterationRecord",
    "MFGModelResult",
    "MFG_MAX_CALLS",
    "MODEL_IDS",
    "PolicyRow",
    "PolicyState",
    "ScoredQRow",
    "confirm_final_policy",
    "solve_unpriced_priced_mfg",
]
