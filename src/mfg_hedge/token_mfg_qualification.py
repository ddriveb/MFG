"""Qualification-only finite-K deviation diagnostics.

The records in this module are deliberately small sufficient data.  They do
not select a response, certify a Nash equilibrium, or claim an MFG result.
The gain convention is baseline cost minus deviating cost, so a positive
value is evidence of a profitable deviation before uncertainty adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

from .reliability_aware_routing import RoutingTrace


def _true_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a true int >= {minimum}")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FiniteKGainRow:
    """One supported state/action simultaneous-gain sufficient row."""

    state_id: str
    action_id: str
    gain_mean: float
    paired_standard_error: float
    effective_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be non-empty")
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be non-empty")
        _finite(self.gain_mean, "gain_mean")
        _finite(self.paired_standard_error, "paired_standard_error", nonnegative=True)
        _true_int(self.effective_n, "effective_n", 1)


@dataclass(frozen=True)
class FixedEpisodeLibrary:
    """Immutable exogenous episode identities reusable across iterations."""

    traces: tuple[RoutingTrace, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.traces, tuple) or not self.traces:
            raise ValueError("episode library must contain a non-empty tuple")
        if any(not isinstance(trace, RoutingTrace) for trace in self.traces):
            raise ValueError("episode library contains an invalid trace")
        keys = tuple(
            (trace.namespace, trace.macro_seed, trace.scenario, trace.episode_index)
            for trace in self.traces
        )
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("episode library identities must be canonical and unique")
        if len({(trace.namespace, trace.macro_seed) for trace in self.traces}) != 1:
            raise ValueError("episode library must use one namespace and macro seed")
        if len({trace.fingerprint for trace in self.traces}) != len(self.traces):
            raise ValueError("episode library traces must be unique")
        if len({trace.topology for trace in self.traces}) != 1:
            raise ValueError("episode library must use one topology")
        expected = hashlib.sha256(
            json.dumps(
                {
                    "identities": keys,
                    "traces": tuple(trace.fingerprint for trace in self.traces),
                    "topology": self.traces[0].topology.fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if self.fingerprint != expected:
            raise ValueError("episode library fingerprint does not match traces")

    @classmethod
    def from_traces(cls, traces: Sequence[RoutingTrace]) -> "FixedEpisodeLibrary":
        rows = tuple(traces)
        if not rows:
            raise ValueError("episode library requires at least one trace")
        ordered = tuple(sorted(rows, key=lambda trace: (
            trace.namespace, trace.macro_seed, trace.scenario, trace.episode_index,
        )))
        keys = tuple(
            (trace.namespace, trace.macro_seed, trace.scenario, trace.episode_index)
            for trace in ordered
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "identities": keys,
                    "traces": tuple(trace.fingerprint for trace in ordered),
                    "topology": ordered[0].topology.fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(ordered, fingerprint)


@dataclass(frozen=True)
class SimultaneousFiniteKGainBound:
    """Maximum row-wise upper bound over the complete supported panel."""

    rows: tuple[FiniteKGainRow, ...]
    critical_value: float
    max_upper_bound: float
    supported_count: int
    complete: bool
    status: str = "complete"
    normalized_max_upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError("gain rows must be a non-empty tuple")
        _finite(self.critical_value, "critical_value", nonnegative=True)
        _finite(self.max_upper_bound, "max_upper_bound")
        _true_int(self.supported_count, "supported_count", 1)
        if self.supported_count != len(self.rows):
            raise ValueError("supported_count must equal row count")
        if self.status not in {"complete", "statistics_insufficient"}:
            raise ValueError("invalid simultaneous bound status")
        if self.normalized_max_upper_bound is not None:
            _finite(
                self.normalized_max_upper_bound,
                "normalized_max_upper_bound",
            )


def compute_simultaneous_finite_k_gain_bound(
    rows: Sequence[FiniteKGainRow],
    *,
    critical_value: float = 2.0,
    min_complete_samples: int = 2,
) -> SimultaneousFiniteKGainBound:
    """Compute a conservative max UCB over every supported row.

    This is intentionally a finite-system qualification primitive: callers
    must supply the full supported state/action panel and paired standard
    errors.  No row is averaged away before the simultaneous maximum.
    """

    if not isinstance(rows, (tuple, list)):
        raise ValueError("gain rows must be a sequence")
    critical_value = _finite(critical_value, "critical_value", nonnegative=True)
    _true_int(min_complete_samples, "min_complete_samples", 1)
    ordered = tuple(rows)
    if not ordered:
        raise ValueError("gain rows must not be empty")
    if any(not isinstance(row, FiniteKGainRow) for row in ordered):
        raise ValueError("gain rows contain an invalid row")
    keys = tuple((row.state_id, row.action_id) for row in ordered)
    if len(set(keys)) != len(keys):
        raise ValueError("gain rows contain duplicate state/action keys")
    if tuple(sorted(keys)) != keys:
        raise ValueError("gain rows must be in canonical order")
    if any(row.effective_n < min_complete_samples for row in ordered):
        raise ValueError("gain panel is below the sample floor")
    upper = tuple(
        row.gain_mean + critical_value * row.paired_standard_error
        for row in ordered
    )
    return SimultaneousFiniteKGainBound(
        ordered,
        critical_value,
        max(upper),
        len(ordered),
        True,
    )


__all__ = [
    "FiniteKGainRow",
    "FixedEpisodeLibrary",
    "SimultaneousFiniteKGainBound",
    "compute_simultaneous_finite_k_gain_bound",
]
