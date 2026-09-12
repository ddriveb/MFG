"""Versioned experiment configuration with fail-fast validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .common_state import CommonStateTimeline


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_name: str
    expert_count: int
    replicas_per_expert: int
    failure_domain_count: int
    regular_token_ratio: float
    urgent_token_ratio: float
    healthy_service_mean: float
    service_time_cv: float
    degraded_slowdown: float
    hedge_delay_quantile: float
    target_max_utilization: float
    healthy_offered_load: float
    softmax_inverse_temperature: float
    policy_damping: float
    price_step: float
    price_damping: float
    tokens_per_run: int
    base_seed: int
    seed_count: int
    smoke_tokens: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        expected = {item.name for item in fields(cls)}
        supplied = set(raw)
        missing = expected - supplied
        unknown = supplied - expected
        if missing:
            raise ValueError(f"Missing configuration keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        for name in (
            "expert_count",
            "replicas_per_expert",
            "failure_domain_count",
            "tokens_per_run",
            "seed_count",
            "smoke_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.replicas_per_expert < 2:
            raise ValueError("At least two replicas are required for cross-domain hedging")
        if self.failure_domain_count < 2:
            raise ValueError("At least two failure domains are required")
        if self.replicas_per_expert < self.failure_domain_count:
            raise ValueError("Every failure domain needs at least one replica in v1")
        ratio_sum = self.regular_token_ratio + self.urgent_token_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError("Token class ratios must sum to 1")
        for name in (
            "regular_token_ratio",
            "urgent_token_ratio",
            "hedge_delay_quantile",
            "target_max_utilization",
            "policy_damping",
            "price_damping",
        ):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        for name in (
            "healthy_service_mean",
            "service_time_cv",
            "degraded_slowdown",
            "healthy_offered_load",
            "softmax_inverse_temperature",
            "price_step",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.degraded_slowdown < 1.0:
            raise ValueError("degraded_slowdown must be at least 1")
        if self.smoke_tokens > self.tokens_per_run:
            raise ValueError("smoke_tokens cannot exceed tokens_per_run")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")

    @property
    def single_domain_failure_load(self) -> float:
        """Offered load after losing one equal-capacity domain."""
        surviving_fraction = (self.failure_domain_count - 1) / self.failure_domain_count
        return self.healthy_offered_load / surviving_fraction

    def audit_findings(self) -> tuple[str, ...]:
        findings: list[str] = []
        if self.single_domain_failure_load > self.target_max_utilization:
            findings.append(
                "single-domain failure makes base load exceed target capacity "
                f"({self.single_domain_failure_load:.3f} > "
                f"{self.target_max_utilization:.3f}); treat it as a transient "
                "scenario or provision more headroom"
            )
        return tuple(findings)


def parse_config_bytes(payload: bytes) -> ExperimentConfig:
    """Parse one immutable UTF-8 JSON payload into a validated configuration."""
    try:
        raw = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("Configuration must be UTF-8 encoded") from exc
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration value must be an object")
    return ExperimentConfig.from_mapping(raw)


def load_config(path: str | Path) -> ExperimentConfig:
    return parse_config_bytes(Path(path).read_bytes())


def load_config_with_sha256(path: str | Path) -> tuple[ExperimentConfig, str]:
    """Read once, then parse and hash the exact same configuration bytes."""
    payload = Path(path).read_bytes()
    return parse_config_bytes(payload), hashlib.sha256(payload).hexdigest()


_BASE_CONFIG_FIELD_NAMES = tuple(item.name for item in fields(ExperimentConfig))

_INT_CONFIG_FIELDS = (
    "schema_version",
    "expert_count",
    "replicas_per_expert",
    "failure_domain_count",
    "tokens_per_run",
    "base_seed",
    "seed_count",
    "smoke_tokens",
)


@dataclass(frozen=True)
class CommonStateExperimentConfig:
    """Self-contained schema-2 configuration for the Common State slice.

    Duplicates every schema-1 base key (no shared-inheritance mechanism) and
    adds the failure timeline. Base constraints are enforced by projecting to
    `ExperimentConfig`; the timeline is validated by `CommonStateTimeline`.
    """

    schema_version: int
    experiment_name: str
    expert_count: int
    replicas_per_expert: int
    failure_domain_count: int
    regular_token_ratio: float
    urgent_token_ratio: float
    healthy_service_mean: float
    service_time_cv: float
    degraded_slowdown: float
    hedge_delay_quantile: float
    target_max_utilization: float
    healthy_offered_load: float
    softmax_inverse_temperature: float
    policy_damping: float
    price_step: float
    price_damping: float
    tokens_per_run: int
    base_seed: int
    seed_count: int
    smoke_tokens: int
    degraded_start: float
    failed_start: float
    recovered_start: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CommonStateExperimentConfig":
        expected = {item.name for item in fields(cls)}
        supplied = set(raw)
        missing = expected - supplied
        unknown = supplied - expected
        if missing:
            raise ValueError(f"Missing configuration keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        coerced: dict[str, Any] = {}
        for item in fields(cls):
            value = raw[item.name]
            if item.name == "experiment_name":
                if not isinstance(value, str):
                    raise ValueError("experiment_name must be a string")
                coerced[item.name] = value
            elif item.name in _INT_CONFIG_FIELDS:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{item.name} must be an int, got {value!r}")
                coerced[item.name] = value
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{item.name} must be a real number, got {value!r}"
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{item.name} must be finite, got {value!r}")
                coerced[item.name] = number
        config = cls(**coerced)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        for name, expected in (
            ("expert_count", 1),
            ("replicas_per_expert", 2),
            ("failure_domain_count", 2),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected} for this slice")
        # Base constraints identical to schema 1.
        self.base_config()
        # Timeline ordering and finiteness.
        self.timeline()

    def base_config(self) -> ExperimentConfig:
        """Project the schema-1 base keys for workload generation."""
        base = {name: getattr(self, name) for name in _BASE_CONFIG_FIELD_NAMES}
        base["schema_version"] = 1
        config = ExperimentConfig(**base)
        config.validate()
        return config

    def timeline(self) -> CommonStateTimeline:
        return CommonStateTimeline(
            degraded_start=self.degraded_start,
            failed_start=self.failed_start,
            recovered_start=self.recovered_start,
        )


def parse_common_state_config_bytes(payload: bytes) -> CommonStateExperimentConfig:
    """Parse one immutable UTF-8 JSON payload into a schema-2 configuration."""
    try:
        raw = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("Configuration must be UTF-8 encoded") from exc
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration value must be an object")
    return CommonStateExperimentConfig.from_mapping(raw)


def load_common_state_config_with_sha256(
    path: str | Path,
) -> tuple[CommonStateExperimentConfig, str]:
    """Read once, then parse and hash the exact same configuration bytes."""
    payload = Path(path).read_bytes()
    return parse_common_state_config_bytes(payload), hashlib.sha256(payload).hexdigest()


_PAIRED_INT_FIELDS = _INT_CONFIG_FIELDS
_PAIRED_NEW_FLOAT_FIELDS = (
    "degraded_start",
    "failed_start",
    "recovered_start",
    "control_window",
    "storm_bin_width",
    "replay_penalty_regular",
    "replay_penalty_urgent",
    "incremental_work_cost",
    "wasted_work_cost",
)


@dataclass(frozen=True)
class PairedExperimentConfig:
    """Self-contained paired configuration (legacy schema 3 or cost schema 4)."""

    schema_version: int
    experiment_name: str
    expert_count: int
    replicas_per_expert: int
    failure_domain_count: int
    regular_token_ratio: float
    urgent_token_ratio: float
    healthy_service_mean: float
    service_time_cv: float
    degraded_slowdown: float
    hedge_delay_quantile: float
    target_max_utilization: float
    healthy_offered_load: float
    softmax_inverse_temperature: float
    policy_damping: float
    price_step: float
    price_damping: float
    tokens_per_run: int
    base_seed: int
    seed_count: int
    smoke_tokens: int
    degraded_start: float
    failed_start: float
    recovered_start: float
    control_window: float
    storm_bin_width: float
    replay_penalty_regular: float
    replay_penalty_urgent: float
    incremental_work_cost: float = 0.0
    wasted_work_cost: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PairedExperimentConfig":
        schema_version = raw.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in (3, 4)
        ):
            raise ValueError(f"Unsupported schema_version: {schema_version!r}")
        cost_fields = {"incremental_work_cost", "wasted_work_cost"}
        expected = {item.name for item in fields(cls)}
        if schema_version == 3:
            expected -= cost_fields
        supplied = set(raw)
        missing = expected - supplied
        unknown = supplied - expected
        if missing:
            raise ValueError(f"Missing configuration keys: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        coerced: dict[str, Any] = {}
        for item in fields(cls):
            if item.name in cost_fields and schema_version == 3:
                coerced[item.name] = 0.0
                continue
            value = raw[item.name]
            if item.name == "experiment_name":
                if not isinstance(value, str):
                    raise ValueError("experiment_name must be a string")
                coerced[item.name] = value
            elif item.name in _PAIRED_INT_FIELDS:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{item.name} must be an int, got {value!r}")
                coerced[item.name] = value
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{item.name} must be a real number, got {value!r}"
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"{item.name} must be finite, got {value!r}")
                coerced[item.name] = number
        config = cls(**coerced)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version not in (3, 4):
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")
        for name, expected in (
            ("expert_count", 1),
            ("replicas_per_expert", 2),
            ("failure_domain_count", 2),
        ):
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected} for this slice")
        self.base_config()
        self.timeline()
        for name in ("control_window", "storm_bin_width"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in ("incremental_work_cost", "wasted_work_cost"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if self.schema_version == 4 and value <= 0.0:
                raise ValueError(f"schema 4 requires {name} to be positive")
            if self.schema_version == 3 and value != 0.0:
                raise ValueError(f"schema 3 requires legacy zero {name}")

    def base_config(self) -> ExperimentConfig:
        """Project the schema-1 base keys used by calibration and workload."""
        base = {name: getattr(self, name) for name in _BASE_CONFIG_FIELD_NAMES}
        base["schema_version"] = 1
        config = ExperimentConfig(**base)
        config.validate()
        return config

    def timeline(self) -> CommonStateTimeline:
        return CommonStateTimeline(
            degraded_start=self.degraded_start,
            failed_start=self.failed_start,
            recovered_start=self.recovered_start,
        )


def parse_paired_config_bytes(payload: bytes) -> PairedExperimentConfig:
    """Parse one immutable UTF-8 JSON payload into a paired configuration."""
    try:
        raw = json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("Configuration must be UTF-8 encoded") from exc
    if not isinstance(raw, dict):
        raise ValueError("Top-level configuration value must be an object")
    return PairedExperimentConfig.from_mapping(raw)


def load_paired_config_with_sha256(
    path: str | Path,
) -> tuple[PairedExperimentConfig, str]:
    """Read once, then parse and hash the exact same configuration bytes."""
    payload = Path(path).read_bytes()
    return parse_paired_config_bytes(payload), hashlib.sha256(payload).hexdigest()
