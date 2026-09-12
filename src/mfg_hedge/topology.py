"""Immutable finite Replica topologies used by the K-scale routing slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


QUALIFICATION_K = (8, 16, 32, 64)
FAILURE_DOMAIN_COUNT = 4


@dataclass(frozen=True)
class Topology:
    """One Logical Expert with K equivalent Replicas in four domains."""

    K: int

    def __post_init__(self) -> None:
        if type(self.K) is not int or self.K not in QUALIFICATION_K:
            raise ValueError("K must be one of the supported true integers 8, 16, 32, 64")
        if self.K % FAILURE_DOMAIN_COUNT:
            raise ValueError("K must be divisible by four")

    @property
    def replica_ids(self) -> tuple[int, ...]:
        return tuple(range(self.K))

    @property
    def domains(self) -> tuple[int, ...]:
        return tuple(replica_id % FAILURE_DOMAIN_COUNT for replica_id in self.replica_ids)

    @property
    def domain_ids(self) -> tuple[int, ...]:
        return tuple(range(FAILURE_DOMAIN_COUNT))

    @property
    def replicas_per_domain(self) -> int:
        return self.K // FAILURE_DOMAIN_COUNT

    def domain_of(self, replica_id: int) -> int:
        if type(replica_id) is not int or replica_id not in self.replica_ids:
            raise ValueError("replica_id is outside the topology")
        return replica_id % FAILURE_DOMAIN_COUNT

    def validate_domain(self, domain_id: int) -> int:
        if type(domain_id) is not int or domain_id not in self.domain_ids:
            raise ValueError("domain_id is outside the topology")
        return domain_id

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "K": self.K,
                "replica_ids": self.replica_ids,
                "domains": self.domains,
                "domain_ids": self.domain_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


DEFAULT_TOPOLOGY = Topology(8)


__all__ = [
    "DEFAULT_TOPOLOGY",
    "FAILURE_DOMAIN_COUNT",
    "QUALIFICATION_K",
    "Topology",
]
