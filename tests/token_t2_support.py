"""Small deterministic fixtures shared by the T2 payoff/deviation tests."""

from mfg_hedge.attribution_episode import EpisodeIdentity, EpisodeProtocol, EpisodeTrace
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.workload import TokenSpec, WorkloadTrace


def make_episode(
    arrivals=(10.0,),
    *,
    timeline=CommonStateTimeline(0.0, 100.0, 200.0),
    cutoff=300.0,
    classes=None,
    service0=None,
    service1=None,
    replay0=None,
    replay1=None,
    hedge0=None,
    hedge1=None,
    base_seed=7001,
):
    count = len(arrivals)
    classes = tuple(classes or (TokenClass.REGULAR,) * count)
    service0 = tuple(service0 or (1.0,) * count)
    service1 = tuple(service1 or (1.0,) * count)
    replay0 = tuple(replay0 or (1.0,) * count)
    replay1 = tuple(replay1 or (1.0,) * count)
    hedge0 = tuple(hedge0 or (1.0,) * count)
    hedge1 = tuple(hedge1 or (1.0,) * count)
    tokens = tuple(
        TokenSpec(index, float(arrival), classes[index])
        for index, arrival in enumerate(arrivals)
    )
    trace = WorkloadTrace(
        base_seed,
        1.7,
        tokens,
        (service0, service1),
        (replay0, replay1),
        (hedge0, hedge1),
    )
    return EpisodeTrace(
        EpisodeIdentity("token-mfg-restoration:t2:test", base_seed, 0),
        base_seed,
        EpisodeProtocol(timeline, cutoff),
        trace,
    )


class FixedSource:
    def __init__(self, action=ProtectionAction.NORMAL, chooser=None):
        self.action = action
        self.chooser = chooser
        self.calls = []

    def choose(self, observation, policy_key):
        self.calls.append((observation, policy_key))
        if self.chooser is not None:
            return self.chooser(observation, policy_key)
        return self.action


def source_factory(action=ProtectionAction.NORMAL, chooser=None, sink=None):
    def factory():
        source = FixedSource(action, chooser)
        if sink is not None:
            sink.append(source)
        return source

    return factory
