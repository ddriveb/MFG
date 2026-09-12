"""Red tests for the restored causal Token entry and Reservation contract."""

from dataclasses import FrozenInstanceError, replace
import copy
import math
import unittest

from mfg_hedge.attribution_episode import (
    EpisodeIdentity,
    EpisodeProtocol,
    EpisodeTrace,
)
from mfg_hedge.common_state import CommonState, CommonStateTimeline, Phase
from mfg_hedge.domain import ProtectionAction as A, TokenClass as K
from mfg_hedge.hedge_simulation import simulate_hedge_common_state
from mfg_hedge.transient_control import (
    ReservationParameters,
    TimeClassRule,
    simulate_transient_episode,
)
from mfg_hedge.workload import TokenSpec, WorkloadTrace
from mfg_hedge.token_online import (
    OnlineAdmissionDecision,
    ReservationLedger,
    StaticActionSource,
    StaticTimeClassActionSource,
    TokenActionSource,
    TokenObservation,
    simulate_episode_online,
)


N, D, I = A.NORMAL, A.DELAYED_HEDGE, A.IMMEDIATE_HEDGE
TL = CommonStateTimeline(10.0, 20.0, 30.0)
RESERVATION_TL = CommonStateTimeline(0.0, 100.0, 120.0)


def episode(times=(10.0, 10.5, 11.0), *, timeline=TL, cutoff=40.0,
           work=1.0, classes=None, base_seed=1):
    classes = classes or [K.REGULAR] * len(times)
    tokens = tuple(TokenSpec(i, float(t), classes[i]) for i, t in enumerate(times))
    stream = (float(work),) * len(tokens)
    replay = ((float(work),) * len(tokens),) * 2
    hedge = ((float(work),) * len(tokens),) * 2
    trace = WorkloadTrace(base_seed, 1.7, tokens, (stream, stream), replay, hedge)
    return EpisodeTrace(
        EpisodeIdentity("transient-control:v1:test", base_seed, 0),
        base_seed,
        EpisodeProtocol(timeline, cutoff),
        trace,
    )


class _Source:
    def __init__(self, action=N, *, reset=False):
        self.action = action
        self.observations = []
        self.keys = []
        self.calls = 0
        self.reset = reset

    def new_episode(self):
        if not self.reset:
            return self
        return _Source(self.action, reset=True)

    def choose(self, observation, policy_key):
        self.calls += 1
        self.observations.append(observation)
        self.keys.append(policy_key)
        return self.action


class TokenOnlineContractTests(unittest.TestCase):
    def test_observation_is_frozen_and_has_no_hidden_physics(self):
        observation = TokenObservation(
            token_id=0,
            token_class=K.REGULAR,
            arrival_time=10.0,
            phase=Phase.DEGRADED,
            phase_age=0.0,
            primary_replica=0,
            queue_snapshot=(((0, 0),), ()),
            running_attempts=(),
            observed_history=(),
            reservation_window=(10.0, 20.0),
            reservation_cap=2.8125,
            reservation_balance=2.8125,
            public_price=0.0,
        )
        self.assertTrue(TokenObservation.__dataclass_params__.frozen)
        names = {field.name for field in TokenObservation.__dataclass_fields__.values()}
        for forbidden in ("trace", "future_fault", "service_draw", "remaining_work",
                          "future_arrivals", "future_winner", "future_replay"):
            self.assertNotIn(forbidden, names)
        self.assertIs(observation.common_state, CommonState.DEGRADED)
        with self.assertRaises(FrozenInstanceError):
            observation.token_id = 2

    def test_same_observable_prefix_has_same_action_under_hidden_future(self):
        first = TokenObservation(
            0, K.REGULAR, 10.0, Phase.DEGRADED, 1.0, 0,
            (((0, 0),), ()), (), ("D-observed",), (10.0, 20.0),
            2.8125, 1.8125, 0.0,
        )
        second = replace(first)
        source = _Source(I)
        self.assertEqual(source.choose(first, "episode:0:token:0"),
                         source.choose(second, "episode:0:token:0"))

    def test_unrevealed_failure_time_does_not_change_current_observation(self):
        observations = []

        class Recorder:
            def choose(self, observation, policy_key):
                observations.append(observation)
                return N

        simulate_episode_online(
            episode((14.0,), timeline=CommonStateTimeline(0.0, 15.0, 30.0), cutoff=40.0),
            Recorder(),
        )
        simulate_episode_online(
            episode((14.0,), timeline=CommonStateTimeline(0.0, 20.0, 30.0), cutoff=40.0),
            Recorder(),
        )
        self.assertEqual(observations[0], observations[1])

    def test_each_token_calls_choose_once(self):
        source = _Source(N)
        result = simulate_episode_online(episode(), source)
        self.assertEqual(len(result.decisions), 3)
        self.assertEqual(source.calls, 3)
        self.assertEqual(len(set(source.keys)), 3)

    def test_delayed_timer_does_not_call_policy_again(self):
        source = _Source(D)
        result = simulate_episode_online(episode((10.0,)), source, hedge_delay=2.0)
        self.assertEqual(source.calls, 1)
        self.assertEqual(result.simulation.simulation.hedge_timers_voided, 1)

    def test_invalid_actions_bool_string_and_none_fail(self):
        for invalid in (True, "I", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    simulate_episode_online(episode((10.0,)), _Source(invalid))

    def test_replica_id_rejects_bool_and_float_at_all_online_entry_points(self):
        invalid_ids = (False, True, 0.0, 1.0)
        for invalid in invalid_ids:
            with self.subTest(entry="observation", value=invalid):
                with self.assertRaises(ValueError):
                    TokenObservation(
                        token_id=0,
                        token_class=K.REGULAR,
                        arrival_time=10.0,
                        phase=Phase.DEGRADED,
                        phase_age=0.0,
                        primary_replica=invalid,
                        queue_snapshot=(((0, 0),), ()),
                        running_attempts=(),
                        observed_history=(),
                        reservation_window=(10.0, 20.0),
                        reservation_cap=2.8125,
                        reservation_balance=2.8125,
                        public_price=0.0,
                    )
            with self.subTest(entry="admission", value=invalid):
                with self.assertRaises(ValueError):
                    OnlineAdmissionDecision(
                        token_id=0,
                        arrival_time=10.0,
                        token_class=K.REGULAR,
                        phase=Phase.DEGRADED,
                        primary_replica=invalid,
                        requested=N,
                        applied=N,
                        window_start=10.0,
                        window_end=20.0,
                        cap=2.8125,
                        balance_before=2.8125,
                        charge=0.0,
                        balance_after=2.8125,
                        reservation_suppressed=False,
                        reservation_admitted=True,
                    )
            with self.subTest(entry="ledger", value=invalid):
                with self.assertRaises(ValueError):
                    ReservationLedger(RESERVATION_TL).preview(
                        10.0, K.REGULAR, invalid
                    )

    def test_stateful_source_is_reset_for_each_episode(self):
        class Stateful:
            def __init__(self, count=0):
                self.count = count

            def new_episode(self):
                return Stateful()

            def choose(self, observation, policy_key):
                action = I if self.count == 0 else N
                self.count += 1
                return action

        source = Stateful()
        first = simulate_episode_online(episode((10.0,)), source)
        second = simulate_episode_online(episode((10.0,)), source)
        self.assertEqual(first.decisions[0].requested, I)
        self.assertEqual(second.decisions[0].requested, I)
        self.assertEqual(source.count, 0)

    def test_stateless_source_remains_compatible(self):
        class Stateless:
            def choose(self, observation, policy_key):
                return N

        result = simulate_episode_online(episode((10.0,)), Stateless())
        self.assertEqual(result.decisions[0].applied, N)

    def test_reservation_admits_delayed_and_immediate_with_one_charge(self):
        ledger = ReservationLedger(RESERVATION_TL)
        delayed = ledger.admit(0, 10.0, K.REGULAR, 0, D)
        immediate = ledger.admit(2, 10.1, K.REGULAR, 0, I)
        self.assertEqual((delayed.applied, immediate.applied), (D, I))
        self.assertEqual((delayed.charge, immediate.charge), (1.0, 1.0))
        self.assertAlmostEqual(immediate.balance_after, 0.8125)

    def test_reservation_suppresses_without_partial_downgrade(self):
        ledger = ReservationLedger(RESERVATION_TL)
        decisions = [ledger.admit(i, 10.0 + i * .01, K.REGULAR, 0, I)
                     for i in (0, 2, 4)]
        self.assertEqual([decision.applied for decision in decisions], [I, I, N])
        self.assertEqual(decisions[-1].requested, I)
        self.assertTrue(decisions[-1].reservation_suppressed)
        self.assertEqual(decisions[-1].charge, 0.0)

    def test_denied_action_does_not_hedge_but_normal_replay_remains(self):
        result = simulate_episode_online(
            episode((10.0, 10.1, 10.2, 10.3, 19.0), work=4.0),
            _Source(I),
            hedge_delay=2.0,
        )
        decision = result.decisions[4]
        self.assertEqual((decision.requested, decision.applied), (I, N))
        self.assertTrue(decision.reservation_suppressed)
        self.assertFalse(any(
            attempt.attempt_id == 2
            for attempt in result.simulation.simulation.tokens[4].attempts
        ))
        self.assertEqual(result.simulation.simulation.replay_executions, 1)

    def test_admitted_delayed_timer_void_keeps_charge(self):
        result = simulate_episode_online(
            episode((10.0,), timeline=RESERVATION_TL, cutoff=140.0,
                    work=.25), _Source(D), hedge_delay=2.0
        )
        decision = result.decisions[0]
        self.assertEqual(decision.applied, D)
        self.assertEqual(decision.charge, 1.0)
        self.assertEqual(decision.timer_status, "voided")
        self.assertAlmostEqual(decision.balance_after, 1.8125)

    def test_audit_separates_requested_applied_and_two_suppressions(self):
        result = simulate_episode_online(
            episode((10.0, 10.1, 20.0), work=4.0), _Source(D), hedge_delay=2.0
        )
        audit = result.audit()
        self.assertEqual(
            audit["requested_hedges"],
            audit["applied_hedges"] + audit["reservation_suppressed"],
        )
        self.assertGreaterEqual(audit["executor_suppressed"], 0)
        self.assertIn("timer_voided", audit)

    def test_arrival_observation_precedes_dispatch(self):
        source = _Source(N)
        simulate_episode_online(episode((10.0, 10.0)), source)
        self.assertEqual(source.observations[0].queue_snapshot, ((), ()))
        self.assertEqual(source.observations[1].queue_snapshot[0], ((0, 0),))

    def test_online_policy_can_react_to_prior_queue_change(self):
        class QueuePolicy:
            def choose(self, observation, policy_key):
                return D if observation.queue_snapshot[0] else N

        result = simulate_episode_online(
            episode((10.0, 10.0)), QueuePolicy(), hedge_delay=2.0
        )
        self.assertEqual([row.requested for row in result.decisions], [N, D])

    def test_callback_receives_no_trace_or_service_state(self):
        class Inspecting:
            def choose(self, observation, policy_key):
                self.assert_no_hidden(observation)
                return N

            @staticmethod
            def assert_no_hidden(observation):
                for name in dir(observation):
                    self_name = name.lower()
                    if any(word in self_name for word in ("trace", "service_draw", "remaining_work")):
                        raise AssertionError(name)

        simulate_episode_online(episode((10.0,)), Inspecting())

    def test_policy_randomness_does_not_change_workload_or_crn_result(self):
        before = episode((10.0,))
        snapshot = copy.deepcopy(before)

        class Noisy:
            def choose(self, observation, policy_key):
                for _ in range(100):
                    math.sin(_)
                return N

        online = simulate_episode_online(before, Noisy())
        self.assertEqual(before, snapshot)
        self.assertEqual(online.simulation.simulation,
                         simulate_hedge_common_state(
                             before.workload, TL, 2.0, actions={0: N}
                         ))

    def test_input_and_static_action_mapping_are_unchanged(self):
        e = episode((10.0, 10.5))
        original = copy.deepcopy(e)
        static = simulate_hedge_common_state(e.workload, TL, 2.0,
                                             actions={0: I})
        online = simulate_episode_online(e, StaticActionSource({0: I}))
        self.assertEqual(online.simulation.simulation, static)
        self.assertEqual(e, original)

    def test_repeated_online_run_is_exactly_deterministic(self):
        first = simulate_episode_online(episode((10.0, 10.5)), _Source(D), hedge_delay=2)
        second = simulate_episode_online(episode((10.0, 10.5)), _Source(D), hedge_delay=2)
        self.assertEqual(first, second)

    def test_non_historical_arrival_rate_is_not_hardcoded(self):
        e = episode((1.0, 2.0), timeline=CommonStateTimeline(5, 10, 15), cutoff=20)
        result = simulate_episode_online(e, _Source(N))
        self.assertEqual(result.simulation.simulation.completed_tokens, 2)

    def test_static_time_class_adapter_matches_legacy_niin_path(self):
        equivalent_timeline = CommonStateTimeline(0.0, 200.0, 220.0)
        e = episode(
            (99.0, 100.0, 100.1, 150.0, 151.0),
            timeline=equivalent_timeline,
            cutoff=320.0,
            work=.25,
        )
        params = ReservationParameters()
        rule = TimeClassRule(D, I, I, D)
        legacy = simulate_transient_episode(e, rule, params, hedge_delay=2.0)
        online = simulate_episode_online(
            e,
            StaticTimeClassActionSource(rule, late_after=100.0),
            params,
            hedge_delay=2.0,
        )
        self.assertEqual(online.simulation.simulation, legacy.run.simulation)
        self.assertEqual(
            [(row.requested, row.applied, row.charge) for row in online.decisions],
            [(row.requested, row.applied, row.charge) for row in legacy.plan.decisions],
        )


if __name__ == "__main__":
    unittest.main()
