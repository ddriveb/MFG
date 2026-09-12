"""Attribution episode and macro-seed metric tests."""

from dataclasses import replace
import unittest

from mfg_hedge.attribution_episode import (
    BoundarySnapshot,
    EpisodeIdentity,
    EpisodeProtocol,
    EpisodeTrace,
    simulate_episode,
)
from mfg_hedge.attribution_metrics import (
    SLODeadlines,
    build_episode_metrics,
    build_macro_seed_metrics,
    empirical_cvar95,
)
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.domain import ProtectionAction, TokenClass
from mfg_hedge.workload import TokenSpec, WorkloadTrace


TIMELINE = CommonStateTimeline(10.0, 20.0, 30.0)
PROTOCOL = EpisodeProtocol(TIMELINE, arrival_cutoff=40.0)
SLOWDOWN = 2.0
NAMESPACE = "attribution:v1:development"


def wrap_trace(trace: WorkloadTrace, episode_index: int = 0, cutoff: float = 40.0):
    return EpisodeTrace(
        identity=EpisodeIdentity(NAMESPACE, macro_seed=7, episode_index=episode_index),
        episode_seed=trace.base_seed,
        protocol=EpisodeProtocol(TIMELINE, arrival_cutoff=cutoff),
        workload=trace,
    )


def make_trace(arrivals, classes, service_r0, service_r1, replay_r1):
    count = len(arrivals)
    filler = (9.0,) * count
    return WorkloadTrace(
        base_seed=100 + count,
        arrival_rate=1.0,
        tokens=tuple(
            TokenSpec(i, arrival, token_class)
            for i, (arrival, token_class) in enumerate(zip(arrivals, classes))
        ),
        service_times=(tuple(service_r0), tuple(service_r1)),
        replay_service_times=(filler, tuple(replay_r1)),
        hedge_service_times=(filler, filler),
    )


class EmpiricalCvarTests(unittest.TestCase):
    def test_top_ceil_five_percent_is_averaged(self) -> None:
        self.assertEqual(empirical_cvar95(range(1, 21)), 20.0)
        self.assertEqual(empirical_cvar95(range(1, 22)), 20.5)
        self.assertEqual(empirical_cvar95([4.0]), 4.0)

    def test_empty_bool_and_non_finite_samples_are_rejected(self) -> None:
        for values in ((), (True,), (1.0, float("nan")), (1.0, float("inf"))):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    empirical_cvar95(values)


class EpisodeMetricTests(unittest.TestCase):
    def test_phase_class_slo_migration_and_cross_phase_work(self) -> None:
        trace = make_trace(
            arrivals=(9.0, 9.5),
            classes=(TokenClass.REGULAR, TokenClass.URGENT),
            service_r0=(2.0, 9.0),
            service_r1=(9.0, 4.0),
            replay_r1=(9.0, 9.0),
        )
        run = simulate_episode(wrap_trace(trace), SLOWDOWN)
        metrics = build_episode_metrics(run, SLOWDOWN)

        overall = metrics["latency"]["overall"]
        self.assertEqual(overall["sample_count"], 2)
        self.assertAlmostEqual(overall["mean_latency"], 3.5)
        self.assertAlmostEqual(overall["cvar95_latency"], 4.0)
        self.assertEqual(overall["deadline_miss_count"], 1)
        self.assertAlmostEqual(overall["deadline_miss_probability"], 0.5)
        self.assertAlmostEqual(overall["mean_excess_latency"], 1.0)

        h_regular = metrics["latency"]["arrival_phase_class"]["H"]["R"]
        h_urgent = metrics["latency"]["arrival_phase_class"]["H"]["U"]
        self.assertAlmostEqual(h_regular["mean_latency"], 3.0)
        self.assertEqual(h_regular["deadline_miss_count"], 0)
        self.assertAlmostEqual(h_urgent["mean_excess_latency"], 2.0)
        self.assertEqual(metrics["migration_matrix"]["H"]["D"], 2)

        split = metrics["work"]["by_execution_phase_and_replica"]
        self.assertAlmostEqual(split["H"]["0"]["primary"], 1.0)
        self.assertAlmostEqual(split["D"]["0"]["primary"], 1.0)
        self.assertAlmostEqual(split["H"]["1"]["primary"], 0.5)
        self.assertAlmostEqual(split["D"]["1"]["primary"], 3.5)
        self.assertAlmostEqual(metrics["work"]["total_executed_work"], 6.0)
        self.assertTrue(all(metrics["invariants"].values()))

    def test_same_time_failure_snapshot_precedes_arrival_and_dispatch(self) -> None:
        trace = make_trace(
            arrivals=(18.0, 20.0),
            classes=(TokenClass.REGULAR, TokenClass.URGENT),
            service_r0=(100.0, 9.0),
            service_r1=(9.0, 1.0),
            replay_r1=(3.0, 9.0),
        )
        run = simulate_episode(wrap_trace(trace), SLOWDOWN)
        snapshot = run.failed_start_snapshot
        self.assertEqual(snapshot.queued_attempts, (0, 1))
        self.assertEqual(snapshot.live_attempts, (0, 1))
        self.assertEqual(snapshot.remaining_work, (0.0, 3.0))

    def test_metric_contract_rejects_a_different_simulation_slowdown(self) -> None:
        trace = make_trace(
            arrivals=(9.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(2.0,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        run = simulate_episode(wrap_trace(trace), SLOWDOWN)
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_episode_metrics(run, 4.0)

    def test_boundary_snapshot_rejects_impossible_or_nonfinite_state(self) -> None:
        invalid = (
            ((0, -1), (0, 0), (0.0, 0.0)),
            ((0, 0), (0, 2), (0.0, 1.0)),
            ((0, 0), (0, 0), (0.0, float("inf"))),
        )
        for queued, live, remaining in invalid:
            with self.subTest(queued=queued, live=live, remaining=remaining):
                with self.assertRaises(ValueError):
                    BoundarySnapshot(queued, live, remaining)

    def test_corrupt_engine_counters_and_trace_pairing_fail_fast(self) -> None:
        trace = make_trace(
            arrivals=(9.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(2.0,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        run = simulate_episode(wrap_trace(trace), SLOWDOWN)
        corrupt = replace(
            run, simulation=replace(run.simulation, primary_executions=999)
        )
        with self.assertRaisesRegex(RuntimeError, "primary_execution_counter"):
            build_episode_metrics(corrupt, SLOWDOWN)

        different_trace = make_trace(
            arrivals=(8.0,),
            classes=(TokenClass.URGENT,),
            service_r0=(3.0,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        mismatched = replace(run, episode=wrap_trace(different_trace))
        with self.assertRaisesRegex(RuntimeError, "result_tokens_match"):
            build_episode_metrics(mismatched, SLOWDOWN)

        broken_attempt = replace(
            run.simulation.attempts[0], remaining_work=-123.0
        )
        broken_token = replace(
            run.simulation.tokens[0], attempts=(broken_attempt,)
        )
        broken_result = replace(
            run.simulation,
            tokens=(broken_token,),
            attempts=(broken_attempt,),
        )
        with self.assertRaisesRegex(RuntimeError, "attempt_record_algebra"):
            build_episode_metrics(
                replace(run, simulation=broken_result), SLOWDOWN
            )

        shifted_attempt = replace(
            run.simulation.attempts[0], enqueue_time=0.0, queue_delay=9.0
        )
        shifted_token = replace(
            run.simulation.tokens[0], attempts=(shifted_attempt,)
        )
        shifted_result = replace(
            run.simulation,
            tokens=(shifted_token,),
            attempts=(shifted_attempt,),
        )
        with self.assertRaisesRegex(RuntimeError, "per_token_attempt_algebra"):
            build_episode_metrics(
                replace(run, simulation=shifted_result), SLOWDOWN
            )

    def test_storm_uses_fixed_episode_clock_and_hedge_specific_statuses(self) -> None:
        trace = make_trace(
            arrivals=(1.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(2.0,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        trace = replace(trace, hedge_service_times=((9.0,), (1.0,)))
        run = simulate_episode(
            wrap_trace(trace),
            SLOWDOWN,
            actions={0: ProtectionAction.IMMEDIATE_HEDGE},
        )
        metrics = build_episode_metrics(run, SLOWDOWN)
        self.assertEqual(run.simulation.completed_loser_total, 1)
        self.assertEqual(metrics["hedge"]["winners"], 1)
        self.assertEqual(metrics["hedge"]["completed_loser"], 0)
        self.assertAlmostEqual(metrics["storm"]["mean_hedge_launch_rate"], 0.025)
        self.assertAlmostEqual(
            metrics["storm"]["peak_to_mean_hedge_launch_ratio"], 40.0
        )

    def test_dispatcher_assignment_cannot_be_moved_between_equal_draws(self) -> None:
        trace = make_trace(
            arrivals=(1.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(1.0,),
            service_r1=(1.0,),
            replay_r1=(9.0,),
        )
        run = simulate_episode(wrap_trace(trace), SLOWDOWN)
        moved_attempt = replace(run.simulation.attempts[0], replica_id=1)
        moved_token = replace(
            run.simulation.tokens[0],
            primary_replica=1,
            attempts=(moved_attempt,),
        )
        moved_result = replace(
            run.simulation,
            tokens=(moved_token,),
            attempts=(moved_attempt,),
        )
        with self.assertRaisesRegex(RuntimeError, "dispatcher_assignment"):
            build_episode_metrics(
                replace(run, simulation=moved_result), SLOWDOWN
            )

    def test_storm_overload_is_per_replica_and_includes_degraded_capacity(self) -> None:
        trace = make_trace(
            arrivals=(12.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(1.0,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        metrics = build_episode_metrics(
            simulate_episode(wrap_trace(trace), SLOWDOWN), SLOWDOWN
        )
        storm = metrics["storm"]
        self.assertEqual(storm["admitted_load_ratio_bins_by_replica"][0][12], 2.0)
        self.assertTrue(storm["overloaded_bins_by_replica"][0][12])
        self.assertEqual(storm["sustained_overload_duration"], 1.0)

    def test_recovered_window_utilization_excludes_separate_drain_work(self) -> None:
        trace = make_trace(
            arrivals=(39.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(1.0,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        trace = replace(trace, hedge_service_times=((9.0,), (10.0,)))
        metrics = build_episode_metrics(
            simulate_episode(
                wrap_trace(trace),
                SLOWDOWN,
                actions={0: ProtectionAction.IMMEDIATE_HEDGE},
            ),
            SLOWDOWN,
        )
        r0 = metrics["work"]["by_execution_phase_and_replica"]["R"]["0"]
        r1 = metrics["work"]["by_execution_phase_and_replica"]["R"]["1"]
        self.assertEqual(r0["fixed_window_primary"], 1.0)
        self.assertEqual(r1["fixed_window_hedge"], 1.0)
        self.assertEqual(r1["post_cutoff_hedge"], 9.0)
        self.assertEqual(r1["fixed_window_duration"], 10.0)
        self.assertEqual(
            metrics["work"]["post_cutoff_by_replica"]["1"]["total"], 9.0
        )

    def test_boundary_remaining_work_and_recovery_metrics(self) -> None:
        trace = make_trace(
            arrivals=(18.0, 18.5),
            classes=(TokenClass.REGULAR, TokenClass.URGENT),
            service_r0=(100.0, 9.0),
            service_r1=(9.0, 20.0),
            replay_r1=(3.0, 9.0),
        )
        run = simulate_episode(wrap_trace(trace), SLOWDOWN)
        metrics = build_episode_metrics(run, SLOWDOWN)

        failed = metrics["boundaries"]["failed_start"]
        recovered = metrics["boundaries"]["recovered_start"]
        self.assertEqual(failed["queued_attempts"], [0, 1])
        self.assertEqual(failed["live_attempts"], [0, 2])
        self.assertAlmostEqual(failed["remaining_work"][1], 21.5)
        self.assertEqual(recovered["live_attempts"], [0, 2])
        self.assertAlmostEqual(recovered["remaining_work"][1], 11.5)

        recovery = metrics["recovery"]
        self.assertEqual(recovery["pre_recovery_arrival_count"], 2)
        self.assertAlmostEqual(recovery["logical_backlog_area"], 20.0)
        self.assertAlmostEqual(recovery["logical_clear_time"], 41.5)
        self.assertAlmostEqual(recovery["logical_clear_delay"], 11.5)
        self.assertAlmostEqual(recovery["resource_clear_time"], 41.5)
        self.assertAlmostEqual(metrics["post_cutoff_drain_duration"], 1.5)

        bad_snapshot = replace(
            run.failed_start_snapshot, remaining_work=(0.0, 999.0)
        )
        with self.assertRaisesRegex(RuntimeError, "failed_boundary_snapshot"):
            build_episode_metrics(
                replace(run, failed_start_snapshot=bad_snapshot), SLOWDOWN
            )

    def test_deadlines_are_strict_and_validated(self) -> None:
        self.assertEqual(SLODeadlines().for_class(TokenClass.REGULAR), 3.0)
        self.assertEqual(SLODeadlines().for_class(TokenClass.URGENT), 2.0)
        for regular, urgent in ((True, 2.0), (3.0, 0.0), (3.0, float("inf"))):
            with self.subTest(regular=regular, urgent=urgent):
                with self.assertRaises(ValueError):
                    SLODeadlines(regular=regular, urgent=urgent)


class MacroSeedAggregationTests(unittest.TestCase):
    def _run(self, episode_index: int, duration: float):
        trace = make_trace(
            arrivals=(1.0,),
            classes=(TokenClass.REGULAR,),
            service_r0=(duration,),
            service_r1=(9.0,),
            replay_r1=(9.0,),
        )
        trace = WorkloadTrace(
            base_seed=200 + episode_index,
            arrival_rate=trace.arrival_rate,
            tokens=trace.tokens,
            service_times=trace.service_times,
            replay_service_times=trace.replay_service_times,
            hedge_service_times=((9.0,), (duration,)),
        )
        return simulate_episode(
            wrap_trace(trace, episode_index=episode_index),
            SLOWDOWN,
            actions={0: ProtectionAction.IMMEDIATE_HEDGE},
        )

    def test_tokens_pool_but_episode_clocks_never_stack(self) -> None:
        runs = tuple(
            self._run(index, 100.0 if index == 0 else 1.0)
            for index in range(50)
        )
        macro = build_macro_seed_metrics(runs, SLOWDOWN)
        self.assertEqual(macro["inferential_unit"], "macro_seed")
        self.assertEqual(macro["inferential_n"], 1)
        self.assertEqual(macro["episode_count"], 50)
        self.assertEqual(macro["latency"]["overall"]["cvar95_latency"], 34.0)
        # All independent episodes launch once in the same local bin. Stacking
        # their clocks would report 50.0; episode-first averaging reports 1.0.
        self.assertEqual(
            macro["episode_time_means"]["peak_hedge_launch_rate"], 1.0
        )
        self.assertAlmostEqual(
            macro["episode_time_means"]["post_cutoff_drain_duration"],
            1.22,
        )
        self.assertEqual(
            macro["aggregation_contract"],
            "token_pool_rate_components_episode_time_mean",
        )
        self.assertEqual(macro, build_macro_seed_metrics(tuple(reversed(runs)), SLOWDOWN))

    def test_macro_requires_one_dense_complete_identity_group(self) -> None:
        runs = tuple(self._run(index, 1.0) for index in range(50))
        with self.assertRaisesRegex(ValueError, "expected 50 episodes"):
            build_macro_seed_metrics(runs[:-1], SLOWDOWN)
        with self.assertRaisesRegex(ValueError, "fixed at 50"):
            build_macro_seed_metrics(
                runs[:2], SLOWDOWN, expected_episode_count=2
            )
        other_macro = runs[1]
        changed = type(other_macro)(
            episode=EpisodeTrace(
                identity=EpisodeIdentity(NAMESPACE, 8, 1),
                episode_seed=other_macro.episode.episode_seed,
                protocol=other_macro.episode.protocol,
                workload=other_macro.episode.workload,
            ),
            simulation=other_macro.simulation,
            failed_start_snapshot=other_macro.failed_start_snapshot,
            recovered_start_snapshot=other_macro.recovered_start_snapshot,
            post_cutoff_drain_duration=other_macro.post_cutoff_drain_duration,
            degraded_slowdown=other_macro.degraded_slowdown,
            hedge_delay=other_macro.hedge_delay,
        )
        with self.assertRaisesRegex(ValueError, "macro_seed"):
            build_macro_seed_metrics(
                (runs[0], changed, *runs[2:]),
                SLOWDOWN,
            )

    def test_overload_intervals_are_never_joined_across_episode_clocks(self) -> None:
        runs = []
        for index in range(50):
            arrival = 39.0 if index == 0 else (0.0 if index == 1 else 1.0)
            work = 3.0 if index < 2 else 0.5
            trace = WorkloadTrace(
                base_seed=500 + index,
                arrival_rate=1.0,
                tokens=(TokenSpec(0, arrival, TokenClass.REGULAR),),
                service_times=((work,), (9.0,)),
                replay_service_times=((9.0,), (9.0,)),
                hedge_service_times=((9.0,), (9.0,)),
            )
            runs.append(
                simulate_episode(
                    EpisodeTrace(
                        EpisodeIdentity(NAMESPACE, 7, index),
                        500 + index,
                        PROTOCOL,
                        trace,
                    ),
                    SLOWDOWN,
                )
            )
        macro = build_macro_seed_metrics(tuple(runs), SLOWDOWN)
        self.assertAlmostEqual(
            macro["episode_time_means"]["sustained_overload_duration"],
            0.04,
        )
        self.assertEqual(
            macro["episode_time_means"][
                "peak_to_mean_ratio_undefined_episode_count"
            ],
            50,
        )

    def test_post_cutoff_timer_launches_remain_auditable_at_macro_level(self) -> None:
        runs = []
        for index in range(50):
            delayed = index == 0
            trace = WorkloadTrace(
                base_seed=600 + index,
                arrival_rate=1.0,
                tokens=(
                    TokenSpec(
                        0,
                        39.0 if delayed else 1.0,
                        TokenClass.REGULAR,
                    ),
                ),
                service_times=((10.0 if delayed else 1.0,), (9.0,)),
                replay_service_times=((9.0,), (9.0,)),
                hedge_service_times=((9.0,), (1.0,)),
            )
            runs.append(
                simulate_episode(
                    EpisodeTrace(
                        EpisodeIdentity(NAMESPACE, 7, index),
                        600 + index,
                        PROTOCOL,
                        trace,
                    ),
                    SLOWDOWN,
                    actions=(
                        {0: ProtectionAction.DELAYED_HEDGE}
                        if delayed
                        else None
                    ),
                    hedge_delay=2.0,
                )
            )
        macro = build_macro_seed_metrics(tuple(runs), SLOWDOWN)
        self.assertEqual(macro["hedge"]["launched"], 1)
        self.assertEqual(macro["storm"]["in_window_hedge_launch_count"], 0)
        self.assertEqual(macro["storm"]["post_cutoff_hedge_launch_count"], 1)
        self.assertTrue(macro["invariants"]["hedge_launches_partition_at_cutoff"])


if __name__ == "__main__":
    unittest.main()
