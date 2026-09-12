"""Eight-Expert / dual-Backup isolated development experiment tests."""

from dataclasses import replace
import copy
from pathlib import Path
import tempfile
import unittest

from mfg_hedge.attribution_episode import EpisodeIdentity, EpisodeProtocol
from mfg_hedge.common_state import CommonStateTimeline
from mfg_hedge.config import load_config
from mfg_hedge.domain import ProtectionAction as A, TokenClass as K
from mfg_hedge.hedge_simulation import HedgeAttemptStatus as S
from mfg_hedge.scaled_experiment import (
    SCALED_EPISODE_COUNT,
    SCALED_EXPERT_COUNT,
    SCALED_NAMESPACE,
    ScaledEpisodeTrace,
    ScaledTokenSpec,
    ScaledWorkDraw,
    evaluate_scaled_development,
    generate_scaled_episode,
    plan_scaled_actions,
    scaled_trace_fingerprint,
    simulate_scaled_episode,
    write_scaled_artifact,
)
from mfg_hedge.transient_control import ReservationParameters, TimeClassRule

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "v1_minimal.json")
CONFIG = replace(BASE, experiment_name="scaled-test", expert_count=8,
                 replicas_per_expert=3, failure_domain_count=3,
                 healthy_offered_load=.45)
TL = CommonStateTimeline(10, 20, 30)
PROTOCOL = EpisodeProtocol(TL, 40)
N, D, I = A.NORMAL, A.DELAYED_HEDGE, A.IMMEDIATE_HEDGE
FULL_CAP = ReservationParameters(budget_rate=.28125, scale=1)


def draw(primary=(1, 1, 1), replay=(1, 3, 3), hedge=(1, .5, .75),
         second=(1, .6, .8)):
    return ScaledWorkDraw(tuple(map(float, primary)), tuple(map(float, replay)),
                          tuple(map(float, hedge)), tuple(map(float, second)))


def manual_episode(rows, draws, index=0):
    local = [0] * 8
    tokens = []
    for token_id, (time, cls, expert) in enumerate(rows):
        tokens.append(ScaledTokenSpec(token_id, float(time), cls, expert, local[expert]))
        local[expert] += 1
    return ScaledEpisodeTrace(
        EpisodeIdentity("scaled-control:v1:test", 0, index), PROTOCOL,
        expert_count=8, arrival_rate=3.6, tokens=tuple(tokens), work=tuple(draws))


class GenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_complete_and_uniformly_scoped(self):
        first = generate_scaled_episode(CONFIG, SCALED_NAMESPACE, 7, 0)
        second = generate_scaled_episode(CONFIG, SCALED_NAMESPACE, 7, 0)
        other = generate_scaled_episode(CONFIG, SCALED_NAMESPACE, 7, 1)
        self.assertEqual(first, second)
        self.assertEqual(scaled_trace_fingerprint(first), scaled_trace_fingerprint(second))
        self.assertNotEqual(first, other)
        self.assertEqual(first.expert_count, 8)
        self.assertEqual(first.arrival_rate, 3.6)
        self.assertTrue(all(t.arrival_time < 320 for t in first.tokens))
        self.assertEqual({t.expert_id for t in first.tokens}, set(range(8)))
        for expert in range(8):
            ids = [t.expert_token_id for t in first.tokens if t.expert_id == expert]
            self.assertEqual(ids, list(range(len(ids))))
        self.assertTrue(all(len(w.attempt3) == 3 for w in first.work))

    def test_strict_scaled_configuration_and_trace_validation(self):
        for field, value in (("expert_count", 7), ("replicas_per_expert", 2),
                             ("failure_domain_count", 2), ("healthy_offered_load", .5)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    generate_scaled_episode(replace(CONFIG, **{field: value}), SCALED_NAMESPACE, 0, 0)
        valid = manual_episode([(1, K.REGULAR, 0)], [draw()])
        with self.assertRaises(ValueError):
            replace(valid, work=(draw(primary=(1, float("nan"), 1)),))


class AdmissionTests(unittest.TestCase):
    def test_each_expert_has_an_independent_ledger_and_dual_charge_two(self):
        e = manual_episode([
            (10, K.REGULAR, 0), (10.1, K.REGULAR, 1),
            (11, K.REGULAR, 0), (11.1, K.REGULAR, 1),
        ], [draw()] * 4)
        plan = plan_scaled_actions(e, TimeClassRule(I, I, I, I), backup_count=2,
                                   parameters=FULL_CAP)
        self.assertEqual([d.applied for d in plan.decisions], [I, I, N, N])
        self.assertEqual([d.charge for d in plan.decisions], [2, 2, 0, 0])
        audit = plan.audit()
        self.assertEqual(audit["global"]["reserved_work"], 4)
        self.assertEqual(audit["global"]["applied_hedges"], 2)
        self.assertEqual(audit["global"]["applied_backup_copies"], 4)
        self.assertEqual(audit["by_expert"]["0"]["reserved_work"], 2)
        self.assertEqual(audit["by_expert"]["1"]["reserved_work"], 2)

    def test_one_and_two_backups_share_cap_but_have_different_granularity(self):
        e = manual_episode([(10+i/10, K.REGULAR, 0) for i in range(4)], [draw()] * 4)
        one = plan_scaled_actions(e, TimeClassRule(I, I, I, I), backup_count=1,
                                  parameters=FULL_CAP)
        two = plan_scaled_actions(e, TimeClassRule(I, I, I, I), backup_count=2,
                                  parameters=FULL_CAP)
        self.assertEqual(one.decisions[0].cap, two.decisions[0].cap)
        self.assertEqual(one.audit()["global"]["applied_hedges"], 2)
        self.assertEqual(two.audit()["global"]["applied_hedges"], 1)
        self.assertEqual(one.audit()["global"]["reserved_work"], 2)
        self.assertEqual(two.audit()["global"]["reserved_work"], 2)
        self.assertEqual(plan_scaled_actions(e, TimeClassRule(), 0).audit()["global"]["reserved_work"], 0)

    def test_planning_uses_metadata_not_work_or_future_tokens(self):
        e = manual_episode([(10, K.REGULAR, 0), (11, K.URGENT, 1),
                            (12, K.REGULAR, 0)], [draw()] * 3)
        changed = replace(e, work=(draw(primary=(100,100,100)),) * 3)
        rule = TimeClassRule(I, I, I, I)
        self.assertEqual(plan_scaled_actions(e, rule, 2), plan_scaled_actions(changed, rule, 2))
        shortened = replace(e, tokens=e.tokens[:2], work=e.work[:2])
        self.assertEqual(plan_scaled_actions(e, rule, 2).decisions[:2],
                         plan_scaled_actions(shortened, rule, 2).decisions)


class LifecycleTests(unittest.TestCase):
    def test_dual_immediate_first_winner_and_two_running_losers(self):
        e = manual_episode([(10, K.REGULAR, 0)], [draw(primary=(4,1,1))])
        result = simulate_scaled_episode(e, TimeClassRule(I,I,I,I), 2, FULL_CAP)
        token = result.simulation.tokens[0]
        self.assertEqual(token.completion_time, 10.5)
        self.assertEqual(token.winner_attempt_id, 2)
        by_attempt = {a.attempt_id: a for a in token.attempts}
        self.assertEqual(by_attempt[2].status, S.COMPLETED_WINNER)
        self.assertEqual(by_attempt[3].status, S.COMPLETED_LOSER)
        self.assertEqual(by_attempt[0].status, S.COMPLETED_LOSER)
        self.assertAlmostEqual(sum(a.executed_work for a in token.attempts), 5.3)
        self.assertEqual(result.simulation.hedge_launches, 2)

    def test_failure_replays_without_backup_and_live_backup_prevents_replay(self):
        e = manual_episode([(19, K.REGULAR, 0)], [draw(primary=(4,1,1))])
        normal = simulate_scaled_episode(e, TimeClassRule(), 0)
        dual = simulate_scaled_episode(e, TimeClassRule(I,I,I,I), 2, FULL_CAP)
        self.assertEqual(normal.simulation.tokens[0].replay_count, 1)
        self.assertEqual(normal.simulation.tokens[0].completion_time, 23)
        self.assertEqual(normal.simulation.tokens[0].attempts[0].executed_work, .5)
        self.assertEqual(dual.simulation.tokens[0].replay_count, 0)
        self.assertEqual(dual.simulation.tokens[0].completion_time, 19.5)

    def test_f_arrivals_and_replays_balance_across_healthy_backups(self):
        e = manual_episode([(19, K.REGULAR, 0), (19, K.REGULAR, 0),
                            (20, K.REGULAR, 0), (20, K.REGULAR, 0)], [draw()] * 4)
        result = simulate_scaled_episode(e, TimeClassRule(), 0).simulation
        by_token = {t.token_id: t for t in result.tokens}
        self.assertEqual(next(a.replica_id for a in by_token[0].attempts if a.attempt_id == 1), 1)
        self.assertEqual(next(a.replica_id for a in by_token[1].attempts if a.attempt_id == 1), 2)
        self.assertEqual(by_token[2].primary_replica, 1)
        self.assertEqual(by_token[3].primary_replica, 2)

    def test_failure_precedes_exact_completion_and_timer(self):
        e = manual_episode([(19, K.REGULAR, 0)],
                           [draw(primary=(.5, 1, 1), replay=(1, 2, 2))])
        result = simulate_scaled_episode(e, TimeClassRule(D,D,D,D), 1,
                                         FULL_CAP, hedge_delay=1).simulation
        token = result.tokens[0]
        self.assertEqual(token.replay_count, 1)
        self.assertFalse(any(a.attempt_id >= 2 for a in token.attempts))
        self.assertEqual(token.completion_time, 22)
        self.assertEqual(result.hedge_timers_voided, 1)
        self.assertEqual(token.attempts[0].status, S.FAILED_RUNNING)

    def test_common_failure_hits_every_expert_primary_queue(self):
        e = manual_episode([(19, K.REGULAR, 0), (19, K.REGULAR, 1)],
                           [draw(primary=(4,1,1))] * 2)
        result = simulate_scaled_episode(e, TimeClassRule(), 0).simulation
        self.assertEqual([t.replay_count for t in result.tokens], [1, 1])
        failed = [a for a in result.attempts if a.status is S.FAILED_RUNNING]
        self.assertEqual({a.expert_id for a in failed}, {0, 1})
        self.assertTrue(all(a.terminal_time == 20 for a in failed))


class SearchTests(unittest.TestCase):
    def complete_episode(self):
        return manual_episode([
            (0,K.REGULAR,0),(1,K.URGENT,1),(10,K.REGULAR,2),(11,K.URGENT,3),
            (20,K.REGULAR,4),(21,K.URGENT,5),(31,K.REGULAR,6),(32,K.URGENT,7),
        ], [draw()] * 8)

    def test_zero_budget_dual_bank_ties_at_normal_and_keeps_frozen_single(self):
        result = evaluate_scaled_development(
            [self.complete_episode()], ReservationParameters(scale=0), require_frozen=False)
        self.assertEqual(result["dual_backup_search"]["rule_count"], 81)
        self.assertEqual(result["dual_backup_search"]["selected_rule"], "NNNN")
        self.assertEqual(result["arms"]["single_backup_NIIN"]["rule"], "NIIN")
        self.assertEqual(result["arms"]["no_hedge"]["objective"],
                         result["arms"]["single_backup_NIIN"]["objective"])
        self.assertEqual(result, evaluate_scaled_development(
            [self.complete_episode()], ReservationParameters(scale=0), require_frozen=False))

    def test_frozen_count_namespace_and_artifact_guards(self):
        self.assertEqual(SCALED_EXPERT_COUNT, 8)
        self.assertEqual(SCALED_EPISODE_COUNT, 128)
        with self.assertRaisesRegex(ValueError, "128"):
            evaluate_scaled_development([self.complete_episode()], require_frozen=True)
        result = evaluate_scaled_development(
            [self.complete_episode()], ReservationParameters(scale=0), require_frozen=False)
        with tempfile.TemporaryDirectory() as root:
            path = write_scaled_artifact(root, "scaled-test", {"test": True}, result)
            self.assertEqual(sorted(p.name for p in path.iterdir()),
                             ["manifest.json", "rule_results.json", "summary.json"])
            with self.assertRaises(FileExistsError):
                write_scaled_artifact(root, "scaled-test", {}, result)
            broken = copy.deepcopy(result)
            broken["dual_backup_search"]["rules"][-1] = broken["dual_backup_search"]["rules"][0]
            with self.assertRaises(ValueError):
                write_scaled_artifact(root, "broken", {}, broken)
            self.assertFalse((Path(root)/"broken").exists())


if __name__ == "__main__":
    unittest.main()
