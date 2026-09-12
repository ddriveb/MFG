import unittest

from mfg_hedge.token_mfg_response_solver import (
    MFGCostModel,
    MFGSolverConfig,
    PolicyRow,
    PolicyState,
    ActionCostEstimate,
    FiniteKDeviationDiagnostic,
    ForwardSnapshot,
    ForwardStateRow,
    MFGSolverError,
    CallLedger,
    solve_unpriced_priced_mfg,
)


FP = "a" * 64


def policy():
    return PolicyState((PolicyRow("s0", ("a0", "a1"), (0.5, 0.5)),))


def snapshot(
    iteration=0, *, complete=True, share=(0.5, 0.5), occupancy=1.0,
    model_id="unpriced_mfg", policy_fingerprint=None,
):
    return ForwardSnapshot(
        model_id=model_id,
        iteration=iteration,
        policy_fingerprint=policy_fingerprint or policy().fingerprint,
        environment_fingerprint=FP,
        trace_library_fingerprint=FP,
        mu_fingerprint=FP,
        nu_fingerprint=FP,
        eta_fingerprint=FP,
        x_fingerprint=FP,
        states=(ForwardStateRow(
            "s0", "Regular", occupancy,
            (("a0", share[0]), ("a1", share[1])),
            (("a0", share[0]), ("a1", share[1])),
        ),),
        sample_count=16,
        complete=complete,
        scheduler_calls=16,
    )


def q_rows(*, queue=0.0, effective_n=16, first=1.0, second=2.0):
    return (
        ActionCostEstimate(
            "s0", "a0", "Regular", first, 0.0, 0.0, 0.0,
            queue, effective_n, 0.1,
        ),
        ActionCostEstimate(
            "s0", "a1", "Regular", second, 0.0, 0.0, 0.0,
            queue, effective_n, 0.1,
        ),
    )


def deviation():
    return FiniteKDeviationDiagnostic(
        complete=True,
        effective_n=16,
        max_abs_pathwise_difference=0.0,
        trace_library_fingerprint=FP,
        scheduler_calls=144,
    )


class TokenMFGResponseSolverTests(unittest.TestCase):
    def test_cost_models_keep_private_units_and_observable_price_separate(self):
        estimate = q_rows(queue=4.0)[0]
        self.assertEqual(MFGCostModel.unpriced().score(estimate), 1.0)
        self.assertEqual(MFGCostModel.priced().score(estimate), 2.0)
        self.assertEqual(MFGCostModel.unpriced().congestion_charge(estimate), 0.0)
        self.assertEqual(MFGCostModel.priced().congestion_charge(estimate), 1.0)

    def test_policy_state_uses_stable_logit_and_damping(self):
        current = policy()
        response = current.logit_response({"s0": {"a0": 1.0, "a1": 2.0}}, beta=2.0)
        self.assertGreater(response["s0"][0], response["s0"][1])
        updated = current.damped(response, eta=0.2)
        self.assertAlmostEqual(sum(updated.rows[0].probabilities), 1.0)
        self.assertGreater(updated.rows[0].probabilities[0], 0.5)

    def test_bounded_solver_runs_both_models_with_full_iteration_records(self):
        calls = []

        def forward(current, model_id, iteration):
            calls.append(("forward", model_id, iteration, current.fingerprint))
            return snapshot(
                iteration, model_id=model_id, policy_fingerprint=current.fingerprint,
            )

        def continuation(current_snapshot, current, model_id, iteration):
            calls.append(("continuation", model_id, iteration, current_snapshot.environment_fingerprint))
            return q_rows(queue=0.0)

        def finite(current_snapshot, current, model_id, iteration):
            calls.append(("finite", model_id, iteration, current_snapshot.trace_library_fingerprint))
            return deviation()

        result = solve_unpriced_priced_mfg(
            forward, continuation, finite, initial_policy=policy(),
        )
        self.assertEqual(result.model_ids, ("unpriced_mfg", "priced_mfg"))
        self.assertEqual(len(result.model_results), 2)
        for model_result in result.model_results:
            self.assertIn(model_result.status, {"fixed_point", "cycle", "not_converged"})
            self.assertEqual(model_result.claim_boundary, "bounded_mfg_implementation_smoke")
            self.assertFalse(model_result.claims_mfg)
            self.assertFalse(model_result.claims_nash)
            self.assertTrue(model_result.iterations)
            self.assertTrue(all(row.q_rows for row in model_result.iterations))
            self.assertTrue(all(row.policy_rows for row in model_result.iterations))
        self.assertEqual(len(calls) % 6, 0)

    def test_statistics_insufficient_is_terminal_and_fail_closed(self):
        def forward(current, model_id, iteration):
            return snapshot(
                iteration, model_id=model_id, policy_fingerprint=current.fingerprint,
            )

        def continuation(current_snapshot, current, model_id, iteration):
            return q_rows()[:1]

        def finite(current_snapshot, current, model_id, iteration):
            self.fail("finite provider must not run after insufficient Q")

        result = solve_unpriced_priced_mfg(
            forward, continuation, finite, initial_policy=policy(),
        )
        self.assertEqual(result.model_results[0].status, "statistics_insufficient")
        self.assertEqual(result.model_results[0].attempted_calls, 160)

    def test_provider_exception_becomes_physical_failed_without_retry(self):
        calls = []

        def forward(current, model_id, iteration):
            calls.append((model_id, iteration))
            raise RuntimeError("synthetic invariant failure")

        def never(*args):
            self.fail("failed forward must not dispatch later providers")

        result = solve_unpriced_priced_mfg(
            forward, never, never, initial_policy=policy(),
        )
        failed = result.model_results[0]
        self.assertEqual(failed.status, "physical_failed")
        self.assertEqual(failed.failure_type, "RuntimeError")
        self.assertIn("synthetic invariant failure", failed.failure_message)
        self.assertEqual(failed.failure_iteration, 0)
        self.assertEqual(failed.attempted_calls, 16)
        self.assertEqual(len(calls), 2)  # one failed call per independent model

    def test_continuation_exception_is_physical_failed_and_not_retried(self):
        calls = []

        def forward(current, model_id, iteration):
            return snapshot(
                iteration, model_id=model_id, policy_fingerprint=current.fingerprint,
            )

        def continuation(current_snapshot, current, model_id, iteration):
            calls.append((model_id, iteration))
            raise ArithmeticError("continuation invariant failed")

        def never(*args):
            self.fail("finite provider must not run after continuation failure")

        result = solve_unpriced_priced_mfg(
            forward, continuation, never, initial_policy=policy(),
            models=(MFGCostModel.unpriced(),),
        )
        failed = result.model_results[0]
        self.assertEqual(failed.status, "physical_failed")
        self.assertEqual(failed.failure_type, "ArithmeticError")
        self.assertIn("continuation invariant failed", failed.failure_message)
        self.assertEqual(failed.attempted_calls, 16 + 144)
        self.assertEqual(len(calls), 1)

    def test_incomplete_forward_is_statistics_insufficient(self):
        def forward(current, model_id, iteration):
            return snapshot(
                iteration, model_id=model_id, policy_fingerprint=current.fingerprint,
                complete=False,
            )

        def never(*args):
            self.fail("incomplete forward must not dispatch response providers")

        result = solve_unpriced_priced_mfg(
            forward, never, never, initial_policy=policy(),
            models=(MFGCostModel.unpriced(),),
        )
        self.assertEqual(result.model_results[0].status, "statistics_insufficient")
        self.assertEqual(result.model_results[0].attempted_calls, 16)

    def test_call_ledger_rejects_budget_overrun_without_retry(self):
        ledger = CallLedger(10)
        reservation = ledger.reserve(10)
        ledger.settle(reservation, completed=9)
        with self.assertRaises(MFGSolverError):
            ledger.reserve(1)
        self.assertEqual(ledger.completed_calls, 9)
        self.assertEqual(ledger.failed_calls, 0)

    def test_cycle_detection_can_be_exercised_with_explicit_test_config(self):
        step = {"value": 0}

        def forward(current, model_id, iteration):
            step["value"] += 1
            bit = step["value"] % 2
            return snapshot(
                iteration,
                model_id=model_id,
                policy_fingerprint=current.fingerprint,
                share=(1.0 if bit else 0.0, 0.0 if bit else 1.0),
            )

        def continuation(current_snapshot, current, model_id, iteration):
            if iteration % 2:
                return q_rows(first=2.0, second=1.0)
            return q_rows()

        def finite(current_snapshot, current, model_id, iteration):
            return deviation()

        result = solve_unpriced_priced_mfg(
            forward, continuation, finite, initial_policy=policy(),
            config=MFGSolverConfig(max_iterations=4, damping=1.0),
            models=(MFGCostModel.unpriced(),),
        )
        self.assertEqual(result.model_results[0].status, "cycle")


if __name__ == "__main__":
    unittest.main()
