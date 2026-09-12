"""Deterministic equation checks, not queue simulation or MFG validation."""

import json
import math
from decimal import Decimal
from pathlib import Path
from statistics import NormalDist


normal = NormalDist()
variance = math.log(1.25)
sigma = math.sqrt(variance)
mu = -variance / 2


def survival(work):
    return 0.5 * math.erfc((math.log(work) - mu) / sigma / math.sqrt(2))


def hazard(work):
    z = (math.log(work) - mu) / sigma
    return math.exp(-z * z / 2) / (math.sqrt(2 * math.pi) * sigma * work * survival(work))


def integrate(function, end, steps=4000):
    width = end / steps
    total = function(0) + function(end)
    total += math.fsum((4 if index % 2 else 2) * function(index * width)
                      for index in range(1, steps))
    return total * width / 3


# Two running attempts, fixed speeds until the next deterministic boundary.
arrival_rate = 0.9
boundary = 0.4
running = ((0.5, 0.5), (1.0, 1.0))  # (executed work, current speed)


def no_event(time):
    return math.exp(-arrival_rate * time) * math.prod(
        survival(work + speed * time) / survival(work)
        for work, speed in running
    )


def event_density(time):
    return no_event(time) * (arrival_rate + math.fsum(
        speed * hazard(work + speed * time) for work, speed in running
    ))


atom_mass = no_event(boundary)
continuous_mass = integrate(event_density, boundary)
probability_error = abs(continuous_mass + atom_mass - 1)
hold_from_events = integrate(lambda time: time * event_density(time), boundary) + boundary * atom_mass
hold_from_survival = integrate(no_event, boundary)
holding_error = abs(hold_from_events - hold_from_survival)
assert probability_error < 1e-10, probability_error
assert holding_error < 1e-10, holding_error
assert int(Decimal("2.8125") // Decimal("1")) == 2

# Fixed trajectories for the integral identity only, no engine calls.
# Primary wins at 1; Hedge runs from 0 through 2 at unit speed.
latency = 1.0
hedge_work = 2.0
price = 0.5
direct_cost = latency + price * hedge_work
integrated_cost = (1 + price) * 1 + price * 1
assert direct_cost == integrated_cost == 2.0

report = {
    "scope": "deterministic_algebra_and_quadrature_only",
    "queue_simulation_calls": 0,
    "claims_mfg_validation": False,
    "lognormal": {
        "mu": mu,
        "sigma_squared": variance,
        "healthy_delay_q90": math.exp(mu + sigma * normal.inv_cdf(0.9)),
        "hazards_at_0_5_1_2": [hazard(value) for value in (0.5, 1.0, 2.0)],
    },
    "event_kernel": {
        "boundary_atom_mass": atom_mass,
        "continuous_event_mass": continuous_mass,
        "normalization_error": probability_error,
        "mean_holding_time_from_event_kernel": hold_from_events,
        "mean_holding_time_from_survival": hold_from_survival,
        "holding_identity_error": holding_error,
        "simpson_subintervals": 4000,
    },
    "initial_cost_running_hedge_loser": {
        "direct": direct_cost,
        "integrated": integrated_cost,
    },
    "finite_window_whole_admissions": 2,
}
output = Path(__file__).with_name("theory-r1-algebra-check.json")
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
