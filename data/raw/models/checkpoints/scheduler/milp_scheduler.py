"""
MILP Maintenance Scheduler using PuLP.

Given a list of engine units with predicted RUL values, schedules
planned maintenance to minimise total cost (repair + failure penalty).
"""

from dataclasses import dataclass

import pulp
import pandas as pd
from loguru import logger


@dataclass
class EngineUnit:
    unit_id: str
    predicted_rul: float          # cycles remaining
    repair_cost: float = 500.0
    failure_cost: float = 5000.0


def schedule_maintenance(
    units: list[EngineUnit],
    planning_horizon: int = 30,     # days
    max_per_day: int = 3,           # max units maintained per day
) -> pd.DataFrame:
    """
    Solve MILP to find optimal maintenance day for each unit.

    Decision variable: x[i, d] ∈ {0, 1}
        1  →  unit i is maintained on day d

    Objective: minimise sum of repair costs + expected failure penalties
    Constraints:
        - Each unit maintained exactly once within horizon
        - Max units per day ≤ max_per_day
        - Unit must be maintained before RUL runs out
    """
    prob = pulp.LpProblem("MaintenanceScheduler", pulp.LpMinimize)

    days = list(range(1, planning_horizon + 1))
    n = len(units)

    # ── Decision variables ──────────────────────────────────────────────────
    x = pulp.LpVariable.dicts(
        "maintain",
        [(i, d) for i in range(n) for d in days],
        cat="Binary",
    )

    # ── Objective ───────────────────────────────────────────────────────────
    # Cost = repair_cost (always) + failure_cost if maintained after RUL deadline
    prob += pulp.lpSum(
        x[i, d] * (
            units[i].repair_cost
            + units[i].failure_cost * max(0, d - units[i].predicted_rul) / planning_horizon
        )
        for i in range(n)
        for d in days
    )

    # ── Constraints ─────────────────────────────────────────────────────────
    for i in range(n):
        # Each unit maintained exactly once
        prob += pulp.lpSum(x[i, d] for d in days) == 1

        # Must be maintained before failure (with 20% safety margin)
        deadline = max(1, int(units[i].predicted_rul * 0.8))
        prob += pulp.lpSum(x[i, d] for d in days if d <= deadline) >= 1

    for d in days:
        # Capacity constraint
        prob += pulp.lpSum(x[i, d] for i in range(n)) <= max_per_day

    # ── Solve ────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    logger.info(f"Solver status: {pulp.LpStatus[status]}")

    # ── Extract schedule ─────────────────────────────────────────────────────
    records = []
    for i, unit in enumerate(units):
        for d in days:
            if pulp.value(x[i, d]) and pulp.value(x[i, d]) > 0.5:
                records.append(
                    {
                        "unit_id": unit.unit_id,
                        "predicted_rul": unit.predicted_rul,
                        "scheduled_day": d,
                        "repair_cost": unit.repair_cost,
                        "on_time": d <= unit.predicted_rul,
                    }
                )
                break

    schedule = pd.DataFrame(records).sort_values("scheduled_day")
    total_cost = pulp.value(prob.objective)
    logger.info(f"Total scheduled cost: ${total_cost:,.2f}")
    return schedule


if __name__ == "__main__":
    # Quick demo
    sample_units = [
        EngineUnit("ENG-001", predicted_rul=5),
        EngineUnit("ENG-002", predicted_rul=15),
        EngineUnit("ENG-003", predicted_rul=25),
        EngineUnit("ENG-004", predicted_rul=8),
        EngineUnit("ENG-005", predicted_rul=20),
    ]
    df = schedule_maintenance(sample_units, planning_horizon=30, max_per_day=2)
    print(df.to_string(index=False))
