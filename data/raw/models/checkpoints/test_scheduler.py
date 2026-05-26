"""Unit tests for MILP scheduler."""

import pytest
from src.scheduler.milp_scheduler import EngineUnit, schedule_maintenance


def make_units(ruls: list[float]) -> list[EngineUnit]:
    return [EngineUnit(f"ENG-{i:03d}", rul) for i, rul in enumerate(ruls, 1)]


def test_all_units_scheduled():
    units = make_units([10, 20, 25])
    df = schedule_maintenance(units, planning_horizon=30, max_per_day=3)
    assert len(df) == 3


def test_capacity_respected():
    units = make_units([5, 5, 5, 5, 5, 5])
    df = schedule_maintenance(units, planning_horizon=30, max_per_day=2)
    counts = df.groupby("scheduled_day").size()
    assert (counts <= 2).all()


def test_critical_unit_scheduled_early():
    units = make_units([3, 28])
    df = schedule_maintenance(units, planning_horizon=30, max_per_day=2)
    critical = df[df["unit_id"] == "ENG-001"]
    assert critical["scheduled_day"].values[0] <= 3
