"""Tests de la representación y la salida de planes."""

import json

from pyplanner.plan import Plan, simulate
from pyplanner.schema import validate_document
from tests.tasks import chain


def test_plan_formats() -> None:
    task = chain()
    plan = Plan.from_indices(task, (0, 1, 2))
    assert len(plan) == 3
    assert plan.to_ipc() == "(step p0 p1)\n(step p1 p2)\n(step p2 p3)\n; cost = 3 (unit cost)\n"
    document = json.loads(plan.to_json())
    assert document["kind"] == "plan"
    assert document["domain"] == "chain" and document["problem"] == "chain-3"
    assert document["cost"] == 3
    assert document["steps"][0] == {"name": "step", "args": ["p0", "p1"]}
    validate_document(document)


def test_simulate_detects_errors() -> None:
    task = chain()
    assert simulate(task, (0, 1, 2)) is None
    assert simulate(task, (1,)) == "paso 1: (step p1 p2) no es aplicable"
    assert simulate(task, (0, 1)) == "el plan no alcanza el objetivo"
