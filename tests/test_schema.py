"""Tests del esquema JSON (validación estructural)."""

from typing import Any

import pytest

from pyplanner.errors import SchemaValidationError
from pyplanner.schema import SCHEMA_VERSION, load_schema, validate_document


def minimal_domain(**extra: Any) -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "kind": "domain", "name": "d", **extra}


def minimal_problem(**extra: Any) -> dict[str, Any]:
    goal = {"op": "atom", "predicate": "p", "args": []}
    return {
        "version": SCHEMA_VERSION,
        "kind": "problem",
        "name": "p",
        "domain": "d",
        "goal": goal,
        **extra,
    }


def test_schema_is_valid_json_schema() -> None:
    assert load_schema()["$schema"].endswith("2020-12/schema")


def test_minimal_documents_are_valid() -> None:
    validate_document(minimal_domain())
    validate_document(minimal_problem())


@pytest.mark.parametrize(
    ("document", "path"),
    [
        ({"kind": "trace"}, "kind"),
        (minimal_domain(version=99), "version"),
        (minimal_domain(extra=True), "<raíz>"),
        (minimal_domain(name="1bad"), "name"),
        (
            minimal_domain(actions=[{"name": "a", "effect": {"op": "or", "args": []}}]),
            "actions[0].effect.op",
        ),
        (
            minimal_domain(
                actions=[{"name": "a", "effect": {"op": "not", "arg": {"op": "and", "args": []}}}]
            ),
            "actions[0].effect.arg",
        ),
        (
            minimal_domain(
                actions=[
                    {"name": "a", "precondition": {"op": "atom", "predicate": "p", "args": [3]}}
                ]
            ),
            "actions[0].precondition.args[0]",
        ),
        (minimal_domain(requirements=["strips"]), "requirements[0]"),
        (minimal_problem(init=[{"op": "and", "args": []}]), "init[0]"),
    ],
)
def test_invalid_documents(document: dict[str, Any], path: str) -> None:
    with pytest.raises(SchemaValidationError) as exc:
        validate_document(document)
    assert exc.value.path == path


def test_problem_requires_a_goal() -> None:
    document = minimal_problem()
    del document["goal"]
    with pytest.raises(SchemaValidationError, match="goal"):
        validate_document(document)


def test_not_is_only_allowed_in_effects() -> None:
    negated = {"op": "not", "arg": {"op": "atom", "predicate": "p", "args": []}}
    validate_document(minimal_domain(actions=[{"name": "a", "effect": negated}]))
    with pytest.raises(SchemaValidationError):
        validate_document(minimal_domain(actions=[{"name": "a", "precondition": negated}]))
