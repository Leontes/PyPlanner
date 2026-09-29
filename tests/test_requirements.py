import pytest

from pyplanner.errors import PDDLSyntaxError, SemanticError
from pyplanner.loader import load_domain
from pyplanner.pddl import parse_domain
from pyplanner.requirements import (
    KNOWN_REQUIREMENTS,
    REQUIREMENT_EXPANSIONS,
    SUPPORTED_REQUIREMENTS,
    expand_requirements,
    first_undeclared,
    requirement_uses,
    unknown_requirement_message,
    unsupported_requirements,
)


def test_no_requirements_means_strips() -> None:
    assert expand_requirements([]) == {":strips"}


def test_adl_expands_transitively() -> None:
    expanded = expand_requirements([":adl"])
    assert {":typing", ":conditional-effects", ":quantified-preconditions"} <= expanded
    # :quantified-preconditions se expande a su vez.
    assert {":existential-preconditions", ":universal-preconditions"} <= expanded


def test_strips_is_supported() -> None:
    assert unsupported_requirements([":strips"]) == []
    assert unsupported_requirements([]) == []


def test_unsupported_requirements_are_reported_expanded() -> None:
    assert unsupported_requirements([":derived-predicates"]) == [":derived-predicates"]
    assert unsupported_requirements([":fluents"]) == []  # numérico: iteración 3a
    assert unsupported_requirements([":durative-actions"]) == []  # temporal: iteración 3b
    assert unsupported_requirements([":adl", ":time"]) == [":time"]


def test_adl_is_supported() -> None:
    assert unsupported_requirements([":adl"]) == []
    assert unsupported_requirements([":quantified-preconditions"]) == []


def test_tables_are_consistent() -> None:
    assert SUPPORTED_REQUIREMENTS <= KNOWN_REQUIREMENTS
    for requirement, expansion in REQUIREMENT_EXPANSIONS.items():
        assert requirement in KNOWN_REQUIREMENTS
        assert expansion <= KNOWN_REQUIREMENTS


def test_requirement_uses_lists_every_construct() -> None:
    p = {"op": "atom", "predicate": "p", "args": []}
    exists = {"op": "exists", "parameters": [{"name": "?y"}], "arg": p}
    condition = {"op": "forall", "parameters": [{"name": "?w"}], "arg": p}
    document = {
        "kind": "domain",
        "types": [{"name": "t"}],
        "actions": [
            {
                "name": "a",
                "parameters": [{"name": "?x", "type": "t"}],
                "precondition": {
                    "op": "and",
                    "args": [
                        {"op": "not", "arg": {"op": "=", "args": ["?x", "?x"]}},
                        {"op": "imply", "args": [p, exists]},
                    ],
                },
                "effect": {
                    "op": "forall",
                    "parameters": [{"name": "?z"}],
                    "arg": {"op": "when", "condition": condition, "arg": {"op": "not", "arg": p}},
                },
            }
        ],
    }
    uses = [(u.requirement, u.path) for u in requirement_uses(document)]
    assert uses == [
        (":typing", "types"),
        (":typing", "actions[0].parameters[0]"),
        (":negative-preconditions", "actions[0].precondition.args[0]"),
        (":equality", "actions[0].precondition.args[0].arg"),
        (":disjunctive-preconditions", "actions[0].precondition.args[1]"),
        (":existential-preconditions", "actions[0].precondition.args[1].args[1]"),
        (":conditional-effects", "actions[0].effect"),
        (":conditional-effects", "actions[0].effect.arg"),
        (":universal-preconditions", "actions[0].effect.arg.condition"),
    ]
    assert first_undeclared(document, [":adl"]) is None
    missing = first_undeclared(document, [":typing", ":negative-preconditions"])
    assert missing is not None and missing.requirement == ":equality"


@pytest.mark.parametrize(
    ("written", "suggestion"),
    [
        (":durative-inequalities", ":duration-inequalities"),  # errata de planning.wiki
        (":durative-action", ":durative-actions"),
        (":numeric-fluent", ":numeric-fluents"),
        (":typeing", ":typing"),
    ],
)
def test_unknown_requirements_suggest_the_closest(written: str, suggestion: str) -> None:
    assert unknown_requirement_message(written).endswith(f"¿quería decir '{suggestion}'?")
    # Las dos fases dan la sugerencia, y sigue siendo un error.
    text = f"(define (domain d) (:requirements {written}))"
    with pytest.raises(PDDLSyntaxError, match=suggestion):
        parse_domain(text)
    document = {"version": 4, "kind": "domain", "name": "d", "requirements": [written]}
    with pytest.raises(SemanticError, match=suggestion):
        load_domain(document)


def test_unknown_requirements_without_a_close_match() -> None:
    assert unknown_requirement_message(":magic") == "requisito desconocido ':magic'"
