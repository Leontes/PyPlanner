from pyplanner.requirements import (
    KNOWN_REQUIREMENTS,
    REQUIREMENT_EXPANSIONS,
    SUPPORTED_REQUIREMENTS,
    expand_requirements,
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
    assert unsupported_requirements([":fluents"]) == [":numeric-fluents"]
    assert ":typing" in unsupported_requirements([":adl"])


def test_tables_are_consistent() -> None:
    assert SUPPORTED_REQUIREMENTS <= KNOWN_REQUIREMENTS
    for requirement, expansion in REQUIREMENT_EXPANSIONS.items():
        assert requirement in KNOWN_REQUIREMENTS
        assert expansion <= KNOWN_REQUIREMENTS
