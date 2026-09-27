import pytest

from pyplanner.errors import PDDLSyntaxError
from pyplanner.pddl.sexpr import SList, Symbol, parse_sexprs


def test_nested_lists() -> None:
    [expr] = parse_sexprs("(a (b c) ())")
    assert isinstance(expr, SList)
    a, bc, empty = expr.items
    assert isinstance(a, Symbol) and a.value == "a"
    assert isinstance(bc, SList) and [s.value for s in bc.items if isinstance(s, Symbol)] == [
        "b",
        "c",
    ]
    assert isinstance(empty, SList) and empty.items == ()


def test_multiple_top_level_expressions() -> None:
    assert len(parse_sexprs("(a) (b) c")) == 3


def test_unclosed_parenthesis_reports_its_position() -> None:
    with pytest.raises(PDDLSyntaxError, match="sin cerrar") as exc:
        parse_sexprs("(a\n (b)", "x.pddl")
    assert exc.value.location is not None
    assert (exc.value.location.line, exc.value.location.column) == (1, 1)


def test_unexpected_closing_parenthesis() -> None:
    with pytest.raises(PDDLSyntaxError, match="sin abrir") as exc:
        parse_sexprs("(a))")
    assert exc.value.location is not None
    assert exc.value.location.column == 4
