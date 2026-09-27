from pyplanner.pddl.lexer import TokenKind, tokenize


def kinds_and_texts(text: str) -> list[tuple[TokenKind, str]]:
    return [(t.kind, t.text) for t in tokenize(text)]


def test_basic_tokens() -> None:
    assert kinds_and_texts("(on ?x b)") == [
        (TokenKind.LPAREN, "("),
        (TokenKind.SYMBOL, "on"),
        (TokenKind.SYMBOL, "?x"),
        (TokenKind.SYMBOL, "b"),
        (TokenKind.RPAREN, ")"),
    ]


def test_symbols_are_lowercased() -> None:
    assert kinds_and_texts(":INIT On") == [
        (TokenKind.SYMBOL, ":init"),
        (TokenKind.SYMBOL, "on"),
    ]


def test_comments_are_skipped() -> None:
    assert kinds_and_texts("a ; comentario (x)\nb;pegado") == [
        (TokenKind.SYMBOL, "a"),
        (TokenKind.SYMBOL, "b"),
    ]


def test_parentheses_split_symbols() -> None:
    assert [t.text for t in tokenize("(a(b)c)")] == ["(", "a", "(", "b", ")", "c", ")"]


def test_locations_are_one_based() -> None:
    tokens = list(tokenize("(a\n  bc)", "f.pddl"))
    assert [(t.text, t.location.line, t.location.column) for t in tokens] == [
        ("(", 1, 1),
        ("a", 1, 2),
        ("bc", 2, 3),
        (")", 2, 5),
    ]
    assert tokens[0].location.file == "f.pddl"


def test_empty_input() -> None:
    assert list(tokenize("  ; nada\n")) == []
