"""Comprobaciones de integridad de los problemas de prueba (``tests/dataset.toml``, del
catálogo PDDLDomains)."""

from collections import Counter

from tests.catalog import definition, instances

#: Los niveles de versión de PDDL del catálogo: de STRIPS (1) a PDDL+ (6).
LEVELS = {1, 2, 3, 4, 5, 6}


def test_every_level_has_two_collections() -> None:
    collections = {(i.level, i.group) for i in instances()}
    per_level = Counter(level for level, _group in collections)
    assert set(per_level) == LEVELS
    assert all(count >= 2 for count in per_level.values()), per_level


def test_every_collection_gives_its_problems() -> None:
    """Una colección que falte o cambie de nombre en el catálogo no desaparece en silencio
    de la selección."""
    [selection] = definition()["select"]
    found = Counter(i.group.split("/", 1)[1] for i in instances())
    assert found == {c: len(selection["problems"]) for c in selection["collections"]}


def test_problems_are_of_their_collection_level() -> None:
    for instance in instances():
        assert instance.group.startswith(f"level{instance.level}/"), instance.id
