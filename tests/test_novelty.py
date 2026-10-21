"""Búsqueda por novedad: tabla de novedad, IW(k) y BFWS."""

import pytest

from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import State, ground
from pyplanner.plan import simulate
from pyplanner.portfolio import _most_informative
from pyplanner.search import PRESETS, ConfigError, SearchConfig, SearchStatus, search
from pyplanner.search.novelty import NoveltyTable
from pyplanner.search.result import SearchResult
from tests.catalog import collection_dir
from tests.tasks import chain, dead_end, fork, token

# --- Tabla de novedad -----------------------------------------------------------------


def facts(*indices: int) -> State:
    mask = 0
    for i in indices:
        mask |= 1 << i
    return State(mask)


def test_novelty_of_atoms_and_pairs() -> None:
    table = NoveltyTable(2, num_facts=4)
    assert table.novelty(facts(0, 1)) == 1  # átomos nuevos
    assert table.novelty(facts(2)) == 1
    assert table.novelty(facts(0, 2)) == 2  # átomos vistos, pero no juntos
    assert table.novelty(facts(0, 2)) == 3  # nada nuevo
    assert table.novelty(facts(1)) == 3


def test_width_one_only_distinguishes_new_atoms() -> None:
    table = NoveltyTable(1, num_facts=4)
    assert table.novelty(facts(0)) == 1
    assert table.novelty(facts(1)) == 1
    assert table.novelty(facts(0, 1)) == 2  # con anchura 1, "nada nuevo" es 2


def test_partitions_are_independent() -> None:
    table = NoveltyTable(2, num_facts=4)
    assert table.novelty(facts(0), partition=5.0) == 1
    assert table.novelty(facts(0), partition=5.0) == 3
    assert table.novelty(facts(0), partition=4.0) == 1  # nuevo en otra partición


def test_numeric_values_are_atoms() -> None:
    table = NoveltyTable(2, num_facts=1)
    assert table.novelty(State(0, (1.0,))) == 1
    assert table.novelty(State(0, (1.0,))) == 3
    assert table.novelty(State(0, (2.0,))) == 1  # valor nuevo de la variable
    assert table.novelty(State(1, (2.0,))) == 1  # hecho nuevo


def test_falsity_of_negated_facts_is_an_atom() -> None:
    """Si el objetivo pide que un hecho sea falso, hacerlo falso es algo nuevo."""
    table = NoveltyTable(2, num_facts=2, negated=[1])
    assert table.novelty(facts(0, 1)) == 1
    assert table.novelty(facts(0)) == 1  # "no 1" es nuevo
    plain = NoveltyTable(2, num_facts=2)
    assert plain.novelty(facts(0, 1)) == 1
    assert plain.novelty(facts(0)) == 3  # sin él, un subconjunto no aporta nada


def test_running_facts_are_not_negated_atoms() -> None:
    from pyplanner.search.novelty import negated_facts
    from tests.test_temporal_engine import light

    task = light()
    assert not {d.running for d in task.durative} & negated_facts(task)


def test_iw_reaches_goals_that_need_deletions() -> None:
    """lights pide una luz encendida y otra apagada: anchura 2. IW(2) lo resuelve gracias
    a los átomos "no on"; sin ellos, el estado final no aportaba nada nuevo y se podaba."""
    from tests.test_adl_engine import task_of

    assert search(task_of("lights"), PRESETS["iw1"]).status is SearchStatus.FAILED
    assert search(task_of("lights"), PRESETS["iw2"]).status is SearchStatus.SOLVED


def test_invalid_width() -> None:
    with pytest.raises(ValueError, match="1 o 2"):
        NoveltyTable(3, num_facts=1)


# --- Configuración --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"algorithm": "iw", "heuristic": "blind"}, "IW necesita"),
        ({"novelty": 3}, "0, 1 o 2"),
        ({"novelty": 2, "algorithm": "astar"}, "solo se usa con GBFS"),
        ({"novelty": 2, "lazy": True}, "evaluación inmediata"),
    ],
)
def test_invalid_novelty_configurations(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        SearchConfig("x", **changes)  # type: ignore[arg-type]


# --- IW(k) ----------------------------------------------------------------------------


def blocksworld() -> tuple[object, ...]:
    directory = collection_dir("ipc-2000/blocks-strips-untyped")
    domain = read_domain(directory / "domain.pddl")
    return domain, read_problem(directory / "p01.pddl", domain)


def test_iw1_solves_width_one_problems() -> None:
    task = chain()
    result = search(task, PRESETS["iw1"])
    assert result.status is SearchStatus.SOLVED
    assert result.plan is not None and simulate(task, result.plan) is None


def test_iw_that_prunes_does_not_claim_unsolvability() -> None:
    """IW(1) no basta para blocksworld: agota la cola habiendo podado estados, y eso no
    demuestra que no haya plan."""
    task = ground(*blocksworld())  # type: ignore[arg-type]
    result = search(task, PRESETS["iw1"])
    assert result.status is SearchStatus.FAILED
    assert result.statistics.pruned > 0


def test_iw_without_pruning_proves_unsolvability() -> None:
    """En token todos los estados son nuevos: IW(1) no poda nada y sí demuestra que no hay
    plan."""
    result = search(token(), PRESETS["iw1"])
    assert result.status is SearchStatus.UNSOLVABLE
    assert result.statistics.pruned == 0


# --- BFWS -----------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["bfws-ff", "bfws-ff-pref", "bfws-goalcount"])
def test_bfws_solves(config: str) -> None:
    for task in (chain(), fork(), dead_end(), ground(*blocksworld())):  # type: ignore[arg-type]
        result = search(task, PRESETS[config])
        assert result.status is SearchStatus.SOLVED, (config, task.problem_name)
        assert result.plan is not None and simulate(task, result.plan) is None


def test_bfws_is_complete() -> None:
    """BFWS no poda: en una tarea sin plan agota el espacio y lo demuestra."""
    assert search(token(), PRESETS["bfws-ff"]).status is SearchStatus.UNSOLVABLE


def test_failed_ranks_below_limit_reached_in_portfolios() -> None:
    results = {
        "iw": SearchResult(SearchStatus.FAILED, config_name="iw"),
        "gbfs": SearchResult(SearchStatus.LIMIT_REACHED, config_name="gbfs"),
    }
    assert _most_informative(results).config_name == "gbfs"
    results["other"] = SearchResult(SearchStatus.UNSOLVABLE, config_name="other")
    assert _most_informative(results).config_name == "other"
