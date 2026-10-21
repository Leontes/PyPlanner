"""Lotes adaptativos y evaluación híbrida (Numba en CPU / PyTorch en GPU)."""

import math

import pytest

from pyplanner.heuristics import Evaluation, create_heuristic, numba_available
from pyplanner.heuristics.hybrid import HybridHeuristic, _sample_states
from pyplanner.search import PRESETS, ConfigError, SearchConfig, SearchStatus, search
from pyplanner.search.engine import EagerSearch
from tests.catalog import collection_dir
from tests.tasks import chain
from tests.test_heuristics import TORCH_DEVICES

requires_numba = pytest.mark.skipif(not numba_available(), reason="Numba no está instalado")
requires_cuda = pytest.mark.skipif("cuda" not in TORCH_DEVICES, reason="no hay GPU con CUDA")


def slitherlink() -> object:
    from pyplanner.frontend import read_domain, read_problem
    from pyplanner.grounding import ground

    directory = collection_dir("ipc-2023-classical/slitherlink")
    domain = read_domain(directory / "domain.pddl")
    return ground(domain, read_problem(directory / "p01.pddl", domain))


# --- Lote adaptativo ------------------------------------------------------------------


def engine(**changes: object) -> EagerSearch:
    config = PRESETS["gbfs-ff"].with_changes(adaptive_batch=True, **changes)
    return EagerSearch(chain(), config, create_heuristic("ff", chain(), "python"), None)


def values(*hs: float) -> list[Evaluation]:
    return [Evaluation(h) for h in hs]


def test_batch_doubles_on_plateaus_and_resets_on_progress() -> None:
    search_engine = engine(plateau_patience=2, max_batch_size=8)
    search_engine.adapt(values(5))  # primera mejora
    assert search_engine.batch_size == 1
    for expected in (1, 2, 2, 4, 4, 8, 8, 8):
        search_engine.adapt(values(5, 6))  # sin mejora
        assert search_engine.batch_size == expected
    search_engine.adapt(values(7, 4))  # mejora: vuelve al mínimo
    assert search_engine.batch_size == 1


def test_minimum_batch_size_is_the_configured_one() -> None:
    search_engine = engine(batch_size=4, max_batch_size=16, plateau_patience=1)
    search_engine.adapt(values(3))
    search_engine.adapt(values(3))
    assert search_engine.batch_size == 8
    search_engine.adapt(values(2))
    assert search_engine.batch_size == 4


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"batch_size": 8, "max_batch_size": 4}, "max_batch_size"),
        ({"plateau_patience": 0}, "plateau_patience"),
    ],
)
def test_invalid_adaptive_configurations(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        SearchConfig("x", **changes)  # type: ignore[arg-type]


def test_adaptive_search_grows_batches_and_finds_plans() -> None:
    task = slitherlink()
    result = search(task, PRESETS["gbfs-ff-pref-ab"].with_changes(device="python"))  # type: ignore[arg-type]
    assert result.status is SearchStatus.SOLVED
    assert result.statistics.largest_batch > 1


# --- Evaluación híbrida ---------------------------------------------------------------


@requires_numba
def test_hybrid_without_gpu_uses_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    heuristic = create_heuristic("ff", chain(), "hybrid")
    assert isinstance(heuristic, HybridHeuristic)
    assert heuristic.wide is None and heuristic.threshold == math.inf


@requires_numba
@requires_cuda
def test_hybrid_gives_the_same_values_on_both_devices() -> None:
    task = slitherlink()
    heuristic = create_heuristic("ff", task, "hybrid")  # type: ignore[arg-type]
    assert isinstance(heuristic, HybridHeuristic)
    states = _sample_states(task, 64)  # type: ignore[arg-type]
    reference = create_heuristic("ff", task, "numba").evaluate_batch(states)  # type: ignore[arg-type]
    heuristic.threshold = 1  # todo a la GPU
    assert heuristic.evaluate_batch(states) == reference
    assert heuristic.gpu_batches == 1
    heuristic.threshold = math.inf  # todo a Numba
    assert heuristic.evaluate_batch(states) == reference


@requires_numba
@requires_cuda
def test_hybrid_search_is_the_same_search() -> None:
    """Numba y la GPU dan los mismos valores: la búsqueda es idéntica, solo cambia dónde
    se evalúa cada lote."""
    task = slitherlink()
    cpu = search(task, PRESETS["gbfs-ff-pref-ab"])  # type: ignore[arg-type]
    hybrid = search(task, PRESETS["gbfs-ff-pref-hybrid"])  # type: ignore[arg-type]
    assert hybrid.plan == cpu.plan
    assert hybrid.statistics.expanded == cpu.statistics.expanded
