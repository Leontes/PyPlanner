"""Tests de las heurísticas y de la equivalencia entre la referencia y PyTorch."""

import random

import pytest

from pyplanner.frontend import read_domain, read_problem
from pyplanner.grounding import GroundTask, State, ground
from pyplanner.heuristics import (
    DEAD_END,
    HEURISTICS,
    TENSOR_HEURISTICS,
    HeuristicError,
    create_heuristic,
    numba_available,
)
from tests.tasks import chain, dead_end, fork, goal_state, token, unreachable
from tests.tiers import Instance, discover_instances

try:
    import torch

    TORCH_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
except ImportError:  # pragma: no cover - depende del entorno
    TORCH_DEVICES = []

requires_torch = pytest.mark.skipif(not TORCH_DEVICES, reason="PyTorch no está instalado")

#: Dispositivos acelerados disponibles: todos tienen que dar lo mismo que la referencia.
ACCELERATED_DEVICES = (["numba"] if numba_available() else []) + TORCH_DEVICES
requires_accelerated = pytest.mark.skipif(
    not ACCELERATED_DEVICES, reason="ni Numba ni PyTorch están instalados"
)


def values(task: GroundTask, name: str, device: str = "python") -> float:
    return create_heuristic(name, task, device).evaluate(task.init).value


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (chain(), {"blind": 1, "goalcount": 1, "hmax": 3, "hadd": 3, "ff": 3}),
        (fork(), {"blind": 1, "goalcount": 2, "hmax": 1, "hadd": 2, "ff": 1}),
        (token(), {"blind": 1, "goalcount": 2, "hmax": 1, "hadd": 2, "ff": 2}),
    ],
    ids=["chain", "fork", "token"],
)
def test_reference_values(task: GroundTask, expected: dict[str, float]) -> None:
    assert {name: values(task, name) for name in HEURISTICS} == expected


def test_goal_states_are_zero() -> None:
    task = chain()
    for name in HEURISTICS:
        assert create_heuristic(name, task).evaluate(goal_state(task)).value == 0


def test_unreachable_goal_is_a_dead_end() -> None:
    task = unreachable()
    for name in ("hmax", "hadd", "ff"):
        evaluation = create_heuristic(name, task).evaluate(task.init)
        assert evaluation.value == DEAD_END
        assert evaluation.is_dead_end


def test_ff_preferred_operators() -> None:
    task = chain()
    evaluation = create_heuristic("ff", task).evaluate(task.init)
    assert [str(task.actions[a]) for a in evaluation.preferred] == ["(step p0 p1)"]
    fork_task = fork()
    evaluation = create_heuristic("ff", fork_task).evaluate(fork_task.init)
    assert [fork_task.actions[a].name for a in evaluation.preferred] == ["both"]


def test_batch_and_single_evaluation_agree() -> None:
    task = chain()
    heuristic = create_heuristic("ff", task)
    states = [task.init, task.apply(task.init, task.actions[0]), goal_state(task)]
    assert heuristic.evaluate_batch(states) == [heuristic.evaluate(s) for s in states]


def test_unknown_heuristic_or_device() -> None:
    with pytest.raises(HeuristicError, match="desconocida"):
        create_heuristic("magic", chain())
    with pytest.raises(HeuristicError, match="dispositivo desconocido"):
        create_heuristic("ff", chain(), "tpu")
    with pytest.raises(HeuristicError, match="no tiene implementación en PyTorch"):
        create_heuristic("goalcount", chain(), "cpu")


def random_walk_states(task: GroundTask, count: int, seed: int = 0) -> list[State]:
    """Estados alcanzables obtenidos con paseos aleatorios desde el inicial."""
    rng = random.Random(seed)
    states = [task.init]
    state = task.init
    while len(states) < count:
        actions = list(task.applicable(state))
        if not actions:
            state = task.init
            continue
        state = task.apply(state, rng.choice(actions))
        states.append(state)
    return states


TIER1_AND_2 = [i for i in discover_instances() if i.tier <= 2]


@requires_accelerated
@pytest.mark.parametrize("device", ACCELERATED_DEVICES)
@pytest.mark.parametrize("name", TENSOR_HEURISTICS)
@pytest.mark.parametrize("instance", TIER1_AND_2, ids=lambda i: i.id)
def test_accelerated_matches_reference(instance: Instance, name: str, device: str) -> None:
    domain = read_domain(instance.domain)
    task = ground(domain, read_problem(instance.problem, domain))
    states = random_walk_states(task, 200)
    reference = create_heuristic(name, task, "python").evaluate_batch(states)
    accelerated = create_heuristic(name, task, device).evaluate_batch(states)
    assert accelerated == reference


@requires_accelerated
@pytest.mark.parametrize("device", ACCELERATED_DEVICES)
def test_accelerated_handmade_tasks(device: str) -> None:
    for task in (chain(), fork(), token(), unreachable()):
        for name in TENSOR_HEURISTICS:
            states = [task.init, goal_state(task) if task.goal else task.init]
            reference = create_heuristic(name, task, "python").evaluate_batch(states)
            tensor = create_heuristic(name, task, device).evaluate_batch(states)
            assert tensor == reference, (name, device)


def all_states(task: GroundTask) -> list[State]:
    """Todos los estados alcanzables (para tareas pequeñas)."""
    seen = {task.init}
    frontier = [task.init]
    while frontier:
        state = frontier.pop()
        for action in task.applicable(state):
            successor = task.apply(state, action)
            if successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return sorted(seen)


@pytest.mark.parametrize("device", ["python", *ACCELERATED_DEVICES])
def test_batch_mixing_dead_ends_and_live_states(device: str) -> None:
    task = dead_end()
    states = all_states(task)
    evaluations = create_heuristic("ff", task, device).evaluate_batch(states)
    by_position = {
        str(task.atoms(s)[0].args[0]): e for s, e in zip(states, evaluations, strict=True)
    }
    assert by_position["b"].is_dead_end and by_position["b"].preferred == ()
    assert by_position["s"].value == 2 and by_position["a"].value == 1
    assert by_position["t"].value == 0
    assert [str(task.actions[i]) for i in by_position["s"].preferred] == ["(move s a)"]


def test_auto_device_prefers_numba() -> None:
    task = chain()
    heuristic = create_heuristic("ff", task)
    assert heuristic.device == ("numba" if numba_available() else "python")
    # Las triviales no se compilan: no hay nada que acelerar.
    assert create_heuristic("goalcount", task, "numba").device == "python"


def test_auto_device_without_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyplanner.heuristics as heuristics

    monkeypatch.setattr(heuristics, "numba_available", lambda: False)
    assert heuristics.create_heuristic("ff", chain()).device == "python"
    with pytest.raises(HeuristicError, match="necesita Numba"):
        heuristics.create_heuristic("ff", chain(), "numba")
