"""Programación lineal del motor temporal, al estilo de COLIN (Coles, Coles, Fox y Long, 2012).

Algunas variables numéricas no tienen un valor fijo en un estado, porque dependen de
*cuándo* ocurren los pasos:

* las que cambian con **efectos continuos** (``(increase f (* #t r))``): crecen o decrecen
  a un ritmo constante mientras dura la acción;
* las que reciben valores que dependen de ``?duration`` en acciones cuya **duración no es
  fija** (``(at end (decrease (energy) (* 2 ?duration)))`` con ``(<= ?duration 10)``).

Son las **variables lineales**. Sus condiciones y efectos no se deciden sobre el estado,
sino con un programa lineal por estado:

* **incógnitas:** el tiempo ``t_j`` de cada paso y el valor ``y_v,k`` de cada variable
  lineal ``v`` después de su cambio *k*-ésimo (``y_v,0`` es el valor inicial);
* **entre dos cambios** de ``v``, su valor evoluciona según el ritmo acumulado ``r``: en el
  paso ``j`` (después del cambio *k*, en el paso ``c``) vale ``y_v,k + r·(t_j − t_c)``.
  Todos los pasos que leen o cambian ``v`` están ordenados entre sí (por las órdenes al
  estilo de POPF), así que la fórmula es exacta;
* **restricciones:** las de la red temporal; las condiciones de cada paso sobre variables
  lineales (antes de sus efectos); los efectos (``y_v,k+1`` en función del valor anterior);
  la duración ``d`` de cada acción cuya duración no es fija o depende de variables
  lineales (una incógnita más, entre sus límites, con ``d = t_fin − t_inicio`` al terminar),
  que es el valor de ``?duration``; y las invariantes (``over all``) de las acciones en
  curso, en cada extremo de tramo: al empezar la acción (después de sus efectos), antes y
  después de cada paso que cambia alguna de sus variables, y al terminar (antes).
  Una condición lineal sobre valores que cambian linealmente se cumple en todo un tramo si
  se cumple en sus extremos;
* **objetivo:** minimizar la suma de los tiempos (la planificación más temprana).

Si el programa no tiene solución, la secuencia es imposible y el sucesor se descarta. El
estado guarda, para cada variable lineal, su valor justo después de su último cambio en la
solución: es lo que ven la heurística y el objetivo (en un estado objetivo no hay acciones
en curso, así que ese valor ya no cambia).

Se resuelve con HiGHS (``scipy.optimize.linprog``), y solo cuando el paso toca variables
lineales o retrasa pasos anteriores.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

from pyplanner.grounding.numeric import (
    Const,
    NumericAssignment,
    NumericCondition,
    NumExpr,
    Var,
    linear,
    make,
)

#: Holgura para las comparaciones estrictas (``>``) en el programa lineal.
STRICT = 1e-6

#: Incógnita del programa: ``("t", paso)``, ``("y", variable, cambio)`` o ``("d", paso de
#: inicio)`` (duración de una acción).
Symbol = tuple[str, int] | tuple[str, int, int]


@dataclass(frozen=True)
class LinExpr:
    """Expresión lineal sobre las incógnitas: Σ coeficiente·incógnita + constante."""

    terms: Mapping[Symbol, float] = field(default_factory=dict)
    constant: float = 0.0

    def __add__(self, other: "LinExpr") -> "LinExpr":
        terms = dict(self.terms)
        for symbol, weight in other.terms.items():
            terms[symbol] = terms.get(symbol, 0.0) + weight
        return LinExpr(terms, self.constant + other.constant)

    def scale(self, factor: float) -> "LinExpr":
        return LinExpr({s: w * factor for s, w in self.terms.items()}, self.constant * factor)


def time_of(step: int) -> LinExpr:
    return LinExpr({("t", step): 1.0})


def duration_of(start_step: int) -> LinExpr:
    """Duración de la acción que empezó en ``start_step``."""
    return LinExpr({("d", start_step): 1.0})


def constant(value: float) -> LinExpr:
    return LinExpr({}, value)


@dataclass(frozen=True)
class Row:
    """Restricción ``Σ terms + constant ≤ 0`` (o ``= 0`` si ``equality``)."""

    terms: tuple[tuple[Symbol, float], ...]
    constant: float
    equality: bool = False


def condition_rows(condition: NumericCondition, value: LinExpr) -> list[Row]:
    """Filas para ``value ⋈ 0`` (``value`` ya es la expresión de la condición)."""
    terms = tuple(value.terms.items())
    if condition.comparator == "=":
        return [Row(terms, value.constant, equality=True)]
    negated = tuple((s, -w) for s, w in terms)
    margin = STRICT if condition.comparator == ">" else 0.0
    return [Row(negated, -value.constant + margin)]


@dataclass(frozen=True)
class LinearModel:
    """Lo que el programa lineal necesita saber de la tarea (fijo).

    Attributes:
        variables: índices de las variables lineales.
        durations: variables que representan ``?duration`` de una acción durativa (índice
            de la variable → índice de la acción durativa).
        conditions: por acción, sus condiciones sobre variables lineales.
        effects: por acción, sus efectos sobre variables lineales.
        conditional: por acción, sus efectos condicionales sobre variables lineales
            (condición clásica, efectos).
        invariants: por acción durativa, su invariante sobre variables lineales.
        invariant_variables: por acción durativa, todas las variables de esa invariante.
        bounds: por acción durativa, sus límites de duración que dependen de variables
            lineales (al empezar, al terminar).
        timed: acciones durativas cuya duración es una incógnita del programa.
        initial: valores iniciales de las variables.
    """

    variables: frozenset[int]
    durations: Mapping[int, int]
    conditions: Sequence[tuple[NumericCondition, ...]]
    effects: Sequence[tuple[NumericAssignment, ...]]
    conditional: Sequence[tuple[tuple[Any, tuple[NumericAssignment, ...]], ...]]
    invariants: Sequence[tuple[NumericCondition, ...]]
    invariant_variables: Sequence[frozenset[int]]
    bounds: Sequence[tuple[tuple[Any, ...], tuple[Any, ...]]]
    timed: frozenset[int]
    initial: tuple[float | None, ...]

    @property
    def symbolic(self) -> frozenset[int]:
        return self.variables | frozenset(self.durations)


@dataclass(frozen=True)
class LinearState:
    """La parte del programa lineal de una línea temporal (inmutable).

    Attributes:
        chains: por variable lineal, sus cambios: (paso, ritmo después del cambio). El
            cambio 0 es el valor inicial en el origen.
        rows: restricciones acumuladas (sin las de la red temporal, que se añaden al
            resolver).
        rates: ritmos de los efectos continuos de cada acción en curso.
        invariants: acciones en curso con invariante sobre variables lineales.
    """

    chains: Mapping[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    rows: tuple[Row, ...] = ()
    rates: Mapping[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    invariants: frozenset[int] = frozenset()

    def value(self, variable: int, step: int) -> LinExpr:
        """Valor de una variable lineal en el paso ``step`` (antes de sus efectos)."""
        chain = self.chains.get(variable, ((0, 0.0),))
        changed, rate = chain[-1]
        current = LinExpr({("y", variable, len(chain) - 1): 1.0})
        if rate == 0.0:
            return current
        return current + (time_of(step) + time_of(changed).scale(-1.0)).scale(rate)


def linearize(
    expression: NumExpr,
    symbolic: frozenset[int],
    values: Sequence[float | None],
    lookup: Callable[[int], LinExpr],
) -> LinExpr | None:
    """``expression`` como expresión lineal: las variables de ``symbolic`` con ``lookup`` y
    las demás con su valor en ``values``. ``None`` si falta un valor o no es lineal."""
    substituted = _substitute(expression, symbolic, values)
    if substituted is None:
        return None
    form = linear(substituted)
    if form is None:
        return None
    result = constant(form.constant)
    for key, weight in form.terms:
        result = result + lookup(key).scale(weight)
    return result


def _substitute(
    expression: NumExpr, symbolic: frozenset[int], values: Sequence[float | None]
) -> NumExpr | None:
    if isinstance(expression, Const):
        return expression
    if isinstance(expression, Var):
        if expression.key in symbolic:
            return expression
        value = values[expression.key]
        return None if value is None else Const(value)
    args = [_substitute(arg, symbolic, values) for arg in expression.args]
    if any(arg is None for arg in args):
        return None
    return make(expression.op, args)  # type: ignore[arg-type]


def extend(
    model: LinearModel,
    state: LinearState,
    action: int,
    step: int,
    kind_start: bool,
    kind_end: bool,
    durative: int,
    start_step: int,
    before: Sequence[float | None],
    after: Sequence[float | None],
    writes: frozenset[int],
    started: Sequence[tuple[int, float, NumExpr]] = (),
    extra_effects: Sequence[NumericAssignment] = (),
    stn_bounds: tuple[float, float] | None = None,
) -> tuple[LinearState, bool] | None:
    """Añade el paso ``step`` (la acción ``action``) al programa lineal.

    Args:
        kind_start, kind_end: si es el inicio o el fin de la acción durativa ``durative``
            (que empezó en ``start_step``).
        before, after: valores del estado antes y después del paso (para las variables
            que no son lineales).
        writes: variables que cambia el paso.
        started: efectos continuos que empiezan (solo en un inicio; los condicionales ya
            decididos): (variable, signo, ritmo).
        extra_effects: efectos condicionales sobre variables lineales que se disparan.
        stn_bounds: duración mínima y máxima de la red temporal (en un inicio).

    Returns:
        El nuevo estado del programa y si hace falta resolverlo, o ``None`` si algo no está
        definido (una variable sin valor, un ritmo indefinido...).
    """
    symbolic = model.symbolic
    chains = dict(state.chains)
    rows = list(state.rows)

    def before_value(variable: int) -> LinExpr:
        if variable in model.durations:
            # ?duration: solo lo usan los efectos de su propia acción.
            return duration_of(start_step)
        return state.value(variable, step)

    def rows_for(
        conditions: Sequence[NumericCondition],
        lookup: Callable[[int], LinExpr],
        values: Sequence[float | None],
    ) -> bool:
        for condition in conditions:
            value = linearize(condition.expression, symbolic, values, lookup)
            if value is None:
                return False
            rows.extend(condition_rows(condition, value))
        return True

    # Condiciones del paso e invariantes afectadas, antes de los efectos.
    checks = list(model.conditions[action])
    if kind_end and durative >= 0:
        checks.extend(model.invariants[durative])
    touched = [
        d for d in state.invariants if d != durative and model.invariant_variables[d] & writes
    ]
    for other in touched:
        checks.extend(model.invariants[other])
    if not rows_for(checks, before_value, before):
        return None

    # Duración como incógnita: sus límites al empezar y al terminar, y su valor al terminar.
    if durative in model.timed and (kind_start or kind_end):
        duration = duration_of(start_step)
        if kind_start and stn_bounds is not None:
            lower, upper = stn_bounds
            rows.append(Row(tuple(duration.scale(-1.0).terms.items()), lower))
            if upper < math.inf:
                rows.append(Row(tuple(duration.terms.items()), -upper))
        if kind_end:
            difference = duration + time_of(step).scale(-1.0) + time_of(start_step)
            rows.append(Row(tuple(difference.terms.items()), 0.0, equality=True))
        for bound in model.bounds[durative][0 if kind_start else 1]:
            value = linearize(bound.value, symbolic, before, before_value)
            if value is None:
                return None
            gap = duration + value.scale(-1.0)  # d − valor
            if bound.comparator == "=":
                rows.append(Row(tuple(gap.terms.items()), gap.constant, equality=True))
            elif bound.comparator == ">=":
                negated = gap.scale(-1.0)
                rows.append(Row(tuple(negated.terms.items()), negated.constant))
            else:
                rows.append(Row(tuple(gap.terms.items()), gap.constant))

    # Efectos discretos y cambios de ritmo.
    rates = dict(state.rates)
    rate_changes: dict[int, float] = {}
    if kind_start:
        active = []
        for variable, sign, rate_expression in started:
            value = linearize(rate_expression, symbolic, before, before_value)
            if value is None or value.terms:
                return None  # el ritmo tiene que ser una constante
            active.append((variable, sign * value.constant))
            rate_changes[variable] = rate_changes.get(variable, 0.0) + sign * value.constant
        rates[durative] = tuple(active)
    elif kind_end:
        for variable, rate in state.rates.get(durative, ()):
            rate_changes[variable] = rate_changes.get(variable, 0.0) - rate
        rates.pop(durative, None)

    by_variable: dict[int, list[NumericAssignment]] = {}
    for assignment in (*model.effects[action], *extra_effects):
        by_variable.setdefault(assignment.variable, []).append(assignment)
    new_values: dict[int, LinExpr] = {}
    for variable in sorted(set(by_variable) | set(rate_changes)):
        current = state.value(variable, step)
        result = current
        group = by_variable.get(variable, [])
        if len(group) > 1 and not all(a.op in ("increase", "decrease") for a in group):
            return None  # efectos incompatibles sobre la misma variable
        for assignment in group:
            amount = linearize(assignment.value, symbolic, before, before_value)
            if amount is None:
                return None
            if assignment.op == "increase":
                result = result + amount
            elif assignment.op == "decrease":
                result = result + amount.scale(-1.0)
            elif assignment.op == "assign":
                result = amount
            else:  # scale-up / scale-down: el factor tiene que ser constante
                if amount.terms or (assignment.op == "scale-down" and amount.constant == 0):
                    return None
                factor = amount.constant if assignment.op == "scale-up" else 1 / amount.constant
                result = current.scale(factor)
        chain = chains.get(variable, ((0, 0.0),))
        rate = chain[-1][1] + rate_changes.get(variable, 0.0)
        new = LinExpr({("y", variable, len(chain)): 1.0})
        rows.append(Row(tuple((new + result.scale(-1.0)).terms.items()), -result.constant, True))
        chains[variable] = (*chain, (step, rate))
        new_values[variable] = new

    updated = LinearState(chains, tuple(rows), rates, state.invariants)

    def after_value(variable: int) -> LinExpr:
        if variable in new_values:
            return new_values[variable]
        return (
            updated.value(variable, step)
            if variable not in model.durations
            else before_value(variable)
        )

    # Invariantes después de los efectos: la de la acción que empieza y las afectadas.
    after_checks: list[NumericCondition] = []
    invariants = set(state.invariants)
    if kind_start and model.invariants[durative]:
        after_checks.extend(model.invariants[durative])
        invariants.add(durative)
    if kind_end:
        invariants.discard(durative)
    for other in touched:
        after_checks.extend(model.invariants[other])
    if not rows_for(after_checks, after_value, after):
        return None
    changed = bool(new_values) or len(rows) != len(state.rows)
    return LinearState(chains, tuple(rows), rates, frozenset(invariants)), changed


def goal_rows(
    model: LinearModel,
    state: LinearState,
    conditions: Sequence[NumericCondition],
    values: Sequence[float | None],
) -> list[Row] | None:
    """Filas de las condiciones del objetivo sobre variables lineales. En un estado objetivo
    no hay acciones en curso, así que cada variable vale lo que tras su último cambio."""
    rows: list[Row] = []
    for condition in conditions:
        value = linearize(condition.expression, model.symbolic, values, lambda v: state.value(v, 0))
        if value is None:
            return None
        rows.extend(condition_rows(condition, value))
    return rows


def solve(
    state: LinearState,
    out: Sequence[Sequence[tuple[int, float]]],
    initial: Sequence[float | None],
    extra: Sequence[Row] = (),
) -> dict[Symbol, float] | None:
    """Resuelve el programa (con las restricciones de la red temporal ``out`` y las filas
    ``extra``). Devuelve el valor de cada incógnita, o ``None`` si no tiene solución."""
    steps = len(out)
    symbols: dict[Symbol, int] = {("t", j): j for j in range(steps)}
    for variable, chain in state.chains.items():
        for k in range(len(chain)):
            symbols.setdefault(("y", variable, k), len(symbols))
    for row in (*state.rows, *extra):
        for symbol, _ in row.terms:
            if symbol not in symbols:  # y_v,0 de una variable que no ha cambiado
                symbols[symbol] = len(symbols)
    columns = len(symbols)
    upper: list[np.ndarray] = []
    upper_bound: list[float] = []
    equal: list[np.ndarray] = []
    equal_bound: list[float] = []
    for i, edges in enumerate(out):
        for j, weight in edges:
            vector = np.zeros(columns)
            vector[i] += 1.0
            vector[j] -= 1.0
            upper.append(vector)
            upper_bound.append(-weight)
    for constraint in (*state.rows, *extra):
        vector = np.zeros(columns)
        for symbol, weight in constraint.terms:
            vector[symbols[symbol]] += weight
        (equal if constraint.equality else upper).append(vector)
        (equal_bound if constraint.equality else upper_bound).append(-constraint.constant)
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * columns
    bounds[0] = (0.0, 0.0)
    for symbol, column in symbols.items():
        if symbol[0] == "y":
            bounds[column] = (None, None)
            if symbol[2] == 0:  # type: ignore[misc]
                value = initial[symbol[1]]
                if value is None:
                    return None
                bounds[column] = (value, value)
    cost = np.zeros(columns)
    cost[1:steps] = 1.0
    result = linprog(
        cost,
        A_ub=np.array(upper) if upper else None,
        b_ub=np.array(upper_bound) if upper else None,
        A_eq=np.array(equal) if equal else None,
        b_eq=np.array(equal_bound) if equal else None,
        bounds=bounds,
        method="highs",
    )
    if result.status != 0:
        return None
    solution = result.x
    return {symbol: float(solution[column]) for symbol, column in symbols.items()}


def last_values(
    state: LinearState, solution: Mapping[Symbol, float], variables: frozenset[int]
) -> dict[int, float]:
    """Valor de cada variable lineal justo después de su último cambio."""
    values = {}
    for variable in variables:
        chain = state.chains.get(variable)
        if chain is not None:
            values[variable] = solution[("y", variable, len(chain) - 1)]
    return values
