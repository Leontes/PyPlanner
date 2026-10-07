"""Integración de los procesos (PDDL+): cómo cambian las variables mientras pasa el tiempo.

Durante una espera, los procesos activos fijan la derivada de cada variable que cambian:
dxᵢ/dt = fᵢ(x), la suma de los ritmos (con su signo) de los procesos que la tocan. Las
demás variables no cambian. El **flujo** (:class:`Flow`) calcula los valores tras un
tiempo Δ.

**Exacto con series de Taylor.** xᵢ(t + Δ) = Σₖ Δᵏ/k! · Dᵏxᵢ, donde D es la derivada a lo
largo del flujo (D g = Σⱼ ∂g/∂xⱼ · fⱼ). Si los ritmos son polinomios, cada Dᵏxᵢ también, y en
dinámicas como las de car (la velocidad crece con la aceleración y la distancia con la
velocidad) o generator (un caudal que crece con el cuadrado del tiempo) la serie
**termina**: a partir de cierto orden las derivadas son 0 y la suma es exacta, como la que
hace VAL.

**Runge-Kutta 4** cuando no termina (ritmos no polinómicos, con divisiones por variables,
o crecimientos exponenciales como dx/dt = x): una aproximación con subpasos.

Los valores se redondean a :data:`DECIMALS` decimales: así las sumas de pasos (0,01 + 1 + …)
no arrastran ruido de coma flotante, las igualdades como ``(= (v) 0)`` se pueden cumplir y
dos estados iguales lo son también para la detección de repetidos.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from pyplanner.grounding.numeric import Const, NumExpr, Op, Var, evaluate

#: Decimales a los que se redondean los valores numéricos tras cada cambio.
DECIMALS = 9
#: Orden máximo de la serie de Taylor: si a este orden las derivadas no se anulan, la
#: dinámica no es polinómica de grado bajo y se integra con Runge-Kutta.
MAX_ORDER = 12
#: Subpasos de Runge-Kutta por unidad de tiempo (y como mínimo uno por espera).
RK4_STEPS_PER_UNIT = 100
#: Grado máximo de una expresión a lo largo del flujo para buscar dónde se anula.
MAX_CROSSING_DEGREE = 4 * MAX_ORDER

#: Monomio: pares (variable, exponente) ordenados por variable; ``()`` es la constante.
Monomial = tuple[tuple[int, int], ...]
#: Polinomio: monomio → coeficiente (sin coeficientes nulos).
Polynomial = dict[Monomial, float]


def polynomial(expr: NumExpr) -> Polynomial | None:
    """El polinomio de una expresión, o ``None`` si no lo es (divide por una variable)."""
    if isinstance(expr, Const):
        return {(): expr.value} if expr.value else {}
    if isinstance(expr, Var):
        return {((expr.key, 1),): 1.0}
    assert isinstance(expr, Op)
    parts = [polynomial(arg) for arg in expr.args]
    if any(p is None for p in parts):
        return None
    polys: list[Polynomial] = parts  # type: ignore[assignment]
    if expr.op == "+":
        result: Polynomial = {}
        for p in polys:
            result = _add(result, p)
        return result
    if expr.op == "-":
        if len(polys) == 1:
            return _scale(polys[0], -1.0)
        return _add(polys[0], _scale(polys[1], -1.0))
    if expr.op == "*":
        result = {(): 1.0}
        for p in polys:
            result = _multiply(result, p)
        return result
    # División: solo por una constante distinta de 0.
    numerator, denominator = polys
    if set(denominator) != {()}:
        return None
    return _scale(numerator, 1.0 / denominator[()])


def _add(p: Polynomial, q: Polynomial) -> Polynomial:
    result = dict(p)
    for monomial, coefficient in q.items():
        total = result.get(monomial, 0.0) + coefficient
        if total:
            result[monomial] = total
        else:
            result.pop(monomial, None)
    return result


def _scale(p: Polynomial, factor: float) -> Polynomial:
    return {m: c * factor for m, c in p.items() if c * factor} if factor else {}


def _multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            result = _add(result, {_monomial_product(m1, m2): c1 * c2})
    return result


def _monomial_product(m1: Monomial, m2: Monomial) -> Monomial:
    exponents = dict(m1)
    for var, exponent in m2:
        exponents[var] = exponents.get(var, 0) + exponent
    return tuple(sorted(exponents.items()))


def _derivative(p: Polynomial, var: int) -> Polynomial:
    """∂p/∂var."""
    result: Polynomial = {}
    for monomial, coefficient in p.items():
        exponents = dict(monomial)
        exponent = exponents.get(var, 0)
        if not exponent:
            continue
        if exponent == 1:
            del exponents[var]
        else:
            exponents[var] = exponent - 1
        result = _add(result, {tuple(sorted(exponents.items())): coefficient * exponent})
    return result


def _along(p: Polynomial, rates: Mapping[int, Polynomial]) -> Polynomial:
    """Derivada de ``p`` a lo largo del flujo: Σⱼ ∂p/∂xⱼ · fⱼ."""
    result: Polynomial = {}
    for var, rate in rates.items():
        partial = _derivative(p, var)
        if partial:
            result = _add(result, _multiply(partial, rate))
    return result


def _evaluate(p: Polynomial, values: Sequence[float | None]) -> float | None:
    total = 0.0
    for monomial, coefficient in p.items():
        term = coefficient
        for var, exponent in monomial:
            value = values[var]
            if value is None:
                return None
            term *= value**exponent
        total += term
    return total


class Flow:
    """Cambio continuo de las variables con un conjunto fijo de procesos activos.

    Args:
        rates: derivada de cada variable que cambia (la suma de los ritmos con su signo).
    """

    def __init__(self, rates: Mapping[int, NumExpr]) -> None:
        self.rates = dict(rates)
        self._polynomials = self._rate_polynomials()
        #: Por variable, sus derivadas sucesivas D¹x, D²x… hasta la última no nula; ``None``
        #: si la serie no termina (se integra con Runge-Kutta).
        self.series: dict[int, list[Polynomial]] | None = self._taylor()
        self._expressions: dict[int, tuple[NumExpr, list[Polynomial] | None]] = {}

    def _rate_polynomials(self) -> dict[int, Polynomial] | None:
        polys: dict[int, Polynomial] = {}
        for var, rate in self.rates.items():
            poly = polynomial(rate)
            if poly is None:
                return None
            polys[var] = poly
        return polys

    def _taylor(self) -> dict[int, list[Polynomial]] | None:
        polys = self._polynomials
        if polys is None:
            return None
        series: dict[int, list[Polynomial]] = {}
        for var, derivative in polys.items():
            terms = [derivative] if derivative else []
            while terms:
                following = _along(terms[-1], polys)
                if not following:
                    break
                if len(terms) == MAX_ORDER:
                    return None  # no termina: no es exacta
                terms.append(following)
            series[var] = terms
        return series

    def crossings(
        self, expression: NumExpr, values: Sequence[float | None], duration: float
    ) -> list[float] | None:
        """Los instantes de (0, ``duration``) en que ``expression`` se anula a lo largo del
        flujo, ordenados: entre dos seguidos no cambia de signo. ``None`` si no se pueden calcular
        (el flujo no es exacto, la expresión no es un polinomio o falta algún valor).

        Con un flujo exacto, la expresión es un polinomio en el tiempo: g(t) = Σₖ tᵏ/k! ·
        Dᵏg, con las derivadas a lo largo del flujo (como las de las variables).
        """
        series = self._expression_series(expression)
        if series is None:
            return None
        coefficients = []
        factorial = 1.0
        for k, term in enumerate(series):
            factorial *= max(k, 1)
            value = _evaluate(term, values)
            if value is None:
                return None
            coefficients.append(value / factorial)
        return sorted(t for t in _real_roots(coefficients) if 0 < t < duration)

    def _expression_series(self, expression: NumExpr) -> list[Polynomial] | None:
        """D⁰g, D¹g… hasta la última no nula (se calcula una vez por expresión; la caché
        guarda la expresión para que su identificador no se reutilice)."""
        cached = self._expressions.get(id(expression))
        if cached is not None:
            return cached[1]
        polys, poly = self._polynomials, polynomial(expression)
        terms: list[Polynomial] | None = None
        if self.series is not None and polys is not None and poly is not None:
            terms = [poly]
            while terms is not None:
                following = _along(terms[-1], polys)
                if not following:
                    break
                terms = terms + [following] if len(terms) <= MAX_CROSSING_DEGREE else None
        self._expressions[id(expression)] = (expression, terms)
        return terms

    @property
    def exact(self) -> bool:
        """``True`` si el flujo se integra de forma exacta (serie de Taylor finita)."""
        return self.series is not None

    def advance(
        self, values: tuple[float | None, ...], duration: float
    ) -> tuple[float | None, ...] | None:
        """Los valores tras ``duration`` unidades de tiempo, o ``None`` si algún ritmo es
        indefinido. Se redondean los que cambian (los demás ya lo están: los estados se
        redondean al crearse)."""
        result = list(values)
        if self.series is not None:
            for var, terms in self.series.items():
                current = values[var]
                if current is None:
                    return None
                total, factor = current, 1.0
                for k, term in enumerate(terms, start=1):
                    factor *= duration / k
                    value = _evaluate(term, values)
                    if value is None:
                        return None
                    total += factor * value
                result[var] = round(total, DECIMALS)
            return tuple(result)
        integrated = self._runge_kutta(values, duration)
        if integrated is None:
            return None
        for var in self.rates:
            value = integrated[var]
            integrated[var] = None if value is None else round(value, DECIMALS)
        return tuple(integrated)

    def _runge_kutta(
        self, values: tuple[float | None, ...], duration: float
    ) -> list[float | None] | None:
        steps = max(1, math.ceil(duration * RK4_STEPS_PER_UNIT))
        h = duration / steps
        state = list(values)
        variables = list(self.rates)

        def slope(point: Sequence[float | None]) -> list[float] | None:
            result = []
            for var in variables:
                value = evaluate(self.rates[var], point)
                if value is None:
                    return None
                result.append(value)
            return result

        for _ in range(steps):
            k1 = slope(state)
            if k1 is None:
                return None
            k2 = slope(_shift(state, variables, k1, h / 2))
            if k2 is None:
                return None
            k3 = slope(_shift(state, variables, k2, h / 2))
            if k3 is None:
                return None
            k4 = slope(_shift(state, variables, k3, h))
            if k4 is None:
                return None
            for i, var in enumerate(variables):
                current = state[var]
                assert current is not None
                state[var] = current + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        return state


def _real_roots(coefficients: Sequence[float]) -> list[float]:
    """Las raíces reales de Σₖ cₖ·tᵏ (con fórmula hasta grado 2)."""
    degree = len(coefficients) - 1
    while degree > 0 and coefficients[degree] == 0:
        degree -= 1
    if degree == 0:
        return []
    if degree == 1:
        return [-coefficients[0] / coefficients[1]]
    if degree == 2:
        c, b, a = coefficients[:3]
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []
        # Forma estable: sin restar números parecidos.
        q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
        return [q / a, c / q] if q else [0.0]
    roots = np.roots(list(reversed(coefficients[: degree + 1])))
    return [float(r.real) for r in roots if abs(r.imag) <= 1e-9 * max(1.0, abs(r.real))]


def _shift(
    state: list[float | None], variables: list[int], slope: list[float], h: float
) -> list[float | None]:
    point = list(state)
    for i, var in enumerate(variables):
        current = point[var]
        assert current is not None
        point[var] = current + h * slope[i]
    return point
