"""Expresiones, condiciones y efectos numéricos instanciados.

Tras el grounding, una expresión numérica solo contiene constantes y **variables del
estado**. Las funciones estáticas ya se han sustituido por su valor. Durante la
normalización las variables se identifican por su término (``("fuel", ("plane1",))``);
al construir la tarea se renumeran a índices de ``GroundTask.variables``
(:func:`remap`).

Formas canónicas:

* una **condición** es ``expresión ⋈ 0`` con ``⋈`` ∈ {``>=``, ``>``, ``=``}. ``a < b`` se
  guarda como ``b - a > 0``, y así con el resto;
* un **efecto** es ``(variable, operación, expresión)``, con la operación de PDDL
  (``increase``, ``decrease``, ``assign``, ``scale-up``, ``scale-down``).

Cuando una expresión es **lineal** (Σ wᵢ·xᵢ + c) se guarda también su forma lineal. Es
más rápida de evaluar y es la que usa la heurística numérica.

**Valores indefinidos.** Una variable sin valor inicial vale ``None``. Cualquier
expresión que la use es indefinida (``None``), y una condición indefinida es falsa. No se
usa NaN en el estado, porque en Python cada NaN tiene un hash distinto y rompería la
detección de estados repetidos.
"""

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Identificador de una variable: un término durante la normalización, un índice después.
VarKey = Hashable


@dataclass(frozen=True)
class Const:
    """Constante numérica."""

    value: float


@dataclass(frozen=True)
class Var:
    """Variable numérica del estado."""

    key: Any


@dataclass(frozen=True)
class Op:
    """Operación aritmética (``+``, ``-``, ``*``, ``/``; ``-`` con un argumento cambia el signo)."""

    op: str
    args: tuple["NumExpr", ...]


NumExpr = Const | Var | Op


@dataclass(frozen=True)
class Linear:
    """Forma lineal Σ wᵢ·xᵢ + c, con los términos ordenados por variable."""

    terms: tuple[tuple[Any, float], ...]
    constant: float

    def evaluate(self, values: Sequence[float | None]) -> float | None:
        total = self.constant
        for index, weight in self.terms:
            value = values[index]
            if value is None:
                return None
            total += weight * value
        return total


# --- Construcción y simplificación ---------------------------------------------------


def make(op: str, args: Sequence[NumExpr]) -> NumExpr:
    """Construye una operación, plegando las constantes cuando todos los argumentos lo son."""
    if all(isinstance(a, Const) for a in args):
        value = _apply(op, [a.value for a in args])  # type: ignore[union-attr]
        if value is not None:
            return Const(value)
    return Op(op, tuple(args))


def negate(expr: NumExpr) -> NumExpr:
    return make("-", [expr])


def subtract(left: NumExpr, right: NumExpr) -> NumExpr:
    if isinstance(right, Const) and right.value == 0:
        return left
    return make("-", [left, right])


def _apply(op: str, values: Sequence[float]) -> float | None:
    if op == "+":
        return sum(values)
    if op == "*":
        result = 1.0
        for v in values:
            result *= v
        return result
    if op == "-":
        return -values[0] if len(values) == 1 else values[0] - values[1]
    if op == "/":
        return None if values[1] == 0 else values[0] / values[1]
    raise ValueError(f"operador desconocido: {op}")


# --- Evaluación ----------------------------------------------------------------------


def evaluate(expr: NumExpr, values: Any) -> float | None:
    """Valor de la expresión (variables por índice), o ``None`` si es indefinida."""
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        value: float | None = values[expr.key]
        return value
    args = []
    for arg in expr.args:
        value = evaluate(arg, values)
        if value is None:
            return None
        args.append(value)
    return _apply(expr.op, args)


def variables_of(expr: NumExpr) -> set[Any]:
    """Variables que aparecen en la expresión."""
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Var):
        return {expr.key}
    result: set[Any] = set()
    for arg in expr.args:
        result |= variables_of(arg)
    return result


def linear(expr: NumExpr) -> Linear | None:
    """Forma lineal de la expresión, o ``None`` si no es lineal."""
    result = _linear(expr)
    if result is None:
        return None
    terms, constant = result
    ordered = tuple(sorted((k, w) for k, w in terms.items() if w != 0))
    return Linear(ordered, constant)


def _linear(expr: NumExpr) -> tuple[dict[Any, float], float] | None:
    if isinstance(expr, Const):
        return {}, expr.value
    if isinstance(expr, Var):
        return {expr.key: 1.0}, 0.0
    parts = [_linear(arg) for arg in expr.args]
    if any(p is None for p in parts):
        return None
    forms: list[tuple[dict[Any, float], float]] = parts  # type: ignore[assignment]
    if expr.op in ("+", "-"):
        signs = (
            [1.0] * len(forms) if expr.op == "+" else ([-1.0] if len(forms) == 1 else [1.0, -1.0])
        )
        terms: dict[Any, float] = {}
        constant = 0.0
        for sign, (form_terms, form_constant) in zip(signs, forms, strict=True):
            constant += sign * form_constant
            for key, weight in form_terms.items():
                terms[key] = terms.get(key, 0.0) + sign * weight
        return terms, constant
    if expr.op == "*":
        # Lineal solo si como mucho un factor tiene variables.
        variable_parts = [f for f in forms if f[0]]
        if len(variable_parts) > 1:
            return None
        factor = 1.0
        for form_terms, form_constant in forms:
            if not form_terms:
                factor *= form_constant
        if not variable_parts:
            return {}, factor
        base_terms, base_constant = variable_parts[0]
        return {k: w * factor for k, w in base_terms.items()}, base_constant * factor
    if expr.op == "/":
        (numerator_terms, numerator_constant), (denominator_terms, denominator) = forms
        if denominator_terms or denominator == 0:
            return None
        return {
            k: w / denominator for k, w in numerator_terms.items()
        }, numerator_constant / denominator
    return None


def remap(expr: NumExpr, mapping: Callable[[Any], Any]) -> NumExpr:
    """Copia de la expresión con las variables renombradas con ``mapping``."""
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, Var):
        return Var(mapping(expr.key))
    return Op(expr.op, tuple(remap(arg, mapping) for arg in expr.args))


# --- Condiciones y efectos -----------------------------------------------------------


@dataclass(frozen=True)
class NumericCondition:
    """Condición ``expression ⋈ 0`` con ``⋈`` ∈ {``>=``, ``>``, ``=``}."""

    expression: NumExpr
    comparator: str
    linear: Linear | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.linear is None:
            object.__setattr__(self, "linear", linear(self.expression))

    def value(self, values: Sequence[float | None]) -> float | None:
        if self.linear is not None:
            return self.linear.evaluate(values)
        return evaluate(self.expression, values)

    def holds(self, values: Sequence[float | None]) -> bool:
        value = self.value(values)
        if value is None:
            return False
        if self.comparator == ">=":
            return value >= 0
        if self.comparator == ">":
            return value > 0
        return value == 0

    def remap(self, mapping: Callable[[Any], Any]) -> "NumericCondition":
        return NumericCondition(remap(self.expression, mapping), self.comparator)

    def __str__(self) -> str:
        return f"({self.comparator} {_show(self.expression)} 0)"


@dataclass(frozen=True)
class NumericAssignment:
    """Efecto numérico sobre una variable; ``value`` se evalúa en el estado anterior."""

    variable: Any
    op: str
    value: NumExpr

    def new_value(self, values: Sequence[float | None]) -> float | None:
        """Valor de la variable tras el efecto (``None`` si es indefinido)."""
        amount = evaluate(self.value, values)
        if amount is None:
            return None
        if self.op == "assign":
            return amount
        current: float | None = values[self.variable]
        if current is None:
            return None
        if self.op == "increase":
            return current + amount
        if self.op == "decrease":
            return current - amount
        if self.op == "scale-up":
            return current * amount
        return None if amount == 0 else current / amount  # scale-down

    def remap(self, mapping: Callable[[Any], Any]) -> "NumericAssignment":
        return NumericAssignment(mapping(self.variable), self.op, remap(self.value, mapping))

    def __str__(self) -> str:
        return f"({self.op} {self.variable} {_show(self.value)})"


def comparison(
    comparator: str, left: NumExpr, right: NumExpr, negated: bool
) -> list[NumericCondition]:
    """Condiciones canónicas equivalentes a ``left ⋈ right`` (o a su negación).

    Devuelve una lista porque ``not (a = b)`` son dos casos: ``a > b`` o ``a < b``.
    """
    difference = subtract(left, right)
    if negated:
        comparator = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "=": "!="}[comparator]
    if comparator == ">=":
        return [NumericCondition(difference, ">=")]
    if comparator == ">":
        return [NumericCondition(difference, ">")]
    if comparator == "<=":
        return [NumericCondition(negate(difference), ">=")]
    if comparator == "<":
        return [NumericCondition(negate(difference), ">")]
    if comparator == "=":
        return [NumericCondition(difference, "=")]
    return [NumericCondition(difference, ">"), NumericCondition(negate(difference), ">")]


def constant_truth(condition: NumericCondition) -> bool | None:
    """``True`` o ``False`` si la condición no tiene variables; ``None`` en otro caso."""
    if variables_of(condition.expression):
        return None
    return condition.holds(())


def _show(expr: NumExpr) -> str:
    if isinstance(expr, Const):
        return f"{expr.value:g}"
    if isinstance(expr, Var):
        key = expr.key
        if isinstance(key, tuple):
            return f"({' '.join([key[0], *key[1]])})"
        return f"x{key}"
    return f"({' '.join([expr.op, *(_show(a) for a in expr.args)])})"


def evaluate_with(expr: NumExpr, lookup: Mapping[Any, float | None]) -> float | None:
    """Como :func:`evaluate`, pero buscando las variables por clave en ``lookup`` (las que no
    están son indefinidas)."""
    return evaluate(expr, _Lookup(lookup))


class _Lookup:
    """Adaptador para indexar un diccionario como la tupla de valores de un estado."""

    def __init__(self, mapping: Mapping[Any, float | None]) -> None:
        self.mapping = mapping

    def __getitem__(self, key: Any) -> float | None:
        return self.mapping.get(key)
