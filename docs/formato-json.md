# Formato JSON de entrada (versión 6)

El planificador recibe los dominios y problemas en JSON. Este documento es el contrato para
cualquier componente que quiera generar ese JSON sin pasar por PDDL. El esquema formal
está en [`src/pyplanner/schema/pddl.schema.json`](../src/pyplanner/schema/pddl.schema.json)
(JSON Schema 2020-12). Si este documento y el esquema no coinciden, manda el esquema.

## Flujo

```
fichero PDDL ──(fase 1: pyplanner.pddl)──► JSON ──(fase 2: pyplanner.loader)──► AST
                                            ▲
                     otro componente ───────┘
```

- **Fase 1** (`parse_pddl`, `parse_file`, `pyplanner parse`): solo sintaxis. Conserva la
  estructura del PDDL.
- **Fase 2** (`load_domain`, `load_problem`): valida el documento con el esquema, comprueba
  los requisitos y hace la validación semántica. No da por bueno el JSON que recibe.
- **Conversor inverso** (`write_pddl`): JSON → PDDL, por ejemplo para validar con VAL.

## Convenciones

- Cada documento es un dominio (`"kind": "domain"`), un problema (`"kind": "problem"`) o
  un plan (`"kind": "plan"`, la salida del planificador).
- `version` es la versión del formato:
  - v1: STRIPS;
  - v2: PDDL 1.2, subconjunto ADL;
  - v3: PDDL 2.1 numérico y costes de acción (`:action-costs`);
  - v4: PDDL 2.1 temporal (acciones durativas);
  - v5: PDDL 2.2 (predicados derivados y literales iniciales temporizados);
  - v6 (actual): PDDL 3 (restricciones de trayectoria, preferencias e `is-violated`).

  Cada versión solo **añade** construcciones, así que un documento v1 sigue siendo
  válido.
- Los nombres siguen el patrón `[A-Za-z][A-Za-z0-9_-]*`. La fase 2 los pasa a minúsculas,
  porque PDDL no distingue mayúsculas.
- Los argumentos de un átomo son cadenas: una **variable** si empieza por `?` y un
  **objeto o constante** en otro caso.
- Los nodos de fórmulas y efectos llevan un campo `op` que indica su tipo.
- Cualquier nodo puede llevar `loc` (`{"line", "column"}`) con su posición en el PDDL
  original, y el documento puede llevar `source` con el nombre del fichero. Ambos son
  opcionales y solo sirven para los mensajes de error.
- No se admiten campos que no estén en el esquema (`additionalProperties: false`).

## Dominio

| Campo | Obligatorio | Contenido |
|-------|-------------|-----------|
| `version`, `kind`, `name` | sí | |
| `requirements` | no | lista de requisitos (`":strips"`…). Vacía equivale a `:strips` |
| `types` | no | `[{"name": "truck", "parent": "vehicle"}, {"name": "place"}, ...]`; sin `parent`, el supertipo es `object` |
| `constants` | no | `[{"name": "home", "type": "place"}, ...]` |
| `predicates` | no | `[{"name": "at", "parameters": [{"name": "?x", "type": "truck"}, ...]}, ...]` |
| `functions` | no | funciones numéricas, con la misma forma que `predicates` (v3) |
| `actions` | no | ver abajo |
| `durative_actions` | no | acciones durativas (v4); ver [Acciones durativas](#acciones-durativas-v4) |
| `derived` | no | reglas de predicados derivados (v5); ver [PDDL 2.2](#pddl-22-v5) |
| `constraints` | no | restricciones de trayectoria duras (v6); ver [PDDL 3](#pddl-3-v6) |

Una acción tiene `name`, `parameters` y, opcionalmente, `precondition` (una fórmula) y
`effect` (un efecto). Si falta alguno de los dos, se toma como vacío. Las acciones y las
acciones durativas comparten espacio de nombres.

## Problema

| Campo | Obligatorio | Contenido |
|-------|-------------|-----------|
| `version`, `kind`, `name` | sí | |
| `domain` | sí | nombre del dominio; tiene que coincidir con el dominio cargado |
| `goal` | sí | fórmula |
| `requirements` | no | se suman a los del dominio |
| `objects` | no | como `constants`. Si repiten una constante del dominio, se ignoran |
| `init` | no | lista de átomos sin variables, de valores iniciales `{"op": "=", "fluent": …, "value": 3.5}` (v3) y de literales iniciales temporizados `{"op": "at", "time": 10, "literal": …}` (v5) |
| `metric` | no | `{"direction": "minimize" \| "maximize", "expression": …}` (v3) |
| `constraints` | no | restricciones de trayectoria, que pueden ser preferencias (v6); ver [PDDL 3](#pddl-3-v6) |

## Tipos (v2)

- `type` es opcional en constantes, objetos y parámetros (de acciones, predicados y
  cuantificadores). Si falta, el tipo es `object`.
- Una lista de tipos equivale a `(either ...)`: `{"name": "?x", "type": ["truck", "plane"]}`.
- Un supertipo que no se declara como tipo se declara implícitamente, con supertipo
  `object`.
- La fase 2 comprueba que las constantes y los objetos que aparecen en los átomos tienen
  un tipo compatible con el del predicado.

## Modo estricto

Toda construcción tiene que tener su requisito declarado, en el dominio o en el problema.
Las reglas están en `pyplanner/requirements.py`:

| Construcción | Requisito |
|--------------|-----------|
| `types`, o `type` en cualquier declaración | `:typing` |
| `not` sobre un átomo o una igualdad, en una fórmula | `:negative-preconditions` |
| `not` sobre una fórmula compuesta, `or`, `imply` | `:disjunctive-preconditions` |
| `exists` | `:existential-preconditions` |
| `forall` en una fórmula | `:universal-preconditions` |
| `=` | `:equality` |
| `forall` o `when` en un efecto | `:conditional-effects` |
| `functions`, valores iniciales, `compare`, efectos numéricos, `metric` | `:numeric-fluents` |
| perfil de costes de acción: `functions` y sus valores iniciales, `increase` de `(total-cost)` con un número o una función, `minimize (total-cost)` | `:numeric-fluents` o `:action-costs` |
| `durative_actions` | `:durative-actions` |
| duración con `<=` o `>=`, varias restricciones o `time` | `:duration-inequalities` |
| efecto continuo | `:continuous-effects` |
| `derived` | `:derived-predicates` |
| literal inicial temporizado (`at` en `init`) | `:timed-initial-literals` |
| `constraints` y sus operadores modales | `:constraints` |
| `preference`, `is-violated` | `:preferences` |
| métrica de números, aritmética e `is-violated` | `:numeric-fluents` o `:preferences` |

- `not` en un efecto es un borrado STRIPS y no necesita nada.
- `:adl` y `:quantified-preconditions` equivalen a sus requisitos componentes, y
  `:fluents` equivale a `:numeric-fluents`.

## Fórmulas y efectos

| `op` | Campos | Dónde puede aparecer | Desde |
|------|--------|----------------------|-------|
| `atom` | `predicate`, `args` | fórmulas, efectos, `init` | v1 |
| `and` | `args` (lista de nodos) | fórmulas, efectos | v1 |
| `not` | `arg` (un átomo) | efectos (borrado) | v1 |
| `not` | `arg` (una fórmula) | fórmulas | v2 |
| `or` | `args` | fórmulas | v2 |
| `imply` | `args` (exactamente 2) | fórmulas | v2 |
| `exists`, `forall` | `parameters` (al menos 1), `arg` (una fórmula) | fórmulas | v2 |
| `=` | `args` (exactamente 2 términos) | fórmulas | v2 |
| `forall` | `parameters`, `arg` (un efecto) | efectos | v2 |
| `when` | `condition` (fórmula), `arg` (solo `atom`, `not`, `and` y efectos numéricos) | efectos | v2 |
| `compare` | `comparator` (`<`, `<=`, `=`, `>=`, `>`), `left`, `right` (expresiones) | fórmulas | v3 |
| `increase`, `decrease`, `assign`, `scale-up`, `scale-down` | `fluent` (término de función), `value` (expresión) | efectos | v3 |

## Expresiones numéricas (v3)

| `op` | Campos | Notas |
|------|--------|-------|
| `num` | `value` | número |
| `fluent` | `function`, `args` | término de función, p. ej. `{"op": "fluent", "function": "fuel", "args": ["?a"]}` |
| `+`, `*` | `args` (dos o más) | |
| `-` | `args` (uno o dos) | con uno, cambio de signo |
| `/` | `args` (dos) | |
| `total-time` | — | duración del plan; **solo en la métrica**. En un plan secuencial, cada acción dura una unidad |
| `duration` | — | `?duration`, la duración de la acción (v4); **solo dentro de una acción durativa**, y no en su propia duración |
| `#t` | — | tiempo transcurrido (v4); **solo en un efecto continuo** |

- `=` entre dos objetos o variables es la igualdad de `:equality` (`{"op": "=", "args": […]}`);
  entre expresiones es una comparación (`{"op": "compare", "comparator": "=", …}`).
- Los efectos numéricos se evalúan con los valores del estado **anterior** a la acción.
- Una función sin valor inicial está indefinida.

Otras restricciones:

- Una variable cuantificada no puede llevar el nombre de otra variable ya ligada (un
  parámetro o un cuantificador exterior). Dos cuantificadores hermanos sí pueden usar el
  mismo nombre.
- La condición de `when` se evalúa sobre el estado **anterior** a la acción.

### Tabla de la versión 1 (sin cambios)

| `op` | Campos | Dónde puede aparecer |
|------|--------|----------------------|
| `atom` | `predicate`, `args` | fórmulas, efectos, `init` |
| `and` | `args` (lista de nodos) | fórmulas, efectos |
| `not` | `arg` (un átomo) | solo efectos (borrado) |

## Acciones durativas (v4)

Una acción durativa tiene `name`, `parameters`, `duration` y, opcionalmente, `condition` y
`effect` (si faltan, son vacíos):

```json
{
  "name": "drive",
  "parameters": [{"name": "?t", "type": "truck"}, {"name": "?from", "type": "place"}, {"name": "?to", "type": "place"}],
  "duration": [{"comparator": "=", "value": {"op": "fluent", "function": "time-to-drive", "args": ["?from", "?to"]}}],
  "condition": {"op": "and", "args": [
    {"op": "at-start", "arg": {"op": "atom", "predicate": "at", "args": ["?t", "?from"]}},
    {"op": "over-all", "arg": {"op": "atom", "predicate": "road", "args": ["?from", "?to"]}}
  ]},
  "effect": {"op": "and", "args": [
    {"op": "at-start", "arg": {"op": "not", "arg": {"op": "atom", "predicate": "at", "args": ["?t", "?from"]}}},
    {"op": "at-end", "arg": {"op": "atom", "predicate": "at", "args": ["?t", "?to"]}},
    {"op": "decrease", "fluent": {"op": "fluent", "function": "fuel", "args": ["?t"]},
     "value": {"op": "*", "args": [{"op": "#t"}, {"op": "num", "value": 2}]}}
  ]}
}
```

- **`duration`** es la conjunción de las restricciones de duración, cada una
  `{"comparator": "=" | "<=" | ">=", "value": <expresión>}`. Con `"time": "start" | "end"`,
  la expresión se evalúa al empezar o al terminar la acción (`(at end (<= ?duration …))`);
  sin `time`, al empezar. Vacía si la acción no tiene restricciones (`:duration ()`).
- **Condiciones** (`condition`):

  | `op` | Campos | Significado |
  |------|--------|-------------|
  | `at-start`, `at-end` | `arg` (fórmula) | cierta al empezar / al terminar |
  | `over-all` | `arg` (fórmula) | cierta durante toda la acción (sin incluir los extremos) |
  | `and` | `args` (condiciones) | |
  | `forall` | `parameters`, `arg` (una condición) | |

- **Efectos** (`effect`):

  | `op` | Campos | Significado |
  |------|--------|-------------|
  | `at-start`, `at-end` | `arg` (`atom`, `not`, `and` o efecto numérico) | al empezar / al terminar |
  | `increase`, `decrease` | `fluent`, `value` con `#t` | **efecto continuo**: `value` es `#t` o un producto con `#t` como factor (`(* #t 2)`); la función cambia a ese ritmo mientras dura la acción |
  | `and` | `args` (efectos) | |
  | `forall` | `parameters`, `arg` (un efecto) | |
  | `when` | `condition` (una condición con momentos), `arg` (efectos con momento o continuos) | un efecto `at start` solo puede depender de condiciones `at start` |

- `?duration` (`{"op": "duration"}`) puede aparecer en los valores de los efectos numéricos
  de la acción, pero no en sus condiciones ni en su propia duración. Así lo define la
  gramática de PDDL 2.1, y VAL no valida planes de dominios que lo usen en condiciones.
- En PDDL, `(at start X)` se distingue de un átomo del predicado `at` porque `X` es una
  lista.
- El conversor inverso escribe una condición o un efecto ausentes como `(and)`, porque VAL
  no acepta `:condition ()`.

## PDDL 2.2 (v5)

**Predicados derivados.** Cada regla `(:derived (p ?x - t …) <fórmula>)` es un elemento de
`derived`:

```json
{"predicate": "fed", "parameters": [{"name": "?l", "type": "line"}],
 "condition": {"op": "exists", "parameters": [{"name": "?x", "type": "device"}],
               "arg": {"op": "atom", "predicate": "closed", "args": ["?x"]}}}
```

- El predicado se declara también en `predicates`. La regla tiene su misma aridad, y cada
  parámetro, un tipo compatible con el declarado.
- Varias reglas del mismo predicado se suman: el átomo es cierto si lo hace cierto alguna.
- La condición es una fórmula cualquiera, con las variables de la cabeza ligadas, y puede
  usar otros predicados derivados (también el suyo: reglas recursivas).
- Un predicado derivado puede aparecer en precondiciones, condiciones y objetivos, pero
  **no en efectos ni en `init`**: su valor se deduce siempre de los demás hechos.
- Las reglas tienen que ser **estratificables**: ningún predicado derivado puede depender
  negativamente de sí mismo, directa o indirectamente. Una dependencia es negativa si el
  átomo está bajo un número impar de `not` (el antecedente de `imply` cuenta como uno).

**Literales iniciales temporizados** (TIL). `(at 10 (visible a s))` en `:init` es
`{"op": "at", "time": 10, "literal": <átomo>}`, y `(at 20 (not (visible a s)))` lleva
`{"op": "not", "arg": <átomo>}` como `literal`.

- En el instante `time` (≥ 0), el hecho pasa a ser cierto (o falso, con `not`) sin que lo
  cause ninguna acción.
- Solo en dominios temporales (`:durative-actions`): en un plan secuencial no hay tiempo.
- No pueden afectar a predicados derivados, ni dar a un hecho los dos valores en el mismo
  instante. Repetir un mismo literal no tiene efecto.
- En PDDL, `(at <número> …)` es siempre un TIL: un objeto no puede llamarse como un número.

## PDDL 3 (v6)

**Preferencias.** `{"op": "preference", "name": "p1", "arg": …}`; `name` es opcional (una
preferencia sin nombre no se puede citar en la métrica). Varias preferencias pueden
compartir nombre, y una dentro de un `forall` lo comparte con todas sus instancias. Solo
pueden ir:

- en la conjunción exterior (`and`, `forall`) del objetivo y de las precondiciones, con una
  fórmula como `arg`;
- en las condiciones de una acción durativa, envolviendo una condición con momento
  (`at-start`, `at-end` u `over-all`);
- en las restricciones del problema, envolviendo una restricción.

Nunca dentro de `or`, `not`, `imply`, `exists` ni de otra preferencia.

**Restricciones de trayectoria.** Se cumplen sobre la secuencia de estados del plan (el
inicial y el que deja cada paso). Los operadores modales no se anidan: van directamente en
`constraints` o dentro de `and`, `forall` y (solo en el problema) `preference`.

| `op` | Campos | Significado |
|------|--------|-------------|
| `at-end` | `arg` | φ en el estado final |
| `always` | `arg` | φ en todos los estados |
| `sometime` | `arg` | φ en algún estado |
| `at-most-once` | `arg` | φ es cierta en un solo tramo de estados seguidos, como mucho |
| `sometime-after` | `args` [φ, ψ] | si φ en un estado, ψ en ese o en uno posterior |
| `sometime-before` | `args` [φ, ψ] | si φ en un estado, ψ en uno estrictamente anterior |
| `within` | `time`, `arg` | φ en algún estado de instante ≤ `time` |
| `hold-after` | `time`, `arg` | φ en todos los estados de instante > `time` |
| `hold-during` | `from`, `to`, `arg` | φ en todos los estados de instante en [`from`, `to`) |
| `always-within` | `time`, `args` [φ, ψ] | cada vez que φ, ψ en un plazo de `time` |

En un dominio sin acciones durativas, el instante de un estado es el número de pasos del
plan hasta él. Los instantes son ≥ 0, y en `hold-during`, `from` ≤ `to`. Las restricciones
del dominio son siempre duras: no admiten preferencias.

**`is-violated`.** `{"op": "is-violated", "name": "p1"}` es el número de incumplimientos de
la preferencia `p1` (de todas sus instancias), y solo puede aparecer en la métrica. El
nombre tiene que ser el de alguna preferencia del dominio o del problema.

## Ejemplo

```json
{
  "version": 1,
  "kind": "domain",
  "name": "blocks",
  "requirements": [":strips"],
  "predicates": [
    {"name": "holding", "parameters": [{"name": "?x"}]},
    {"name": "clear", "parameters": [{"name": "?x"}]},
    {"name": "ontable", "parameters": [{"name": "?x"}]},
    {"name": "handempty", "parameters": []}
  ],
  "actions": [
    {
      "name": "put-down",
      "parameters": [{"name": "?x"}],
      "precondition": {"op": "atom", "predicate": "holding", "args": ["?x"]},
      "effect": {"op": "and", "args": [
        {"op": "not", "arg": {"op": "atom", "predicate": "holding", "args": ["?x"]}},
        {"op": "atom", "predicate": "clear", "args": ["?x"]},
        {"op": "atom", "predicate": "handempty", "args": []},
        {"op": "atom", "predicate": "ontable", "args": ["?x"]}
      ]}
    }
  ]
}
```

## Plan (salida)

El planificador devuelve los planes con `"kind": "plan"` (`pyplanner solve -f json`):

```json
{
  "version": 3,
  "kind": "plan",
  "domain": "blocks",
  "problem": "blocks-4-0",
  "cost": 6,
  "steps": [
    {"name": "pick-up", "args": ["b"]},
    {"name": "stack", "args": ["b", "a"]}
  ]
}
```

- `cost`: suma de los costes de las acciones. Sin una métrica que se pueda usar como
  coste, es el número de acciones.
- `metric` (opcional, v3): valor de la métrica del problema, cuando los costes de las
  acciones la representan (ver [motor-busqueda.md](motor-busqueda.md#motor-numérico-iteración-3a)).
  Es el valor inicial de la métrica más `cost`, y coincide con el que calcula VAL.

## Errores

Todos los errores heredan de `pyplanner.errors.PDDLError` y llevan, si se conocen, la
posición en el PDDL (`location`) y la ruta en el JSON (`path`, p. ej.
`actions[0].effect.args[1]`).

| Error | Fase | Causa |
|-------|------|-------|
| `PDDLSyntaxError` | 1 | PDDL mal formado |
| `SchemaValidationError` | 2 | el JSON no cumple el esquema, o no es JSON |
| `UnsupportedFeatureError` | 1 y 2 | requisito o construcción que esta versión no soporta |
| `UndeclaredRequirementError` | 1 (dominios) y 2 | construcción usada sin declarar su requisito (modo estricto) |
| `SemanticError` | 2 | predicado sin declarar, aridad incorrecta, variable libre, objeto desconocido, nombre repetido, dominio que no coincide, tipo no declarado o incompatible, ciclo de tipos… |
