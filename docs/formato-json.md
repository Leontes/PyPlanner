# Formato JSON de entrada (versión 3)

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
  - v3 (actual): PDDL 2.1 numérico y costes de acción (`:action-costs`).

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

Una acción tiene `name`, `parameters` y, opcionalmente, `precondition` (una fórmula) y
`effect` (un efecto). Si falta alguno de los dos, se toma como vacío.

## Problema

| Campo | Obligatorio | Contenido |
|-------|-------------|-----------|
| `version`, `kind`, `name` | sí | |
| `domain` | sí | nombre del dominio; tiene que coincidir con el dominio cargado |
| `goal` | sí | fórmula |
| `requirements` | no | se suman a los del dominio |
| `objects` | no | como `constants`. Si repiten una constante del dominio, se ignoran |
| `init` | no | lista de átomos sin variables y de valores iniciales `{"op": "=", "fluent": …, "value": 3.5}` (v3) |
| `metric` | no | `{"direction": "minimize" \| "maximize", "expression": …}` (v3) |

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
  "version": 1,
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
