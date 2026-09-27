# Formato JSON de entrada (versión 1)

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
- `version` es la versión del formato (ahora mismo, 1). Las versiones posteriores solo
  **añaden** construcciones, así que un documento v1 seguirá siendo válido.
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
| `constants` | no | `[{"name": "home"}, ...]` |
| `predicates` | no | `[{"name": "at", "parameters": [{"name": "?x"}, ...]}, ...]` |
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
| `init` | no | lista de átomos sin variables |

## Fórmulas y efectos (v1: STRIPS)

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
| `SemanticError` | 2 | predicado sin declarar, aridad incorrecta, variable libre, objeto desconocido, nombre repetido, dominio que no coincide… |
