# JSON input format (version 7)

**English** · [Español](../es/formato-json.md)

The planner takes domains and problems as JSON. This document is the contract for any
component that wants to generate that JSON without going through PDDL. The formal schema
is in [`src/pyplanner/schema/pddl.schema.json`](../../src/pyplanner/schema/pddl.schema.json)
(JSON Schema 2020-12). If this document and the schema disagree, the schema wins.

## Flow

```
PDDL file ──(phase 1: pyplanner.pddl)──► JSON ──(phase 2: pyplanner.loader)──► AST
                                          ▲
                   another component ─────┘
```

- **Phase 1** (`parse_pddl`, `parse_file`, `pyplanner parse`): syntax only. It keeps the
  structure of the PDDL.
- **Phase 2** (`load_domain`, `load_problem`): validates the document against the schema,
  checks the requirements and performs the semantic validation. It does not trust the JSON
  it receives.
- **Reverse writer** (`write_pddl`): JSON → PDDL, for instance to validate with VAL.

## Conventions

- Each document is a domain (`"kind": "domain"`), a problem (`"kind": "problem"`) or a plan
  (`"kind": "plan"`, the planner's output).
- `version` is the version of the format:
  - v1: STRIPS;
  - v2: PDDL 1.2, ADL subset;
  - v3: numeric PDDL 2.1 and action costs (`:action-costs`);
  - v4: temporal PDDL 2.1 (durative actions);
  - v5: PDDL 2.2 (derived predicates and timed initial literals);
  - v6: PDDL 3 (trajectory constraints, preferences, `is-violated` and object fluents);
  - v7 (current): PDDL+ (processes, events and negated literals in the initial state).

  Each version only **adds** constructs, so a v1 document is still valid.
- Names follow the pattern `[A-Za-z][A-Za-z0-9_-]*`. Phase 2 lowercases them, because PDDL
  is case-insensitive.
- **Each name is declared only once, anywhere:** the domain, the problem, the types
  (including `object`), the constants, the objects, the predicates, the functions, the
  actions, the processes and the events cannot share a name. The error says what the
  repeated name was (`el nombre 'satellite' ya se ha definido previamente como nombre del
  dominio`; the messages are in Spanish). The only exception are preferences, which may
  share a name among themselves (but not with anything else). Variables (`?x`) have their
  own scope.
- The arguments of an atom are strings: a **variable** if it starts with `?` and an
  **object or constant** otherwise.
- Formula and effect nodes have an `op` field that gives their kind.
- Any node may have `loc` (`{"line", "column"}`) with its position in the original PDDL,
  and the document may have `source` with the file name. Both are optional and only serve
  the error messages.
- Fields that are not in the schema are not accepted (`additionalProperties: false`).

## Domain

| Field | Required | Contents |
|-------|----------|----------|
| `version`, `kind`, `name` | yes | |
| `requirements` | no | list of requirements (`":strips"`…). Empty means `:strips` |
| `types` | no | `[{"name": "truck", "parent": "vehicle"}, {"name": "place"}, ...]`; without `parent`, the supertype is `object` |
| `constants` | no | `[{"name": "home", "type": "place"}, ...]` |
| `predicates` | no | `[{"name": "at", "parameters": [{"name": "?x", "type": "truck"}, ...]}, ...]` |
| `functions` | no | numeric functions, with the same shape as `predicates` (v3) |
| `actions` | no | see below |
| `durative_actions` | no | durative actions (v4); see [Durative actions](#durative-actions-v4) |
| `derived` | no | derived predicate rules (v5); see [PDDL 2.2](#pddl-22-v5) |
| `constraints` | no | hard trajectory constraints (v6); see [PDDL 3](#pddl-3-v6) |
| `processes`, `events` | no | processes and events (v7); see [PDDL+](#pddl-v7) |

An action has `name`, `parameters` and, optionally, `precondition` (a formula) and `effect`
(an effect). If either is missing, it is taken as empty. Actions and durative actions share
a namespace.

## Problem

| Field | Required | Contents |
|-------|----------|----------|
| `version`, `kind`, `name` | yes | |
| `domain` | yes | name of the domain; it must match the loaded domain |
| `goal` | yes | formula |
| `requirements` | no | added to the domain's |
| `objects` | no | like `constants`. They cannot repeat any name of the domain (not even a constant) |
| `init` | no | list of ground atoms, of initial values `{"op": "=", "fluent": …, "value": 3.5}` (v3), of timed initial literals `{"op": "at", "time": 10, "literal": …}` (v5) and of negated literals `{"op": "not", "arg": <atom>}` (v7, redundant) |
| `metric` | no | `{"direction": "minimize" \| "maximize", "expression": …}` (v3) |
| `constraints` | no | trajectory constraints, which may be preferences (v6); see [PDDL 3](#pddl-3-v6) |

## Types (v2)

- `type` is optional in constants, objects and parameters (of actions, predicates and
  quantifiers). If missing, the type is `object`.
- A list of types means `(either ...)`: `{"name": "?x", "type": ["truck", "plane"]}`.
- A supertype that is not declared as a type is declared implicitly, with supertype
  `object`.
- Phase 2 checks that the constants and objects that appear in atoms have a type
  compatible with the predicate's.

## Strict mode

Every construct must have its requirement declared, in the domain or in the problem. The
rules are in `pyplanner/requirements.py`:

| Construct | Requirement |
|-----------|-------------|
| `types`, or `type` in any declaration | `:typing` |
| `not` on an atom or an equality, in a formula | `:negative-preconditions` |
| `not` on a compound formula, `or`, `imply` | `:disjunctive-preconditions` |
| `exists` | `:existential-preconditions` |
| `forall` in a formula | `:universal-preconditions` |
| `=` | `:equality` |
| `forall` or `when` in an effect | `:conditional-effects` |
| `functions`, initial values, `compare`, numeric effects, `metric` | `:numeric-fluents` |
| action-cost profile: `functions` and their initial values, `increase` of `(total-cost)` by a number or a function, `minimize (total-cost)` | `:numeric-fluents` or `:action-costs` |
| `durative_actions` | `:durative-actions` |
| duration with `<=` or `>=`, several constraints or `time` | `:duration-inequalities` |
| continuous effect | `:continuous-effects` |
| `derived` | `:derived-predicates` |
| timed initial literal (`at` in `init`) | `:timed-initial-literals` |
| `constraints` and their modal operators | `:constraints` |
| `preference`, `is-violated` | `:preferences` |
| functions with `type`, function terms as arguments, `assign` of an object or `undefined`, object initial values | `:object-fluents` |
| metric of numbers, arithmetic and `is-violated` | `:numeric-fluents` or `:preferences` |
| `processes`, `events` | `:time` |

- `not` in an effect is a STRIPS delete and needs nothing, and neither does a negated
  literal in `init`.
- The continuous effects of a process need `:time` and `:numeric-fluents`, not
  `:continuous-effects` (which is the one of durative actions).
- `:adl` and `:quantified-preconditions` stand for their component requirements, and
  `:fluents` stands for `:numeric-fluents`.

## Formulas and effects

| `op` | Fields | Where it can appear | Since |
|------|--------|---------------------|-------|
| `atom` | `predicate`, `args` | formulas, effects, `init` | v1 |
| `and` | `args` (list of nodes) | formulas, effects | v1 |
| `not` | `arg` (an atom) | effects (delete) | v1 |
| `not` | `arg` (a formula) | formulas | v2 |
| `or` | `args` | formulas | v2 |
| `imply` | `args` (exactly 2) | formulas | v2 |
| `exists`, `forall` | `parameters` (at least 1), `arg` (a formula) | formulas | v2 |
| `=` | `args` (exactly 2 terms) | formulas | v2 |
| `forall` | `parameters`, `arg` (an effect) | effects | v2 |
| `when` | `condition` (formula), `arg` (only `atom`, `not`, `and` and numeric effects) | effects | v2 |
| `compare` | `comparator` (`<`, `<=`, `=`, `>=`, `>`), `left`, `right` (expressions) | formulas | v3 |
| `increase`, `decrease`, `assign`, `scale-up`, `scale-down` | `fluent` (function term), `value` (expression) | effects | v3 |

## Numeric expressions (v3)

| `op` | Fields | Notes |
|------|--------|-------|
| `num` | `value` | number |
| `fluent` | `function`, `args` | function term, e.g. `{"op": "fluent", "function": "fuel", "args": ["?a"]}` |
| `+`, `*` | `args` (two or more) | |
| `-` | `args` (one or two) | with one, negation |
| `/` | `args` (two) | |
| `total-time` | — | plan duration; **only in the metric**. In a sequential plan, each action lasts one unit |
| `duration` | — | `?duration`, the duration of the action (v4); **only inside a durative action**, and not in its own duration |
| `#t` | — | elapsed time (v4); **only in a continuous effect** |

- `=` between two objects or variables is the equality of `:equality`
  (`{"op": "=", "args": […]}`); between expressions it is a comparison
  (`{"op": "compare", "comparator": "=", …}`).
- Numeric effects are evaluated with the values of the state **before** the action.
- A function without an initial value is undefined.

Other restrictions:

- A quantified variable cannot take the name of another variable already bound (a
  parameter or an outer quantifier). Two sibling quantifiers may use the same name.
- The condition of `when` is evaluated on the state **before** the action.

### Version 1 table (unchanged)

| `op` | Fields | Where it can appear |
|------|--------|---------------------|
| `atom` | `predicate`, `args` | formulas, effects, `init` |
| `and` | `args` (list of nodes) | formulas, effects |
| `not` | `arg` (an atom) | effects only (delete) |

## Durative actions (v4)

A durative action has `name`, `parameters`, `duration` and, optionally, `condition` and
`effect` (empty if missing):

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

- **`duration`** is the conjunction of the duration constraints, each
  `{"comparator": "=" | "<=" | ">=", "value": <expression>}`. With
  `"time": "start" | "end"`, the expression is evaluated when the action starts or ends
  (`(at end (<= ?duration …))`); without `time`, when it starts. Empty if the action has no
  constraints (`:duration ()`).
- **Conditions** (`condition`):

  | `op` | Fields | Meaning |
  |------|--------|---------|
  | `at-start`, `at-end` | `arg` (formula) | true at the start / at the end |
  | `over-all` | `arg` (formula) | true throughout the action (excluding the endpoints) |
  | `and` | `args` (conditions) | |
  | `forall` | `parameters`, `arg` (a condition) | |

- **Effects** (`effect`):

  | `op` | Fields | Meaning |
  |------|--------|---------|
  | `at-start`, `at-end` | `arg` (`atom`, `not`, `and` or numeric effect) | at the start / at the end |
  | `increase`, `decrease` | `fluent`, `value` with `#t` | **continuous effect**: `value` is `#t` or a product with `#t` as a factor (`(* #t 2)`); the function changes at that rate while the action runs |
  | `and` | `args` (effects) | |
  | `forall` | `parameters`, `arg` (an effect) | |
  | `when` | `condition` (a condition with time specifiers), `arg` (timed or continuous effects) | an `at start` effect can only depend on `at start` conditions |

- `?duration` (`{"op": "duration"}`) can appear in the values of the action's numeric
  effects, but not in its conditions or in its own duration. That is how the PDDL 2.1
  grammar defines it, and VAL does not validate plans of domains that use it in
  conditions.
- In PDDL, `(at start X)` is told apart from an atom of the predicate `at` because `X` is a
  list.
- The reverse writer writes a missing condition or effect as `(and)`, because VAL does not
  accept `:condition ()`.

## PDDL 2.2 (v5)

**Derived predicates.** Each rule `(:derived (p ?x - t …) <formula>)` is an element of
`derived`:

```json
{"predicate": "fed", "parameters": [{"name": "?l", "type": "line"}],
 "condition": {"op": "exists", "parameters": [{"name": "?x", "type": "device"}],
               "arg": {"op": "atom", "predicate": "closed", "args": ["?x"]}}}
```

- The predicate is also declared in `predicates`. The rule has the same arity, and each
  parameter a type compatible with the declared one.
- Several rules for the same predicate add up: the atom is true if any of them makes it
  true.
- The condition is any formula, with the head variables bound, and it may use other
  derived predicates (also its own: recursive rules).
- A derived predicate can appear in preconditions, conditions and goals, but **not in
  effects or in `init`**: its value is always deduced from the other facts.
- The rules must be **stratifiable**: no derived predicate may depend negatively on
  itself, directly or indirectly. A dependency is negative if the atom is under an odd
  number of `not` (the antecedent of `imply` counts as one).

**Timed initial literals** (TIL). `(at 10 (visible a s))` in `:init` is
`{"op": "at", "time": 10, "literal": <atom>}`, and `(at 20 (not (visible a s)))` has
`{"op": "not", "arg": <atom>}` as `literal`.

- At instant `time` (≥ 0), the fact becomes true (or false, with `not`) without any action
  causing it.
- Only in temporal domains (`:durative-actions`): a sequential plan has no time.
- They cannot affect derived predicates, nor give a fact both values at the same instant.
  Repeating the same literal has no effect.
- In PDDL, `(at <number> …)` is always a TIL: an object cannot be named like a number.

## PDDL 3 (v6)

**Preferences.** `{"op": "preference", "name": "p1", "arg": …}`; `name` is optional (an
unnamed preference cannot be referred to in the metric). Several preferences may share a
name, and one inside a `forall` shares it with all its instances. They can only appear:

- in the outer conjunction (`and`, `forall`) of the goal and of the preconditions, with a
  formula as `arg`;
- in the conditions of a durative action, wrapping a condition with a time specifier
  (`at-start`, `at-end` or `over-all`);
- in the problem constraints, wrapping a constraint.

Never inside `or`, `not`, `imply`, `exists` or another preference.

**Trajectory constraints.** They hold over the sequence of states of the plan (the initial
one and the one each step leaves). Modal operators are not nested: they go directly in
`constraints` or inside `and`, `forall` and (only in the problem) `preference`.

| `op` | Fields | Meaning |
|------|--------|---------|
| `at-end` | `arg` | φ in the final state |
| `always` | `arg` | φ in every state |
| `sometime` | `arg` | φ in some state |
| `at-most-once` | `arg` | φ is true in at most one run of consecutive states |
| `sometime-after` | `args` [φ, ψ] | if φ in a state, ψ in that one or a later one |
| `sometime-before` | `args` [φ, ψ] | if φ in a state, ψ in a strictly earlier one |
| `within` | `time`, `arg` | φ in some state at instant ≤ `time` |
| `hold-after` | `time`, `arg` | φ in every state at instant > `time` |
| `hold-during` | `from`, `to`, `arg` | φ in every state at an instant in [`from`, `to`) |
| `always-within` | `time`, `args` [φ, ψ] | whenever φ, ψ within `time` |

In a domain without durative actions, the instant of a state is the number of plan steps
up to it. Instants are ≥ 0, and in `hold-during`, `from` ≤ `to`. Domain constraints are
always hard: they do not admit preferences.

**Object fluents** (`:object-fluents`, PDDL 3.1). A function with `type` (a type or a list
of types) takes objects as values: `{"name": "loc", "parameters": […], "type": "place"}`.
Without `type`, it is numeric.

- An argument of an atom or of an equality can be a function term:
  `{"op": "atom", "predicate": "at", "args": [{"op": "fluent", "function": "loc", "args":
  ["?r"]}]}`.
- `{"op": "assign", "fluent": …, "value": …}` takes as value an object, a variable, another
  function term or `{"op": "undefined"}`.
- Initial value: `{"op": "=", "fluent": …, "value": "a"}`.
- `=` between a term and an object or a variable is an equality (`"op": "="`); between two
  functions, phase 1 does not know whether they are numeric and writes it as a comparison
  (`compare`), which phase 2 interprets according to the declarations.

Phase 2 **compiles them into predicates** before building the AST (see
`pyplanner/loader/object_fluents.py`): `(loc ?r) - place` becomes the predicate
`(loc ?r ?value - place)`, and each term is replaced by a variable bound with `exists` (or
with `forall` and `when` in effects). They are not allowed in the metric, in durations or
in continuous effects. Since VAL does not support object fluents, plans are validated
against the compiled document, written with the reverse writer.

**`is-violated`.** `{"op": "is-violated", "name": "p1"}` is the number of violations of the
preference `p1` (of all its instances), and it can only appear in the metric. The name
must be that of some preference of the domain or of the problem.

## PDDL+ (v7)

**Processes.** `(:process p :parameters (…) :precondition <formula> :effect <effect>)` is
an element of `processes`. While the precondition holds, its continuous effects change the
functions as time passes:

```json
{"name": "moving", "parameters": [],
 "precondition": {"op": "atom", "predicate": "running", "args": []},
 "effect": {"op": "increase", "fluent": {"op": "fluent", "function": "d", "args": []},
            "value": {"op": "*", "args": [{"op": "#t"},
                                          {"op": "fluent", "function": "v", "args": []}]}}}
```

- The effect only has continuous effects (`increase` or `decrease` with `#t`, with the same
  shape as those of durative actions: `#t` or `(* #t <expression>)`), and `and` and
  `forall` of them.
- The precondition is a formula without preferences.

**Events.** `(:event e :parameters (…) :precondition <formula> :effect <effect>)` is an
element of `events`, with the shape of an action: it happens as soon as its precondition
holds, with instantaneous effects (without `#t`).

**Negated literals in the initial state.** `(not (p))` in `:init` (valid in PDDL 3.1) is
`{"op": "not", "arg": <atom>}`. It is redundant, because what is not declared is false, but
it cannot negate a fact that the same `init` declares true.

## Example

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

## Plan (output)

The planner returns plans with `"kind": "plan"` (`pyplanner solve -f json`):

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

- `cost`: sum of the action costs. Without a metric that can be used as cost, it is the
  number of actions.
- `metric` (optional, v3): value of the problem metric, when the action costs represent it
  (see [search-engine.md](search-engine.md#numeric-engine)). It is the initial value of
  the metric plus `cost`, and it matches the one VAL computes. With preferences (v6), it
  also adds the weights of the violated ones, and `cost` is only the actions'
  contribution.
- `violated` (optional, v6): instances of goal and constraint preferences that the plan
  violates, as `"name arg1 arg2"` (see
  [search-engine.md](search-engine.md#preferences-and-anytime-search)).

## Errors

Every error inherits from `pyplanner.errors.PDDLError` and carries, when known, the
position in the PDDL (`location`) and the path in the JSON (`path`, e.g.
`actions[0].effect.args[1]`).

| Error | Phase | Cause |
|-------|-------|-------|
| `PDDLSyntaxError` | 1 | malformed PDDL |
| `SchemaValidationError` | 2 | the JSON does not match the schema, or is not JSON |
| `UnsupportedFeatureError` | 1 and 2 | requirement or construct that this version does not support |
| `UndeclaredRequirementError` | 1 (domains) and 2 | construct used without declaring its requirement (strict mode) |
| `SemanticError` | 2 | undeclared predicate, wrong arity, free variable, unknown object, repeated name, domain that does not match, undeclared or incompatible type, type cycle… |
