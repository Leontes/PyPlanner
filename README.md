# PyPlanner

**English** · [Español](README.es.md)

A PDDL planner in Python based on **forward heuristic search**, designed to be embedded in
other projects through JSON. It does not try to beat the state of the art: it aims to be an
up-to-date planner that can be understood, controlled and modified.

## Supported language

PDDL from STRIPS to PDDL+, including PDDL 3:

| PDDL version | What it includes |
|--------------|------------------|
| STRIPS | actions with preconditions, add and delete effects |
| PDDL 1.2 | ADL subset: types, negative and disjunctive preconditions, quantifiers, equality and conditional effects |
| PDDL 2.1 | numeric fluents (with `:action-costs`) and durative actions, with continuous effects |
| PDDL 2.2 | derived predicates and timed initial literals |
| PDDL 3 | trajectory constraints, preferences (also in temporal tasks and with processes) and object fluents |
| PDDL+ | processes and events |

**Strict mode:** every construct must have its requirement declared in `:requirements`.
An input that uses something not supported yet is rejected with a clear error.

## Features

- **Two-phase parser** joined by JSON: PDDL → JSON (with a versioned schema) → internal
  objects. Another system can pass the tasks directly as JSON
  ([format](docs/en/json-format.md)).
- **Grounding** with relaxed reachability analysis and ADL normalization (DNF, expanded
  quantifiers, conditional effects).
- **Numeric fluents and action costs:** states with numeric values, static functions
  substituted, the metric as action cost when possible, and numeric heuristics (*h^hbd*).
- **Temporal planning:** durative actions split into start and end, POPF-style orderings
  over a temporal network, and linear programming (COLIN-style) for continuous effects.
- **PDDL 2.2:** derived predicates (axioms evaluated by strata in each state) and timed
  initial literals, with deadlines and windows in the heuristic to prune the states that
  can no longer make it in time.
- **PDDL 3:** trajectory constraints (the ten modal operators, also timed) watched by
  monitors in the engine; preferences compiled into costs with the metric weights (also
  the maximized ones), with *anytime* search that returns the best plan within the time
  limit, also in temporal tasks and with processes and events; and object fluents
  (PDDL 3.1) compiled into predicates.
- **PDDL+:** processes and events with discretized time in the style of ENHSP (a wait of
  δ, 1 by default), exact integration with Taylor series (Runge-Kutta if the series does
  not terminate), events at their exact instant, and plan validation with a step ten
  times finer (if it fails, the search runs again with δ/2). It supports durative actions
  (with conditional and conditional continuous effects), timed initial literals,
  trajectory constraints and preferences. Temporal tasks with non-linear continuous
  effects also go to this engine.
- **Heuristics:** blind, goalcount, h_max, h_add and h_FF with helpful actions; with timed
  initial literals, h_TFF (a relaxed plan that respects times and windows); with
  processes, also the projection of the invariants of the running actions.
- **Searches:** GBFS, A* and WA*, with eager or lazy evaluation, helpful actions,
  type-based exploration, novelty search (IW and BFWS), landmarks in alternating queues
  (LAMA-style), batch evaluation and *anytime* search (with preferences). Also *enforced
  hill-climbing* (FF) and random walks (Arvand), outside the portfolios because they do
  not solve anything the members do not already solve.
- **Performance:** heuristics and successor generation compiled with Numba, with results
  identical to the Python reference implementation. Batch evaluation on GPU with PyTorch
  is also available.
- **Parallel portfolio:** predefined sets of searches for 1 to 7 cores; the first one to
  find a plan wins (with preferences, the best plan all of them find within the given
  time, sharing the bound).
- **Plans** in IPC format (the one VAL reads) or in JSON.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Leontes/PyPlanner.git
cd PyPlanner
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[fast]"        # recommended: includes Numba
```

| Extra | What it adds | When |
|-------|--------------|------|
| (none) | the planner in Python, with jsonschema and SciPy (linear programming in the temporal engine) | always works |
| `fast` | Numba: compiled heuristics and successors, about 10 times faster | recommended |
| `gpu` | PyTorch: batch heuristic evaluation on CPU or GPU | experiments with large batches |
| `dev` | pytest, ruff, mypy | development |

Old GPUs (Maxwell, such as the GTX 970) only work with the PyTorch builds for CUDA 12.6:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Language

The command-line help, the messages and the errors of the planner are in Spanish. The
identifiers (subcommands, options, configuration names, JSON keys) are in English.

## Usage

### From the command line

```bash
# Find a plan: the plan goes to standard output and the statistics to standard error
pyplanner solve domain.pddl problem.pddl                   # 6-core portfolio (or fewer)
pyplanner solve -j 1 -t 60 domain.pddl problem.pddl         # a single core, 60 s
pyplanner solve -c astar-hmax domain.pddl problem.pddl      # optimal plan
pyplanner solve -f json -o plan.json domain.pddl problem.pddl
pyplanner solve -t 30 domain.pddl preferences.pddl         # with preferences: the best in 30 s
pyplanner solve --delta 0.5 domain.pddl problem.pddl       # PDDL+: time step of 0.5
pyplanner configs                                            # configurations and portfolios

# Phase 1: PDDL → JSON
pyplanner parse domain.pddl -o domain.json

# Phases 1 and 2: check that a domain and a problem are valid (.pddl or .json)
pyplanner check domain.pddl problem.pddl
```

Exit codes of `solve`:

- 0: plan found;
- 2: invalid input;
- 3: proven that there is no plan;
- 4: no plan found (within the limits, or with a search that cannot prove there is none,
  such as IW, a temporal search that prunes repeated states, or one with discretized
  time).

### As a library

```python
from pyplanner.frontend import read_domain, read_problem
from pyplanner.planner import solve
from pyplanner.search import portfolio_for_cores

domain = read_domain("domain.pddl")  # also accepts .json
problem = read_problem("p01.pddl", domain)
result = solve(domain, problem, portfolio_for_cores(2), time_limit=60)
if result.plan is not None:
    print(result.plan.to_ipc())  # or result.plan.to_dict() for the JSON
```

To embed it in another system:

- **JSON input:** `pyplanner.loader.load_domain` and `load_problem` take the JSON directly
  and validate it (structure, requirements and semantics).
- **JSON output:** the plan is a document with `"kind": "plan"`.
- **Errors:** every input error inherits from `pyplanner.errors.PDDLError`.
- **Processes:** the portfolio creates processes, so the code that calls `solve` must be
  inside `if __name__ == "__main__":`. With the `spawn` start method (if PyTorch is
  loaded, if the process has several threads, or for the GPU members), each process
  imports the main script again; without that guard, the planner detects it and returns
  `ERROR` with the cause.
- **Embedded Python** (in a C++ program, for instance): `spawn` launches
  `sys.executable`, which is then the host program. Before calling `solve`, set the
  interpreter with `multiprocessing.set_executable(...)`.
- **Memory:** each portfolio member has a hard limit (`RLIMIT_DATA`) slightly above its
  share of `memory_limit`; past it, its search ends with `limit_reached`. The process
  that calls `solve` is not limited: grounding and a single-configuration search only have
  the soft limit. The CLI does limit itself.
- **Time limit:** without `time_limit`, each search runs until it finds a plan, proves
  there is none or runs out of memory; on a hard problem that can take a long time. From
  another program, always pass it.

## Layout

| Path | Contents |
|------|----------|
| `src/pyplanner/pddl/` | phase 1: lexer, PDDL → JSON parser and the reverse writer |
| `src/pyplanner/schema/` | JSON schema (the contract between phases and with other systems) |
| `src/pyplanner/loader/` | phase 2: JSON → AST, with validation |
| `src/pyplanner/grounding/` | grounding, ADL normalization and the ground task |
| `src/pyplanner/heuristics/` | heuristics: Python reference, Numba and PyTorch |
| `src/pyplanner/search/` | search, configurations, portfolios and successor generation |
| `src/pyplanner/portfolio/` | parallel execution of configurations |
| `src/pyplanner/plan/` | plans: IPC and JSON formats, internal validation |
| `src/pyplanner/planner.py` | facade: from domain and problem to plan |
| `src/pyplanner/cli.py` | command-line interface |
| `tests/` | tests (`pytest`); `dataset.toml`: the test problems |
| `vendor/PDDLDomains/` | submodule: the PDDL domain catalog of the test problems |
| `scripts/` | VAL build and performance measurements (`benchmark.py`) |
| `docs/en/`, `docs/es/` | documentation, in English and in Spanish |

`tools/` (VAL built locally), `salidas/` (local outputs) and `privado/` (private
documents) are not in git.

## Documentation

- [JSON format](docs/en/json-format.md): the JSON input and output contract.
- [Search engine](docs/en/search-engine.md): how the engine works and its configurations.
- [Test problems](docs/en/tests.md): the problems of the tests, from the PDDLDomains
  catalog and grouped by PDDL version.

## Development

```bash
git clone --recursive https://github.com/Leontes/PyPlanner.git   # with the catalog
git submodule update --init vendor/PDDLDomains                    # or, in an existing clone
pip install -e ".[dev,fast]"
scripts/build_val.sh     # builds VAL into tools/bin/Validate (needs git, cmake and g++)
pytest                   # tests
ruff check . && ruff format --check . && mypy
```

- **Test problems:** the tests use problems from the
  [PDDLDomains](https://github.com/Leontes/PDDLDomains) catalog, in the
  `vendor/PDDLDomains` submodule pinned to a commit, with at least two collections per
  PDDL version, from STRIPS to PDDL+ ([test problems](docs/en/tests.md)). The tests solve
  all of them with every portfolio and validate the plans with
  [VAL](https://github.com/KCL-Planning/VAL). If VAL is not built, those tests are
  skipped.
- **Measuring performance:** `scripts/benchmark.py` measures configurations on a catalog
  collection or dataset, with time and memory limits per search, and validates the plans
  with VAL (`python scripts/benchmark.py --help`).
- **Equivalence:** the Numba and PyTorch implementations have tests that check they give
  exactly the same results as the Python reference.

## License

The code and the documentation are distributed under the [MIT license](LICENSE) (© 2026
José Segura). The domains and problems of the PDDLDomains catalog (submodule) have their
own terms: see its `LICENSES.md`.
