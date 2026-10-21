# Test problems

**English** · [Español](../es/pruebas.md)

The tests solve a set of problems that covers the whole language the planner accepts, from
STRIPS to PDDL+, and validate every plan with [VAL](https://github.com/KCL-Planning/VAL).
The problems come from the [PDDLDomains](https://github.com/Leontes/PDDLDomains) catalog,
which is in the `vendor/PDDLDomains` submodule, pinned to a commit.

## The dataset

[`tests/dataset.toml`](../../tests/dataset.toml) is a catalog dataset: problems `p01` and
`p02` of fifteen collections, at least two per **PDDL version level**. As in the catalog's
`by-level` dataset, each collection is a group `level<N>/<collection>`. The catalog
computes the level from the requirements each problem uses, on a cumulative scale from
STRIPS (1) to PDDL+ (6). A planner that reaches level *N* must solve every problem of
levels 1 to *N* (see "PDDL version (levels)" in the catalog's `docs/en/catalog.md`).

| Level | Version | Collection | Declared requirements |
|------:|---------|------------|-----------------------|
| 1 | STRIPS | `ipc-2000/blocks-strips-untyped` | `:strips` |
| 1 | STRIPS | `ipc-1998/gripper-round-1-strips` | none (implicit STRIPS) |
| 2 | PDDL 1.2 (ADL) | `ipc-2023-classical/rubiks-cube` | `:adl` |
| 2 | PDDL 1.2 (ADL) | `ipc-2023-classical/slitherlink` | `:typing :adl` |
| 3 | PDDL 2.1, numeric | `ipc-2023-numeric/counters` | `:strips :typing :equality :adl :numeric-fluents` (*) |
| 3 | PDDL 2.1, numeric | `ipc-2023-numeric/zenotravel` | `:typing :fluents` (*) |
| 3 | PDDL 2.1, temporal | `ipc-2002/driverlog-time-simple-automatic` | `:typing :durative-actions :numeric-fluents` (*) |
| 3 | PDDL 2.1, temporal | `ipc-2002/zenotravel-time-simple-automatic` | `:durative-actions :typing :numeric-fluents` (*) |
| 3 | PDDL 2.1, temporal | `propios/greenhouse` (†) | `:typing :durative-actions :numeric-fluents :continuous-effects :duration-inequalities :conditional-effects :negative-preconditions` |
| 4 | PDDL 2.2 | `ipc-2004/psr-middle-derived-predicates-adl` | `:adl :derived-predicates` |
| 4 | PDDL 2.2 | `ipc-2004/satellite-time-time-windows-strips` (‡) | `:strips :equality :typing :fluents :durative-actions :timed-initial-literals` |
| 5 | PDDL 3 | `ipc-2006/trucks-time-constraints` | `:typing :adl :durative-actions :fluents :constraints` |
| 5 | PDDL 3 | `ipc-2006/storage-preferences-qualitative` (‡) | `:typing :adl :preferences :constraints` |
| 6 | PDDL+ | `smtplan/car-nodrag` (‡) | `:typing :durative-actions :fluents :time :negative-preconditions :timed-initial-literals` |
| 6 | PDDL+ | `smtplan/generator-events` (*) (‡) | `:fluents :durative-actions :duration-inequalities :adl :typing :time :continuous-effects` |

[`tests/catalog.py`](../../tests/catalog.py) reads the dataset and hands the problems to the
tests. The pytest identifiers are `<group>/<problem>`, for example
`level3/ipc-2002/driverlog-time-simple-automatic/p01`. They are used by:

- `test_dataset_parser.py`: the two phases of the parser (PDDL → JSON → AST);
- `test_dataset_solve.py`: the portfolio of each number of cores (1 to 7) solves every
  problem, and VAL validates the plan and its metric;
- `test_heuristics.py`: the Numba and PyTorch heuristics give the same values as the
  Python reference;
- `test_dataset.py`: the integrity of the dataset. Every level has at least two
  collections, every listed collection provides its problems (a collection that goes
  missing or is renamed in the catalog does not silently disappear) and every problem
  belongs to the level of its group.

To add problems, extend `problems` or `collections` in `tests/dataset.toml`.

## The catalog in the submodule

- **Initializing it:** `git clone --recursive`, or in an existing clone,
  `git submodule update --init vendor/PDDLDomains`. Without it, the tests that use the
  catalog fail with an error that says so.
- **Always the submodule's:** `tests/catalog.py` imports the `pddldomains` package from the
  submodule itself (not an installed one) and uses its `dominios/` directory (not
  `$PDDL_DOMINIOS`). If another one were imported, it raises an error.
- **Without writing to it:** the tests analyze the collections in memory, without the
  catalog's cache, so they write nothing to the submodule. The datasets measured by
  `scripts/benchmark.py` do use the cache, in the submodule's `.cache/`, which the catalog
  ignores in git.
- **Updating it** is an explicit change in PyPlanner: `git -C vendor/PDDLDomains fetch`,
  `git -C vendor/PDDLDomains checkout <commit>`, run the tests and commit the submodule. A
  change in the catalog never reaches PyPlanner without that step.

## Conventions

They are the catalog's (see its `docs/en/catalog.md`):

- Files are copied **unmodified**. There are two exceptions, fixed in the file itself:
  errata in `:requirements`, marked here with (*), and mismatches of other kinds when it is
  decided to fix them, marked with (‡). Each collection explains its own in its card,
  `info.toml`.
- **Every requirements mismatch is treated as an erratum of the file:** the missing
  requirement is added by hand, and strict mode makes no exceptions.
- **Strict mode:** the planner requires every construct used to have its requirement
  declared. If `:requirements` is not declared, `:strips` is assumed.
- **Unique names:** the parser does not accept a name declared twice anywhere (types,
  domain, actions…). Several (‡) errata in the catalog are renamings for that reason.

## Notes

- The IPC 2023 has no temporal, PDDL 3 or PDDL+ track. That is why levels 1, 3 (temporal),
  4, 5 and 6 come from the historical IPC archive and from the SMTPlan benchmarks. The
  PDDL 3 ones are from IPC 2006: storage combines preferences with untimed constraints
  (`sometime`, `at-most-once`) and a metric with `is-violated`; trucks, timed constraints
  (`within`) over durative actions.
- The ENHSP PDDL+ benchmarks were discarded because they use `:constraint`, an extension
  of that planner that is not part of the standard.
- (†) `greenhouse` **is not an official benchmark**. It was created for PyPlanner to test
  edge cases of temporal PDDL 2.1 expressiveness that do not appear in any public
  benchmark found (neither in the IPCs nor in those of COLIN/OPTIC or VAL): continuous
  effects inside `forall` and `when`, conditional effects with `over all` conditions,
  non-fixed `?duration` in `at start` effects, a duration whose maximum depends on a
  variable that changes continuously, and a conditional effect on a continuous variable.
  Its plans are validated with VAL like the rest. `tests/test_temporal_engine.py` also
  uses it with other problems and piece by piece.
- Several domains use features of lower levels: for instance, `car` uses
  `:timed-initial-literals` and trucks needs the temporal engine. That is expected,
  because the scale is cumulative.
