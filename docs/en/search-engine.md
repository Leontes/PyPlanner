# Search engine

**English** · [Español](../es/motor-busqueda.md)

How the engine is built and how to use it.

## Flow

```
Domain + Problem (AST) ──ground()──► GroundTask ──search() / run_portfolio()──► SearchResult ──► Plan
```

| Step | Module | What it does |
|------|--------|--------------|
| Grounding | `pyplanner.grounding` | Instantiates the actions reachable in the relaxation and removes the static facts |
| Heuristics | `pyplanner.heuristics` | Evaluate batches of states; reference implementation and PyTorch |
| Search | `pyplanner.search` | GBFS / A* / WA*, eager or lazy, with helpful actions and batches |
| Portfolio | `pyplanner.portfolio` | Several configurations in parallel processes; the first one wins (with preferences, the best within the time limit) |
| Plan | `pyplanner.plan` | IPC or JSON output and internal validation (simulation) |
| Facade | `pyplanner.planner.solve` | Chains all of the above |

## ADL normalization

The grounder (`grounding/grounder.py`) normalizes each action instance with
`grounding/normalize.py`, so that the search only sees conjunctive structures:

- **Preconditions and goals → DNF of literals.** Quantifiers are expanded over the objects
  of their type (`exists` → disjunction, `forall` → conjunction), negations are pushed down
  to the atoms and `imply` is treated as `(or (not a) b)`. `=` and static facts are
  evaluated right away; contradictory and subsumed clauses are removed. If the DNF exceeds
  10 000 clauses, it aborts with a clear error (such formulas have to be written with
  derived predicates).
- **Disjunctive precondition:** one `GroundAction` per clause, with the same name and
  arguments, so they look the same in the plan.
- **Effects → unconditional effects + conditional effects** (`forall` is expanded; each
  clause of the condition of a `when` gives a conditional effect). If the precondition
  already guarantees the condition, the effect becomes unconditional; if it contradicts
  it, the effect is dropped.
- **Goal:** disjunction of clauses. If none is reachable, the goal is left empty and the
  task is recognized as unsolvable without searching.
- **Relaxed reachability:** negative literals are assumed reachable, and a conditional
  effect counts if some clause of its condition is reachable.
- **Candidates:** only the assignments that make the necessary atoms of the precondition
  reachable are normalized: those of the outer conjunction and, from each disjunction of
  atoms, those of one of its disjuncts. Equalities in the outer conjunction join equal
  parameters or fix them to a constant. The fixpoint is **semi-naive**: in each round, a
  schema only looks for the candidates that use some new fact (see
  `grounding/grounder.py`).

## Ground task and states

- A **state** is a `State(facts, values)`: `facts` is an `int` used as a bitset over
  `GroundTask.facts`, and `values` the tuple of values of the numeric variables
  (`GroundTask.variables`; empty in propositional tasks, `None` if the value is undefined).
  It is a `NamedTuple`: it is compared and used as a dictionary key like a tuple, at a
  measured cost of about 0.3 µs per successor. An action is applicable if
  `facts & pre == pre`, `facts & neg == 0` (negative preconditions) and its numeric
  conditions hold.
- **Conditional effects:** conditions are evaluated on the state **before** the action;
  then facts are deleted and added (a fact deleted and added at once stays true). The
  conditional effects of each action are indexed by their first condition fact, so
  applying it only checks those that can fire. In rubiks-cube each turn has hundreds of
  conditional effects and only a few fire.
- **Static** facts are not in the state.
- **Successor generation** (`search/successors.py`), with a **canonical order**:
  1. first, the actions without positive preconditions;
  2. then, for each true fact in increasing order, the actions whose first precondition is
     that fact;
  3. within each group, by index.

  No sorting is needed, and there are two implementations with the same order: the
  reference one (`GroundTask.applicable`) and one compiled with Numba, which checks the
  preconditions with 64-bit words. The search uses the compiled one except with
  `--device python`; since the order is the same, it expands the same nodes with both.
- To generate successors, actions are grouped by their first precondition: only the
  actions whose first precondition is true in the state are looked at.
- Facts and actions have a deterministic order, so two identical runs give the same plan.

## Numeric engine

- **Grounding.**
  - Functions that no action modifies are **static**: they are replaced by their value
    when grounding. A comparison with an undefined static value is false.
  - Only the **relevant** functions are state **variables**: those read in some condition
    or in the goal, or in the value of an effect on another relevant variable (fixpoint).
    Those that only appear in the metric, such as `total-cost` or `total-fuel-used`, are
    not in the state: if they were, two equal states with different accumulated cost would
    be different and duplicate detection would stop working.
  - Numeric conditions are normalized to `expression ⋈ 0` with `⋈` ∈ {`>=`, `>`, `=`};
    `(not (= a b))` gives two clauses with `>`. Linear expressions are also stored in
    linear form (sorted terms and a constant), which is the one the compiled kernels use.
- **Effect semantics** (`GroundAction.apply`):
  - numeric effects are evaluated with the values of the previous state;
  - several `increase`/`decrease` on the same variable add up;
  - any other conflict, a division by zero or an undefined value make the action
    inapplicable (`apply` returns `None` and the search drops the successor);
  - an undefined value makes any comparison false.
- **The metric as cost.** If the metric minimizes a linear expression of `total-time` and of
  functions that actions only change with constant `increase`/`decrease` (after
  substituting the static ones), each action contributes a fixed amount: that is its
  **cost**, and minimizing the plan cost minimizes the metric. That is the case of
  `:action-costs` and of metrics like zenotravel's (`total-time` counts as the plan
  length). Otherwise (maximized, non-linear metrics or negative costs), the search uses
  unit cost and the grounder leaves a **note**, which the CLI shows.
- **Output.** The plan carries the cost and, when the costs represent the metric, its value
  (`; metric = …` in IPC, `"metric"` in JSON), which matches VAL's `Final value`. Without
  a usable metric, the IPC comment says `(unit cost)`.
- **Numeric heuristic: *h^hbd*** (Scala, Haslum, Thiébaux and Ramírez, 2020). Each distinct
  numeric condition is one more node of the relaxation. A relaxed action whose effect
  increases a linear condition by a constant amount Δ achieves it by repeating
  ⌈deficit/Δ⌉ times (for `=`, moving towards 0), and the edge costs repetitions × cost. The
  repetitions depend on the state. Effects that are not constant increments and non-linear
  conditions are **optimistic** (one repetition is enough), so the heuristic never reports
  a false dead end. Effects that worsen a condition are ignored, like deletes. Assigning a
  constant to the only variable of a condition achieves it only if that value satisfies
  it (otherwise, it only serves to define the variable when it has no value). A variable
  increment (`increase x y`) only counts if it is positive, and its repetitions are counted
  with the state's value (see "Rates set by actions"). h_FF counts each operator by its
  cost times the maximum number of repetitions it needs.

## Temporal engine

The temporal engine follows POPF (Coles, Coles, Fox and Long, 2010): forward search over
start and end actions, with a simple temporal network (STN) that decides when each step
happens. The code is in `grounding/temporal.py`.

- **Compilation.** Each durative action `a` gives two classical actions and a fact
  `running@a`:
  - `a@start`: `at start` conditions, the `over all` literals that its start effects
    cannot touch (the **start guard**, see below) and `¬running@a` (an instance does not
    overlap with itself); `at start` effects and `running@a`;
  - `a@end`: `at end` and `over all` conditions, and `running@a`; `at end` effects and
    `¬running@a`.

  The goal also requires that no action is still running. Everything else is reused as
  is: grounder, ADL and numeric normalization, compiled successors and *h^hbd* (Python,
  Numba and PyTorch) on the task of starts and ends. The domain's instantaneous actions
  coexist with the durative ones.
- **Start guard.** An `over all` literal that the start effects cannot touch (another
  predicate, or arguments that by their types cannot be the same objects; and not derived)
  has the same value right after starting as before: if it does not hold, the start
  breaks its own invariant at once and is dropped. Requiring it already in `a@start` does
  not change the semantics, but grounding stops instantiating starts that can never end.
  In satellite, `take_image` has all its conditions in `over all`: without the guard, its
  start would be instantiated for every satellite, direction, instrument and mode, and p17
  would have 57 916 facts (almost all `running@a`) and 957 MB; with it, 11 262 and 138 MB,
  with the same actions. In the IPC 2006 temporal preferences, storage p06 and pathways
  p08 would exhaust 3 GB when grounding. In the discretized engine, the guard also
  excludes what events change, since they fire at the same instant.

  The guard goes in the precondition of `a@start`, so both the search (it stops generating
  starts that the invariant would drop) and the heuristic (it asks for the `over all`
  conditions already at the start, like POPF and TFD) see it. It changes the trajectory of
  almost every search. Removing it from the relaxation was tried: with the same successor
  order, BFWS made exactly the same searches, but with temporal preferences the portfolio
  solved 3 fewer instances (with somewhat better plans).
- **Temporal state.** A `State` (facts and values) with an attached timeline. It is
  compared only by its logical part (POPF-style memoization), so duplicate detection works
  as in classical planning. It is incomplete: a repeated state with a different schedule
  is dropped, and with deadlines (TILs, timed constraints, maximum durations) that other
  schedule might have reached the goal. That is why, if the search empties the queue after
  dropping some repeated state, it ends with `failed` (no plan found), not with
  `unsolvable` (there is none): trucks p01 with `lazy-ff-pref` would empty the queue and
  claim it had no solution, and it does have one.
- **Timeline on demand.** When a successor is generated its timeline is checked, but not
  stored: the state only remembers where it comes from and recomputes it when it is
  expanded. The search generates many more states than it expands, and each timeline grows
  with the plan length: storing them all cost 10 to 14 KB per state in large tasks
  (3.5 GB in zenotravel 17 with only 3000 expansions), and this way, 2 to 4.
- **Orderings.** For each fact and variable, the last step that changed it and the steps
  that have read it since are stored. A new step goes, with a separation ε, after the last
  change of what it reads or modifies and after the steps that read what it writes.
  Independent steps are not ordered: the temporal network makes them simultaneous, and the
  plan comes out in parallel even if the search chose them in sequence. Two details so
  that VAL accepts the plans, especially with TILs:
  - a write that does not change the value (adding a fact that is already true) is also
    separated from the readers, because VAL does not accept that at the same instant one
    action reads a fact and another writes it. Only unconditional effects and conditional
    effects that fire count;
  - the `running@a` facts count like the others: a new start of the same action goes after
    its previous end also in time, not only in the sequence.
- **Temporal network.** Constraints `t_j ≥ t_i + w`: the orderings (ε) and the duration of
  each action (minimum and maximum). The earliest times are maintained with incremental
  propagation. Only a maximum duration can create a positive cycle, and it necessarily
  goes through the new step, so it is detected when the step is added. The earliest times
  are the schedule of the plan.
- **Durations.** Fixed (`=`) or with inequalities; with functions, they are evaluated at the
  start (or at the end, with `(at end ...)`). `?duration` in effects is replaced by the
  duration when it is fixed and does not depend on changing functions; otherwise, the
  linear program decides it (see below).
- **Invariants.** `over all` conditions are checked after each step in all the running
  actions. Their facts and variables count as read by the start and by the end, so the
  steps that change them fall outside the interval.
- **Conditional effects** with the condition at the start and the effect at the end: an
  auxiliary fact remembers, at the start, whether the condition held. With `over all`
  conditions, the fact is a **watcher**: it is switched on at the start if the conditions
  hold and the engine switches it off as soon as the `over all` condition stops holding
  during the action.
- **ε = 0.01.** VAL groups as simultaneous the happenings closer than its tolerance (0.01 by
  default), so with ε = 0.001 (POPF's value) a plan only validates with
  `Validate -t 0.001`. With 0.01 the plan is valid with the default settings, at the price
  of 0.01 units per dependency.
- **Metric.** Only `(:metric minimize (total-time))`, except with preferences (see
  "Preferences and anytime search"): its value is the total duration. The search counts
  steps (unit cost); it does not optimize the duration.
- **Output.** `t: (action args) [d]` in IPC, with `; makespan = …`; in JSON, `time` and
  `duration` in each step and `makespan` in the plan.
- **Linear programming** (COLIN-style; `grounding/temporal_lp.py`, with HiGHS through
  SciPy). The **linear variables** are those that change with continuous effects
  (`(increase f (* #t r))`) and those that receive values that depend on `?duration` in
  actions with non-fixed duration. Their conditions and effects are taken out of the
  search actions and decided with one linear program per state:
  - unknowns: the time of each step and the value of each linear variable after each
    change; between two changes it evolves according to the sum of the active rates;
  - constraints: the temporal network, the conditions of each step, the effects,
    `?duration = t_end − t_start` and the invariants at each end of a segment (a linear
    condition on values that change linearly holds on a segment if it holds at its ends);
  - objective: minimize the sum of the times. It is only solved when the step touches
    linear variables or delays earlier steps.

  The state stores the value of each linear variable after its last change, which is what
  the heuristic and the goal see. Goal conditions on linear variables are checked by adding
  them to the program (they may be satisfiable with a different schedule, e.g. by
  stretching an action), and the final schedule includes them. The heuristic uses a version
  of the task with the continuous effects integrated at the end (rate × duration) and the
  conditions on linear variables. With variables that change continuously the state space
  is infinite: the search cannot prove that there is no plan. Also supported: continuous
  effects inside `forall` (expanded when grounding) and inside `when` with `at start`
  conditions (the rate is decided at the start); non-fixed `?duration` in `at start`
  effects (the duration is an unknown, between its bounds, linked to `t_end − t_start` at
  the end); durations that depend on linear variables (their bounds are constraints of the
  program); and conditional effects on linear variables with classical conditions. A
  hand-made certification domain (`greenhouse`, in `tests/test_temporal_engine.py`)
  gathers all of them, validated with VAL: there are no public benchmarks that use them.
- **Limits of linear programming** (clear error): conditions of conditional effects on
  linear variables and disjunctive invariants on them (the effect would depend on which
  solution is chosen: it would need integer linear programming), non-linear invariants and
  conditions, and non-fixed `?duration` in the rate of a continuous effect (rate × time is
  not linear).

## Derived predicates

- **Grounding.** Each `:derived` rule is instantiated as one more schema of the
  reachability fixpoint: its condition acts as precondition and its head as add effect.
  Each reachable clause of the condition (in DNF) gives an `Axiom` (derived fact,
  condition, stratum). Derived predicates are never static.
- **State.** Derived facts are in the same bitset as the others. `GroundTask.apply`
  recomputes them in each successor with `AxiomEvaluator` (`pyplanner.grounding.axioms`),
  and the initial state already comes with them computed. That is why the search shortcut
  that applies the action directly is disabled when there are axioms.
- **Evaluation by strata.** The stratum of a predicate is the lowest one that leaves the
  predicates it uses positively in its stratum or before, and those it negates strictly
  before. In each stratum the least fixpoint is computed with counters, like Fast
  Downward's *AxiomEvaluator*:
  - the part of the condition that does not change within the stratum is checked once;
  - the facts of the stratum itself (recursive rules) decrease a counter;
  - axioms are grouped by their first positive external fact, and only those of the true
    facts are checked.

  Negation is closed-world: a derived fact is false if no axiom makes it true. In psr it
  costs about 50 µs per state.
- **Heuristics.** Each axiom is a relaxed action of cost 0. They all share a dummy operator
  of cost 0 (index `len(task.actions)`) that is never a helpful action, so the Numba and
  PyTorch kernels do not change. Conditions that **negate** a derived fact are assumed true
  in the relaxation, as Fast Downward does by default. Its complement is not added by any
  action: it depends on all its axioms ceasing to hold, and treating it as one more fact
  would report dead ends that are not. It is an approximation: in psr,
  `(forall (?b) (not (affected ?b)))` does not count in the heuristic.
- **Temporal engine.** Axioms are recomputed after each step, before checking invariants
  and watchers. No action writes a derived fact, so reading one counts, for the POPF-style
  orderings, as reading all the facts and variables it depends on transitively. It is
  correct, although it may order more than necessary. A derived predicate that depends on
  a linear variable (with continuous effects or non-fixed `?duration`) gives a clear
  error: the linear program cannot decide its value.

## Timed initial literals

- **Compilation.** Each distinct instant *t_k* of the TILs is a classical action `til@k` of
  cost 0 that applies its literals. They are chained with `til-done@k` facts, so the
  search applies them in order and when it suits it. It is done at the AST level, like
  the start and end actions: that way the grounder treats as non-static the facts that
  only the TILs change (`visible` in satellite) and the heuristic sees them as relaxed
  actions. A problem with TILs is always a temporal task, even if the domain has no
  durative actions.
- **Fixed instant.** In the temporal network, the step of `til@k` has an edge from the
  origin with weight *t_k* and another one back with weight −*t_k*. If the orderings force
  it to go later, the successor is dropped. If a delay propagated by a maximum duration
  ends up moving the origin, too.
- **Deadlines, as in OPTIC.** Every step that the sequence puts before a TIL must happen no
  later than it: an edge back to the origin with weight −*t* of the next pending TIL.
  Without this, the state would keep believing true a fact that a TIL has already
  deleted, because time passes even if the search does not apply the TIL. The POPF-style
  orderings do the rest: a TIL that deletes a fact goes after the steps that read it.
- **Plan.** TILs do not appear in the output plan.
- **Memoization and parents.** In a temporal task, two equal states can have different
  timelines. When it finds a better path to a known state, the eager search only changes
  its parent if it has not expanded it yet, and then it also stores the state of the new
  path, with its timeline. With TILs (cost 0) this happens often, and changing the parent
  of an already expanded node gave plans that did not meet the deadlines when replayed
  from the start.
- **Deadlines in the heuristic** (`pyplanner.heuristics.deadlines`). In tasks with TILs,
  the heuristic is wrapped in `DeadlineHeuristic`, which prunes the states that cannot meet
  the deadlines and leaves the value and the helpful actions to the base heuristic:
  - **Earliest instants.** A Dijkstra over the relaxation, like h_max, where the cost is
    absolute time:
    - true facts are true since the last step that changed them;
    - the end of a durative action goes, at the earliest, after its start plus the minimum
      duration (running actions, from their real start);
    - pending TILs happen at their instant.
  - **Windows.** Facts that only TILs change are only true in the intervals given by the
    state and the pending TILs. A precondition on them delays the action until the first
    interval that satisfies it, and an `over all` condition requires the whole action to
    fit.
  - **Soundness.** Everything is a lower bound, so a state with a solution is never
    pruned.
  - **Implementation.** The inputs of each state are prepared with NumPy. The Dijkstra has
    a reference version in Python and another one in Numba (`compiled_deadlines`), with the
    same result; it stops as soon as it reaches the goal. It costs about the same as the
    compiled h_FF (1 to 2 times).
  - **Timeline.** It needs the timeline of each evaluated state: it computes it without
    storing it (`TemporalState.peek_timeline`), so as not to lose the memory saving of not
    storing them.
  - **Deferred in eager searches.** Evaluating every successor with the check doubled the
    time in the cases where it saved no expansions: most successors are never expanded.
    The eager search asks the heuristic for it (`Heuristic.defer_dead_end_check`) and runs
    it when each node is popped from the queue. A node pruned that way does not use up its
    queue's turn (`retry`), so the order of the live nodes is the same as if it had been
    pruned when generated, and the search expands the same nodes. The lazy search already
    evaluates when popping from the queue.
  - **Without pending TILs** nothing is checked: there are no deadlines any more, and the
    relaxation of the base heuristic detects the same dead ends.
  - **A relaxed plan that respects times:** that is h_TFF (see the next section), a
    separate heuristic that includes this check.

## Temporal relaxed plan: h_TFF

h_FF does not see time: in satellite, its relaxed plan uses antennas whose visibility
window has already passed, and its helpful actions lead to dead ends. h_TFF (`tff`;
`heuristics/temporal_ff.py`) extracts a relaxed plan that respects times, like the
temporal planning graph of POPF and OPTIC (Coles, Coles, Fox and Long, 2010):

- **Earliest instants:** those of the deadline check (the Dijkstra above, with minimum
  durations, pending TILs and windows).
- **Supporter of each fact:** the relaxed action with the lowest index among those that
  give it its minimum (instant, level), with h_FF's level (0 if true in the state, 1 + the
  maximum of the preconditions). It does not depend on the order of the computation, and
  the relaxed plan has no cycles.
- **Goal clause:** the one that holds earliest (with its windows); if none can hold, the
  state is a dead end. These are the same dead ends as the deadline check, so h_TFF is not
  wrapped in it.
- **Value and helpful actions**, as in h_FF: the cost of the operators of the relaxed plan
  (times their numeric repetitions) and the operators of the plan with all their
  preconditions true, also those of the facts with windows.
- **Three implementations with the same result:** the Python reference (Dijkstra by
  (instant, level)), Numba and PyTorch. In PyTorch, instants and levels are two fixpoints
  over the batch. Levels are computed only over the *tight* edges (those that give the
  fact its exact instant): h_FF's trick in PyTorch (the last iteration in which the cost
  went down) does not work with maxima, because a precondition with slack can be reached
  by a shorter path that is not the earliest one. Fitting the windows uses the intervals
  of each state in a padded tensor.
- **Only with TILs.** Without deadlines, the only thing that changes with respect to h_FF
  is choosing supporters by instant instead of by cost, and that gives worse relaxed plans
  (like h_max compared to h_add): temporal zenotravel went from 17 to 13 solved instances
  and driverlog from 16 to 14–15. So in tasks without TILs (temporal or not) `tff` creates
  h_FF, with its wrappers, and its configurations work for any task. Removing from h_FF
  only the actions that do not fit in time, while keeping the supporters by cost, does not
  help: they are almost never the ones it chooses (in satellite, the same expansions as
  h_FF).
- It costs about the same as the compiled h_FF (satellite p02: 0.04 s per 1500 states each,
  plus preparing the inputs).

## Trajectory constraints

PDDL 3 hard constraints (`:constraints` of the domain and of the problem) are engine
**monitors** (`pyplanner.grounding.monitors`). Each ground constraint (`forall` is expanded
over the objects) is a small automaton with its own facts in the state, updated after each
step with the new state, like the axioms but with memory.

- **Untimed:** `always` drops the step that violates it; `sometime` has a `done` fact that
  the goal requires; `at-most-once` remembers whether φ has held and whether it has stopped
  holding; `sometime-before` remembers whether ψ has held in an earlier state;
  `sometime-after` leaves φ pending until ψ holds, and the goal requires nothing to be
  pending; `at end` is added to the goal.
- **Timed:** they use instant **marks**. In a temporal task, each instant is a TIL without
  literals (its `til-done@k` fact is the mark): by OPTIC's rule, the steps before a
  pending TIL happen no later than it, and the deadline heuristic understands the windows
  they create. In a classical task, the instant is the number of steps (as in VAL, action k
  happens at instant k): a step counter, saturated at the last instant that matters,
  switches the marks on.
  - `within t φ`: `done` if φ holds before the mark;
  - `hold-after` and `hold-during`: drop the step that violates φ with the appropriate
    marks;
  - `always-within t φ ψ`: in classical tasks, a counter of the steps it has been pending;
    in temporal tasks, the timeline remembers the step that triggered φ and, when ψ holds,
    the temporal network receives the edge "ψ no later than t after φ", like a maximum
    duration.
- **Orderings in the temporal engine.** A step that can change something watched reads
  and writes the monitor's facts (and reads its marks), so those steps are ordered in time
  as in the sequence: the result of a monitor depends on the order.
- **Heuristic.** The good outcomes of the monitors (`done`, removing what is pending) are
  relaxed rules of cost 0 with the axioms' dummy operator. Those that only prune states do
  not count, so the relaxation stays optimistic.
- **Cost.** A monitor is only re-evaluated if the step changes something in its formulas
  or its marks.
- **VAL** validates `within`, `hold-during`, `always-within` and the untimed operators, but
  its `hold-after` does not work (it fails with valid plans or crashes): those plans are
  checked with our own simulation.

## Preferences and anytime search

A preference is a desirable condition: each violation adds to the metric the coefficient
of its `(is-violated p)`, its **weight**. The grounder compiles them into action costs
(Keyder and Geffner, 2009; `pyplanner.grounding.preferences`), so the search only minimizes
costs and the cost of a plan is its metric.

- **Weights.** They come from the linear form of the metric, which must be minimized.
  Preferences that do not appear in it (or weigh 0) do not count and are not compiled.
  Without a metric, none counts (a note is left). A non-linear metric or a negative weight
  are rejected with a clear error.
- **Goal and constraint preferences** (evaluated at the end of the plan). Each instance (one
  per assignment of the surrounding `forall`; `is-violated` counts the violated ones) is
  settled in a **final phase**:
  - `_end` (cost 0) is applied with the hard goal satisfied; after it, no domain action is
    applicable;
  - each instance is settled with `_collect` (cost 0) if it holds or with `_forgo` (cost:
    its weight) if not;
  - they are settled in a fixed order, so the final phase does not multiply the paths.
- **Constraint preferences.** They are **soft** monitors: where a hard one would drop the
  step, a soft one marks its `violated` fact, deletes the others and stops updating, and
  that step costs its weight on top (a transition **penalty**, which the search adds to g):
  once violated, it can no longer be avoided. Those that can only be violated mid-plan
  (`always`, `at-most-once`, `sometime-before`, `hold-after`, `hold-during`) do not go
  through the final phase; those decided at the end (`sometime`, `within`, what is pending
  in `sometime-after` and `always-within`) are settled in it according to `done` or what
  is pending. `(preference p (at end φ))` is like a goal preference.
- **Trivial constraints.** Those that hold in any plan because, with the static and the
  unreachable facts, their formulas become true or false (`always` of something always
  true, `sometime-before` of something impossible...) are dropped, hard or soft. A `forall`
  over `always`, `hold-*` or those with a trigger is instantiated like an action: only with
  the assignments that make reachable the atoms needed to violate or trigger it. In
  storage, the constraint on incompatible crates in adjacent areas only goes over the
  connected areas: in p20 it goes from 2.8 million instances to the 37 000 that matter, and
  grounding from 197 s and 2.7 GB to 6 s and 350 MB.
- **Precondition preferences** (they count at each application). The action is split
  according to which instances it satisfies: the variant that violates some costs their
  weights on top and requires them to be violated, so the cost is exact. With more than
  256 variants per action, error.
- **Step cost.** If the metric only measures preferences, actions would cost 0 and the
  heuristic would not tell a short plan from a long one. Each domain action also costs a
  thousandth of the smallest weight: satisfying is preferred to forgoing, and between two
  equally good plans, the shorter one. The reported metric discounts it.
- **The plan** does not include the final phase (its actions start with `_`, which cannot
  start a PDDL name). It reports the metric and the violated instances
  (`; violated preferences: ...` in IPC, `violated` in the JSON).
- **Heuristic.** The final phase is part of the relaxation: h_FF adds the weights of the
  preferences it cannot satisfy and the step cost of those it can.

**Anytime search** (`pyplanner.search.anytime`). With preferences, the first plan is
rarely the best, and searching for the optimum in one go is not viable in large tasks. As
in LAMA, searches are chained, each one with the **bound** of the best plan found:

1. the chosen configuration with **unit cost**, without a bound: any plan, as soon as
   possible (its real cost is computed afterwards). With the real costs, a greedy search
   chases the preferences and may not terminate: in storage, without a hard goal, it
   explores endlessly states where they look reachable;
2. the chosen configuration with the real costs and the **incumbent** (see below);
3. WA* with h_FF and helpful actions, with weights 5, 3, 2 and 1 (lazy or eager, with the
   exploration and the tie-breaking of the chosen one);
4. A* with h_max, which is admissible: if it empties the queue, the best plan is
   **optimal**.

After each plan it moves on to the next one; the last one repeats while it improves. It
ends when time runs out (60 s by default, `-t` to change it), when the optimum is proven or
when the memory limit is reached.

- **Incumbent.** Each expanded state in which the hard goal holds is a possible plan: from
  it, the final phase costs exactly the weights of the preferences it does not satisfy. If
  that plan improves the bound, the search records it, lowers the bound to it and goes on;
  if it ends without another plan, it returns the incumbent. A greedy search that moves
  forward satisfying preferences thus leaves better and better plans, even if it never
  enters the final phase by itself.
- **Pruning by bound.** The search drops the nodes with g (which includes the penalties)
  equal to or greater than the bound; with an admissible heuristic, also those with g + h.
  If an eager search with reopening (A* or WA*) empties the queue, there is no plan better
  than the bound.
- **Optimal** means minimum search cost: metric plus the step cost. It is the one with the
  lowest metric unless a better one needs a thousand more steps per unit of the smallest
  weight.
- **In the portfolio**, each member runs its anytime search starting with its
  configuration, reports each better plan and all of them share the best bound
  (`multiprocessing.Value`). It ends when time runs out, when all of them have finished or
  when one proves the optimum.
- **Storage** (IPC 2006, qualitative preferences): p01 and p02 reach the proven optimum
  (metrics 0 and 1) in less than a second, with any portfolio. The metrics match VAL's.
- **Maximized metrics:** maximizing is minimizing the negation. The costs are those of the
  negation and the plan metric is reported with its sign:
  `(maximize (+ (* 9.9 (- 1 (is-violated d1))) …))` is minimizing the weight 9.9 of `d1`.
  If that leaves some weight or cost negative, the error (or the unit cost) is the usual
  one.
- **In temporal tasks:** the task of starts and ends is compiled the same way (final phase,
  soft monitors with the instant marks of the trajectory constraints, action variants),
  with these changes:
  - the preferences of an `at start` or `at end` condition of a durative action are
    preferences of the precondition of its start or end action; those of `over all` are
    not supported (clear error);
  - the final phase actions are not steps of the temporal network (they do not lengthen
    the plan), and `_end` also requires that no durative action is still running;
  - the order of the steps that change something a monitor watches is computed with an
    index from watched facts and variables to their monitors, computed once, instead of
    going through all the monitors for each action: in storage p10 (4139 monitors and 9546
    actions), grounding takes 1.3 s instead of 26 s;
  - `total-time` in a metric with preferences is the plan duration, which is not an action
    cost: the search does not optimize it (note) and the plan metric adds it with its
    weight. Without preferences, the metric of a temporal task is still only
    `(minimize (total-time))`.
- **Limits.** VAL computes goal preferences wrongly if the problem has an empty
  `(:constraints (and))` section; a `sometime-before` with a strict numeric comparison
  (`(> x 0)`) always looks violated to it; and in pathways (IPC 2006) it reports p0a as
  violated but does not add its weight to the metric.

## Processes and events

Domains with processes and events (PDDL+) go to an engine with **discretized time**
(`pyplanner.grounding.discrete`), in the style of ENHSP, DiNo or UPMurphi. Temporal tasks
without `:time` stay in the temporal engine, which is more precise, except those it does
not support because of their continuous effects (non-linear rates, like the refuelling of
generator-nonlinear, where the flow grows with the time spent refuelling): those go to the
discretized engine, with a note. If it does not support them either, the error is the
temporal engine's.

- **Search steps:** the instantaneous actions, the start and end of each durative action,
  and a `_wait` action that lets δ time units pass (1 by default; `--delta` in the CLI,
  `delta` in `solve`). The wait is shortened if a durative action reaches a bound of its
  duration earlier, so that it can end just in time. There is also a wait until the next
  change (`_wait-until`, see "Rates set by actions").
- **Exact integration** (`pyplanner.grounding.flows`): during a wait, the variables follow
  the dynamics of the active processes. They are integrated with Taylor series, which for
  polynomial dynamics (car: the speed grows with the acceleration and the distance with
  the speed; generator: a flow that grows with the square of time) **terminate** and give
  the same values as VAL. If the series does not terminate, Runge-Kutta 4 with 100
  substeps per unit. Values are rounded to 9 decimals: sums of steps do not accumulate
  noise and equalities such as `(= (v) 0)` can hold.
- **Events at their exact instant.** After each change, events fire in cascade. If the
  precondition of an event starts to hold during a wait (or the set of active processes
  changes), the instant is found by bisection on the exact flow, as VAL does, and the wait
  goes on until δ is completed. Without this, an event would be detected at the end of the
  wait: in generator, a tank would be left with negative fuel and the generator would
  receive too much, and the search would end up exploiting that error (5 of 8
  generator-events instances; 8 of 8 with the exact instant).
- **Invariants:** the `over all` conditions of the running durative actions are checked
  after each change (not in the middle of a wait).
- **An instantaneous action** is followed by a wait of ε = 0.01: VAL does not accept two
  actions at the same instant, and during that ε the processes keep acting, as in VAL.
- **Durative actions:** they are compiled into a start (it starts a clock and sets the
  bounds of its duration), a process `a@run` (the clock and the continuous effects) and an
  end (the clock within the duration). `over all` conditions are invariants.
- **Cost:** δ per wait and about 2ε per action (ε for the time that passes and ε for being
  one more step; without the second, A* returned plans with dozens of useless actions that
  made time pass ε by ε). It is 1/64, so that the sums of costs are exact, and the wait
  until the next change costs what it advances. The reported metric is the plan duration.
- **Heuristic:** in the relaxation, the wait is an action with one conditional effect per
  process (its rates × δ); *h^hbd* counts the repetitions if the rate is constant, like the
  clock of a durative action. Events are relaxed rules of cost 0 over facts. What the
  relaxation does not see, the invariants and the rates set by actions, is added by the
  next two sections.
- **Novelty:** variables that change continuously do not count. They take a new value in
  each wait, so every state looked new and the table grew endlessly (in generator-events
  p06, 85 000 atoms and 2 GB in 20 s). Capping their distinct values (16, 64, 256, 1024)
  instead of ignoring them gives results that depend on the cap with no trend.
- **Validation with a finer step:** the plan is simulated again with δ/10. If it fails (an
  invariant that breaks in the middle of a wait, for instance), `solve` searches again
  with δ/2, up to 3 times, within the time limit. It is stricter than VAL, which only
  checks invariants at *happenings*: in generator p02, a plan that VAL accepted left the
  fuel at −1.08 between two *happenings*. Each wait is walked in chunks of δ/10 to check
  the instants in between, but it ends in the search's state: the rounding of thousands of
  chunks would accumulate, and a plan that runs out of fuel just at the end of a long wait
  would be left at −10⁻⁹.
- **No plan:** exhausting the search proves nothing (with a smaller δ there could be a
  plan): it ends with `failed`.
- **Timed initial literals:** they are events of a global clock, `til@clock`, that advances
  at rate 1, so they fire at their exact instant. A wait that ends exactly at the instant
  of a TIL also lets ε pass: VAL does not accept that an action reads a fact at the same
  instant a TIL changes it.
- **Trajectory constraints:** the monitors of the tasks without processes. The instants of
  `within`, `hold-after` and `hold-during` are events of the same clock (TILs without
  literals). As in VAL, they are checked at *happenings*: after each action, and after a
  wait only if an event or a TIL changed the facts. The end of a wait in which nothing
  happens is not a *happening*: also checking them there accepted plans that VAL rejects
  (a `within` that holds in the middle of a wait). The wait until the next change also
  goes through the monitors.
- **Conditional effects in durative actions.** With `over all` conditions, a watcher like
  the temporal engine's: it is switched on at the start if its conditions hold and
  switched off as soon as the `over all` condition stops holding. During a wait, facts only
  change with events, at their exact instant, and it is checked there: an event that
  breaks the condition switches the watcher off even if another one restores it before the
  end of the wait. A conditional continuous effect is decided at the start (as in the
  temporal engine, it can only depend on `at start` conditions): a fact `rate@a@k`
  remembers whether the condition held, and a separate process, `a#k@run`, applies the
  rate while the action is running.
- **Preferences:** as in the tasks without processes, compiled into costs. The final phase
  (`_end`, `_collect`, `_forgo`) goes after the waits and stays outside time: it does not
  let ε pass or fire events. `_end` also requires that no durative action is running. The
  metric is no longer the duration: each wait costs what a step costs. Constraint
  preferences are the soft monitors of preferences, also at *happenings*.
- **Limits** (clear error): `always-within` with processes and events.

## Heuristic with processes and events

The relaxation ignores what worsens conditions, so it does not see invariants: for it, the
generator can run for 1000 units with 960 of fuel and waiting is enough. Greedy searches
would wait until the invariant broke and it was too late to refuel. What is missing is
added by `pyplanner.heuristics.projection`:

- **Durations in the relaxation.** The duration bounds that do not depend on anything that
  changes are compared with the clock directly, without an intermediate function the
  relaxation would not know how to satisfy. Assigning a constant (the clock to 0 at the
  start) does not count as achieving any condition on that variable (see "Numeric
  engine"). Clocks are 0 from the initial state. With this, h(initial) in generator is
  about 1000 waits (without it, 0.04).
- **Projection** (`DiscreteTask.shortfall`): the passing of time without doing anything is
  simulated until the mandatory end (the minimum duration) of each running durative
  action with an invariant. If the invariant breaks earlier, what was missing counts, and
  the heuristic adds those shortfalls. In the simulation, each running action ends when it
  reaches its maximum duration (or the minimum, if it has no maximum) or when its own
  invariant breaks: the effects of its end are applied even if its conditions do not hold,
  so that a process that depends on them also stops acting (a refuelling stops when the
  tank is full). Without a maximum or a break, an action would go on forever: refuelling
  endlessly would cover any shortfall.
- **Lookahead.** If only the running actions counted, starting the generator would raise
  the value at once (from 1000 to 1040 in generator-toricelli p02), and the lazy search
  would keep exploring the earlier states: starting and ending a refuelling within ε gives
  a new state each time (93 000 states without getting past the initial state). That is
  why starting each durative action with an invariant and a minimum duration that the base
  gives as helpful is also simulated, and the value is the largest of the shortfalls.
- **Helpful actions.** Those of the base do not fix a shortfall (for the relaxation,
  waiting is enough), and the lazy search, which gives successors the parent's value,
  would go down waiting until refuelling no longer helped. When a state with a shortfall
  is expanded, the helpful actions are those that reduce it. It takes as many simulations
  as candidate actions (those that change values or facts of the conditions of processes
  and events), so they are only computed in the states that are expanded
  (`Heuristic.helpful`), not in all the evaluated ones.
- **Breaks that do not last.** The bisection of a wait only sees the changes that persist
  at the end of the segment, which in a wait of δ is as intended (the validation with δ/10
  finds the rest). The projection advances segments of hundreds of units. In
  generator-events, with the tanks hooked up late, the fuel goes below 0 and recovers (the
  flow grows with the square of time), and a projection by segments would not see it. That
  is why it checks between every two instants at which one of the numeric conditions
  involved becomes zero. With an exact flow, a polynomial condition is a polynomial in
  time: its coefficients are its derivatives along the flow (`Flow.crossings`), computed
  once per condition, and its roots, with closed formulas up to degree 2. With
  Runge-Kutta, only the bisection.
- **Performance.** The flow rounds only the variables that change, and the values after an
  action or an event, only those that change (the others are the same objects, already
  rounded). Checking for changes during a wait evaluates what belongs to the starting
  state once, and only the processes with numeric conditions (facts do not change while
  time passes). That way, the search of generator-toricelli p09 with `bfws-ff-pref` takes
  half the time (from 13.1 to 6.2 s with the profiler).

## Rates set by actions

In some domains, the rate of a process is set by actions: for instance, an economy with a
production process whose rate depends on the workers that are assigned or removed, where
reaching the goal requires chaining actions and waits of thousands of time units. Without
these mechanisms, h_FF would treat the wait as if a single one were enough for any amount,
the only helpful action would be waiting and the search would not move forward:

- **Variable increments in the relaxation** (`pyplanner.heuristics.relaxation`): the effect
  of the wait, rate × δ, is not constant, and *h^hbd* would treat it optimistically: one
  wait would be enough for any amount, even with a rate of 0. The edge requires the change
  to be positive (one more node of the relaxation, achieved by the actions that raise the
  rate) and counts the waits with the state's rate or, if smaller, with the smallest one a
  single action gives: ⌈deficit / max(rate·δ, ρ)⌉. That way, starting a rate does not
  raise the value and speeding it up lowers it, and the helpful actions become those that
  start production.
- **Waiting until the next change** (`_wait-until`, `DiscreteTask.wait_until`): it advances
  to the first instant at which a numeric condition holds, of an action or an event
  applicable by its facts or of the goal, or a durative action reaches a bound of its
  duration. The instants come from the exact roots of the flow (`Flow.crossings`); an
  invariant that breaks along the way invalidates the step. Collecting 150 units at a
  constant rate is one step instead of 150. It costs what it advances
  (`GroundTask.transition`), so A* still minimizes the duration, and can improve it: in the
  tank of the tests, from 2.02 to 1.67, because it does not wait whole steps of δ. It is
  not in the relaxation: it is helpful whenever `_wait` is.
- **Exact costs.** With repetitions in the hundreds or thousands, sums of costs of 0.02
  would accumulate rounding noise, and in CUDA (which adds in a different order) ties
  between supporters would be broken differently: h_FF would give a different value from
  the reference. The cost of an instantaneous action is 1/64 (a binary fraction close to
  2ε) and the sums are exact: Python, Numba and PyTorch agree in every state.
- **Quality.** The greedy search finds sooner the plans that wait until the next
  condition, and they are not always the shortest: in car-nodrag, accelerating, waiting
  until d ≥ 30 and braking takes 15.50 (braking earlier, on the δ grid, 11.03).

## Heuristics

| Name | Admissible | Helpful actions | Numba | PyTorch |
|------|------------|-----------------|-------|---------|
| `blind` | yes | no | no | no |
| `goalcount` | no | no | no | no |
| `hmax` | yes | no | yes | yes |
| `hadd` | no | no | yes | yes |
| `ff` | no | yes | yes | yes (costs and relaxed plan in tensors) |
| `tff` | no | yes | yes | yes (instants, levels and relaxed plan in tensors); in tasks without TILs it is `ff` |

**Devices** (`SearchConfig.device`, `--device`):

| Device | What it is | When to use it |
|--------|------------|----------------|
| `auto` (default) | `numba` if installed; otherwise `python` | always, unless another one is to be forced |
| `python` | reference implementation in pure Python | tests and machines without Numba |
| `numba` | the reference compiled with Numba (`pip install -e ".[fast]"`); 12–19 times faster | the fast option for normal searches |
| `cpu`, `cuda` | PyTorch in batches (`pip install -e ".[gpu]"`) | only with large batches; on GPU, 4 times faster than Python with batches of 256 |
| `hybrid` | each batch in Numba or on the GPU according to its size, with a threshold calibrated at the start | with adaptive batches and large tasks (see "Adaptive batches") |

All three give **exactly the same values and helpful actions** as the reference; the tests
check it on the test problems and on hand-made tasks, also random ones, with many ties and
zero costs.

- **Interface:** `evaluate_batch(states) -> list[Evaluation]`. Each `Evaluation` carries the
  value (`inf` if it is a dead end) and the helpful actions.
- **Relaxed task** (`RelaxedTask`, shared by both implementations):
  - **Complementary facts:** each fact that appears negated has a "not-p" fact, true if p is
    false and made true by the actions that delete p. That way negations become positive
    **only inside the relaxation**, as in FF.
  - **Relaxed actions:** one for the unconditional effects of each operator and another
    one for each conditional effect (with the condition added to the precondition), as in
    Fast Downward.
  - **Numeric conditions:** one node per condition, with edges whose repetitions depend on
    the state (*h^hbd*, see above).
  - **Goal:** the value is the minimum among the clauses.
- **Python reference:** generalized Dijkstra over the delete relaxation. It stops when it
  has extracted all the goal facts, and it records the best supporter during the
  traversal itself, so h_FF does not have to look for it afterwards.
- **Numba** (`heuristics/compiled.py`): the same Dijkstra and the same extraction, compiled,
  over the relaxed task in CSR arrays. The whole batch is evaluated in one call and the
  states are decoded inside the compiled code. The first run after installing compiles the
  kernels (a few seconds) and stores them in the disk cache.
- **PyTorch:** a fixpoint over the whole batch with `index_add_` and `scatter_reduce`, using
  lists of action-fact edges rather than dense matrices. It computes in `float64`: with
  integer costs it is exact; with non-integer costs, the order of the sums can change the
  last bit of some value.
- **Tie-breaking (cost, level).** With zero costs (`:action-costs` allows them), the cost
  alone is not enough to choose supporters without cycles. Facts are ordered by
  (cost, level): the level of an action is 1 + the maximum level of its preconditions, and
  that of a true fact is 0. The best supporter is the relaxed action with the **lowest
  index** among those that give the fact its minimum (cost, level). It does not depend on
  the extraction order, so PyTorch reproduces it without Dijkstra: the level of a fact is
  the last iteration of the fixpoint in which its cost goes down (proof in
  `TensorRelaxation._fact_costs`).
- **h_FF:** the goal clause with the lowest value is used (the first one if tied). The value
  counts each **operator** only once (by its maximum repetitions), even if the relaxed plan
  uses several of its effects, and it is added in operator order in the three
  implementations.
- **h_FF in tensors:** the extraction of the relaxed plan is also done for the whole batch
  at once, layer by layer backwards from the goals (see `heuristics/tensor.py`).

## Search configuration

`SearchConfig` fully defines a search and is validated when it is created. Fields:

- `algorithm`: `gbfs`, `astar`, `wastar` (with `weight`), `iw` (see "Novelty search"),
  `ehc` or `mrw` (see "Local searches");
- `heuristic`;
- `preferred_operators`: two alternating queues, the normal one and the helpful-actions
  one;
- `lazy`: lazy evaluation; it cannot be used with A*. The actions of a node with the same
  priority go in a single queue entry, which reserves consecutive tie-breakers: they come
  out in the same order as one entry per successor, without its memory (with random
  tie-breaking, one per successor);
- `tie_breaking`: `fifo`, `lifo` or `random`, with `seed`;
- `exploration`: `"type-based"` adds a **type-based exploration** queue (Xie et al., 2014)
  that alternates with the others: it picks a random (h, g) type and a random node of that
  type. It keeps the greedy search from getting stuck in plateaus and in regions with many
  dead ends;
- `landmarks`: adds the landmark heuristic (h_LM) with its own queues, which alternate with
  those of `heuristic`, in the style of LAMA (see "Landmarks"); only with GBFS without
  novelty;
- `novelty`: novelty width (0, 1 or 2; see "Novelty search");
- `batch_size`; with `adaptive_batch`, the minimum of an adaptive batch that doubles (up to
  `max_batch_size`) after `plateau_patience` batches without improving the best h (see
  "Adaptive batches");
- `device`: `auto` (default), `python`, `numba`, `cpu` or `cuda` (see above);
- `max_time`, `max_expansions` and `max_memory` (MB of resident memory of the process;
  checked every 32 iterations by reading `/proc/self/statm`). Portfolio members and the
  CLI also have a hard limit (see "Command-line usage").

Guarantees:

- **Optimal:** A* with `blind` or `hmax` returns optimal plans, also with batches. In the
  eager search, a goal is only accepted if it is the first one popped from the queue in its
  batch.
- **Complete:** all configurations are, except IW(k) and random walks. The dead ends that
  are pruned are states from which the goal is unreachable even in the relaxation. IW
  prunes states by their novelty: if it empties the queue after pruning some, it ends with
  `failed` (no plan, but without proving there is none), never with `unsolvable`. Random
  walks do not end without a plan: the time limit stops them.
- **Cancellable:** `search(..., stop=event)` checks the event periodically.

Predefined configurations (`pyplanner configs`):

| Name | Configuration | Notes |
|------|---------------|-------|
| `gbfs-ff-pref-tb` | GBFS + h_FF + helpful actions + type-based exploration | |
| `gbfs-ff-pref` | GBFS + h_FF + helpful actions | |
| `lazy-ff-pref` | lazy GBFS + h_FF + helpful actions | evaluates far fewer states |
| `lazy-ff-pref-tb` | the previous one + type-based exploration (seed 1) | |
| `gbfs-ff-lm-pref` | GBFS + h_FF and landmarks in alternating queues + helpful actions | see "Landmarks" |
| `lazy-ff-lm-pref` | the previous one, lazy | |
| `lazy-ff-lm-pref-tb` | the previous one + type-based exploration (seed 1) | the best single configuration on classical tasks |
| `gbfs-tff-pref`, `lazy-tff-pref`, `lazy-tff-pref-tb`, `bfws-tff-pref`, `gbfs-tff-pref-tb`, `gbfs-tff-pref-hybrid` | the h_FF ones with h_TFF | in tasks without TILs, the same as the h_FF ones |
| `gbfs-ff` | GBFS + h_FF | |
| `gbfs-add` | GBFS + h_add | |
| `wastar-add-3` | WA* (w = 3) + h_add | |
| `gbfs-goalcount-rnd` | GBFS + pending goals, random tie-breaking | very cheap; adds diversity |
| `bfws-ff-pref` | BFWS: GBFS with priority (novelty, h_FF) + helpful actions | the most robust (solves all the test problems); in the portfolios, with `tff` (`bfws-tff-pref`, the 1-core one) |
| `bfws-ff` | BFWS with h_FF | |
| `bfws-goalcount` | BFWS with pending goals | without a relaxation heuristic |
| `iw1`, `iw2` | IW(1), IW(2) | incomplete; only for low-width problems |
| `ehc-tff-pref` | *enforced hill-climbing* (FF) + h_TFF + helpful actions | if it gets stuck, it goes on with GBFS; outside the portfolios |
| `mrw-tff`, `mrw-tff-pref` | random walks (Arvand), pure or with helpful actions (MHA) | incomplete; outside the portfolios |
| `astar-hmax` | A* + h_max | optimal |
| `astar-blind` | blind A* | optimal; uniform cost |

## Local searches

To add diversity to the portfolio, two searches that do not keep a global queue but move
forward from the best state found (`pyplanner.search.local`):

- ***Enforced hill-climbing*** (`algorithm="ehc"`; Hoffmann and Nebel, 2001): from the
  current state, a breadth-first search up to the first state with a lower h (or a goal),
  which becomes the current one. With helpful actions, the breadth-first search only uses
  those of each state and, if that finds no way out, it tries with all of them. If that
  fails too (a dead end the heuristic does not recognize), it goes on with GBFS with the
  same heuristic and the remaining time, like FF: that way it is complete.
- **Random walks** (`algorithm="mrw"`; Arvand, Nakhost and Müller, 2009): from the current
  state, random walks of 10 steps, and only the state each one reaches is evaluated (in
  batches). As soon as one improves the lowest h, the search jumps to it; after 2000 walks
  without improving it, it jumps to the best one anyway, and after 7 such jumps it starts
  over from the initial state. Walks get longer (×1.5 every 300 without improving). With
  helpful actions, each step chooses them with probability proportional to exp(Q/10),
  where Q counts the times the action has been helpful in the evaluated states (MHA). It is
  incomplete.

Both leave cycles in the plan (returning to an already visited state); they are removed at
the end if the plan without them is still valid and does not cost more. With preferences,
the *anytime* search goes from the random walks directly to WA*: they do not end without a
plan, so with the bound they would take all the time.

They are not in the portfolios: on the 97 measured instances that the 7-core portfolio did
not solve, they add none.

## Novelty search

Lipovetzky and Geffner (2012, 2017); code in `search/novelty.py` and `search/engine.py`.

- **Novelty** of a state with respect to the ones already seen: 1 if it makes true an atom
  that no previous state had, 2 if it forms a pair of atoms never seen together, 3
  otherwise. The atoms are the true facts; the falsity of the facts that appear negated in
  some precondition, condition or goal (if the goal asks for a light switched off,
  switching it off is something new: without this, IW(2) did not solve `lights`); and, for
  numeric variables, the (variable, value) pairs. In temporal tasks the internal
  `running@a` facts are excluded: with thousands of durative actions, they made novelty
  expensive and not very discriminating (from 33 to 29 solved instances).
- **Bitsets:** the atoms of a state form an integer. For the pairs, each atom keeps the
  bitset of the atoms seen together with it, so measuring and recording a state costs
  O(atoms of the state) operations on integers, in C.
- **BFWS** (`algorithm="gbfs"` with `novelty`): the priority is (novelty, h), with the
  novelty measured among the states with the same h value. Among states equally good for
  the heuristic, those that explore something new are expanded first. It is complete: it
  does not prune.
- **IW(k)** (`algorithm="iw"`): breadth-first search that drops the states with novelty
  greater than k. It solves right away the problems of width ≤ k (those that can be solved
  going only through new states), but it is incomplete.
- Only with eager evaluation: the novelty of a state is measured when it is generated.

## Landmarks

Code in `heuristics/landmarks.py` and `search/engine.py`. In the style of LAMA (Richter and
Westphal, 2010).

- **What they are.** A landmark is a fact that holds at some point in every plan. Those of
  the delete relaxation are computed, which are also landmarks of the task, over the facts
  of `RelaxedTask`: those of the task, the complements of the negated ones and the numeric
  conditions. That way there are also negative and numeric landmarks.
- **Computation**, once, from the initial state: the fixpoint
  `LM(p) = {p} ∪ ⋂ₐ ⋃_{q ∈ pre(a)} LM(q)`, with `a` the relaxed actions that achieve `p`
  (Zhu and Givan; Keyder, Richter and Helmert, 2010). The landmarks of the goal are the
  union of those of its facts; with a disjunctive goal, those common to its clauses. In the
  measured collections it takes at most 0.13 s.
- **Orderings:** natural (`l` goes before `l'` if `l ∈ LM(l')`) and greedy-necessary (`l` is
  a precondition of every action that can achieve `l'` for the first time, so it has to
  hold right before).
- **h_LM is path-dependent**, as in LAMA. Each node carries its *accepted* landmarks: one is
  accepted when it holds and its predecessors already are. The value is what is still to be
  accepted plus what has been accepted but must be achieved again: those of the goal that
  do not hold and those that must hold right before another one not yet accepted. If an
  already seen state is reached by another path, it keeps the landmarks accepted by both.
- **Alternating queues.** With `landmarks=True`, the search has two more queues (the normal
  one and the helpful-actions one) ordered by h_LM, which take turns with those of h_FF.
  The helpful actions of a node are those of h_FF plus those that achieve one of the next
  landmarks (the missing ones with all their predecessors accepted); these are computed
  when the node is expanded, not when it is evaluated.
- If the goal is not reachable even in the relaxation, h_LM is a dead end.
- **Cost:** 5 % to 15 % of the search time (profiled in barman, openstacks, parking and
  visit-all).
- Only with GBFS without novelty, eager or lazy.

## Adaptive batches and hybrid evaluation

Code in `search/engine.py` and `heuristics/hybrid.py`.

- **Adaptive batch.** The batch size goes back to the minimum as soon as the best h seen
  improves, and doubles after 8 consecutive batches without improving it (up to 256). While
  the heuristic guides, the search is the classic one; in plateaus it expands many nodes
  at once and evaluates all their successors together.
- **Hybrid evaluation** (`device="hybrid"`). Numba and PyTorch give exactly the same values,
  so each batch can go to either one without changing the search. Numba's cost per state
  does not depend on the batch but grows with the task; the GPU has a high fixed cost per
  call and grows less. At the start both are measured on the task itself (random walk) and
  the size from which the GPU is faster is set; if it never is, everything goes to Numba.
  Cost per state of h_FF (µs) with a GTX 970:

  | Task | Numba (any batch) | GPU, batch 64 | GPU, batch 256 |
  |------|-------------------|---------------|----------------|
  | slitherlink p02 (1920 actions) | 45–59 | 253 | 158 |
  | driverlog 15 (5184) | 550 | 445 | 280 |
  | zenotravel 15 (20 160) | 1765 | 861 | 756 |

  PyTorch on CPU never beats Numba.
- **Chunks.** PyTorch batches are evaluated in chunks of at most 5·10⁷ elements (states ×
  relations of the relaxed task): on Windows, the GPU that drives the display has a
  watchdog that aborts any computation longer than about 2 s (`cudaErrorLaunchTimeout`),
  and batches of thousands of states of a large task exceeded it.
- **A single search per GPU.** Two processes sharing the GPU lose all the advantage
  (zenotravel 14: 17.5 s with two, 3.4 s with one).
- **Mixed portfolio.** The GPU members start with `spawn` and the others with `fork`; if
  there is already a plan, those still starting are terminated right away. A GPU member
  costs a few tenths of a second on small tasks (gripper p01: 1.1 s against 0.9 s).

Results (temporal driverlog and zenotravel 12–17, one process, 60 s):

| Configuration | Total time | Highlights |
|---------------|------------|------------|
| `gbfs-ff-pref` (batch 1, Numba) | — | zenotravel 16: 34.2 s |
| `gbfs-ff-pref-ab` (adaptive, Numba) | 176 s | zenotravel 16: 25.2 s |
| `gbfs-ff-pref-hybrid` (adaptive, hybrid) | 136 s | zenotravel 16: 14.9 s; driverlog 17: 18.2 s |

- **The adaptive batch is neutral for the search** (with patience 4 it lost instances; with
  8, it does not). Its value is that it enables the GPU, which cuts 30–55% on large tasks.
- **In the portfolio it adds little on these tasks** (the lazy search already solves
  zenotravel in a few seconds), but it is the GPU member of the 6- and 7-core portfolios
  (see "Portfolio").

## Portfolio

**Which searches are launched.** It depends on the number of cores (`-j N`). There is a
predefined portfolio for each number of cores from 1 to 7 (`PORTFOLIOS` in
`search/config.py`), and each one contains the previous one:

| Cores | Searches | What is added |
|-------|----------|---------------|
| 1 | `bfws-tff-pref` | BFWS, the most robust: it solves all the test problems and gives the best plans with preferences; it runs in the calling process |
| 2 | + `lazy-ff-lm-pref-tb` | the lazy search with landmarks and type-based exploration, the best single configuration on classical tasks |
| 3 | + `gbfs-goalcount-rnd` | the cheapest and random one |
| 4 | + `wastar-add-3` | WA*, which gives the best plans with preferences |
| 5 | + `lazy-ff-pref` | the lazy search without landmarks |
| 6 | + `gbfs-tff-pref-hybrid` | adaptive batch and hybrid CPU/GPU evaluation, the only GPU member: it almost ties with GBFS with type-based exploration (one instance fewer, 1 % faster) and stays for large tasks, where the GPU pays off |
| 7 | + `gbfs-tff-pref-tb` | GBFS with type-based exploration |

- Chosen by measuring on 475 instances of the PDDLDomains catalog, classical and
  non-classical. Each one solves all the test problems. From 2 to 5 cores they are the best
  possible combination of their size; the 6- and 7-core ones, because of the GPU member,
  are one instance short of the best.
- **The h_FF members use `tff`** (h_TFF in tasks with TILs and h_FF, with the same search,
  in the others), except `lazy-ff-pref`: with h_FF it solves satellite p10 and p14, and
  with h_TFF it does not. The lazy search with landmarks has no `tff` version. In tasks
  with TILs (satellite and trucks, 22 instances), the 1-core portfolio goes from 6 solved
  to 8 and the 6- and 7-core ones from 11 to 12; in the others nothing changes.
- With more than 7 cores, the 7-core one is used. There are never more searches than
  cores.
- The 1- to 5-core portfolios do not use the GPU: they work the same on any machine and do
  not pay the PyTorch and CUDA start-up on small tasks. Without a GPU, the hybrid member of
  the 6- and 7-core ones evaluates everything with Numba.
- **By default 6 cores are used**, for maximum robustness, but limited to those the machine
  has (`os.sched_getaffinity`). On a single-core system the 1-core portfolio is used
  without having to ask for it.
- With `-c` the configurations are chosen by hand. Asking for more configurations than
  cores is an error.

**How they run.**

- One process per configuration; the first plan wins and the rest are cancelled. With
  preferences, the portfolio is *anytime* (see
  [Preferences and anytime search](#preferences-and-anytime-search)): it keeps the best
  plan within the time limit.
- `fork` is used when it is safe: without CUDA, without PyTorch loaded in the parent and
  with a single thread (also system threads: a program that embeds Python usually has
  many). In any other case `spawn` is used, and each child imports the main script again:
  if it calls the planner outside `if __name__ == "__main__":`, the child raises a clear
  error instead of repeating the script's work, and the parent reports the cause.
- If a member fails, it reports an `ERROR` status and a message, but it does not bring the
  portfolio down. That includes failing to start it and ending without reporting (detected
  right away, from the exit code).
- `spawn` members read the task from a temporary file, serialized once for all of them.
  Sending it through each child's start-up pipe would not be safe: CPython keeps the other
  end open while it writes, and if the child dies at start-up and the task does not fit in
  the pipe (64 KB; slitherlink p02 takes 589 KB), the parent would wait forever, even with
  a time limit.
- **Numba precompilation:** before launching the children, the parent compiles (or loads
  from the cache) the Numba kernels if some member is going to use them. Without this, a
  child cancelled in the middle of compiling would leave the cache incomplete, and every
  run would pay for the compilation and for the wait at shutdown. With `fork`, the
  children inherit the already compiled kernels.

## Command-line usage

```bash
pyplanner configs                                   # configurations and portfolios
pyplanner solve domain.pddl problem.pddl            # 6-core portfolio (or fewer)
pyplanner solve -j 1 domain.pddl problem.pddl       # a single core
pyplanner solve -j 2 -t 60 domain.pddl p.pddl       # 2 cores, 60 s at most
pyplanner solve -m 4000 domain.pddl p.pddl          # 4000 MB in total for the searches
pyplanner solve -c astar-hmax domain.pddl p.pddl    # optimal plan
pyplanner solve -c gbfs-add --device cuda --batch-size 64 domain.pddl p.pddl
pyplanner solve -f json -o plan.json domain.pddl p.pddl
pyplanner solve -t 30 domain.pddl preferences.pddl  # the best plan in 30 s
```

- The plan goes to standard output and the statistics to standard error; `-q` suppresses
  them.
- **Memory limit** (`-m`, or `memory_limit` in `solve`): it is split equally among the
  portfolio members. By default it is 80% of the available memory, so a long search never
  exhausts the machine's (a greedy search can generate hundreds of thousands of states per
  second). When the limit is reached, the search ends with `limit_reached` and explains it
  in the details.
- **Hard limit** (`pyplanner.search.memory`). The previous limit is checked every 32
  iterations and does not see what grows suddenly between two checks, nor grounding. That
  is why there is also a hard limit with `RLIMIT_DATA`: past it, any allocation fails and
  the search ends with `limit_reached`.
  - **Portfolio members:** each one sets it in its own process at start-up, slightly above
    its share (the larger of its share and what it already uses, plus 512 MB or 25 %).
    Before setting it, it loads PyTorch and CUDA if it uses them: importing PyTorch takes
    about 630 MB.
  - **CLI:** it sets the total on itself before grounding. If it runs out while grounding,
    it ends with exit code 4.
  - **As a library**, the calling process is not limited: grounding and a
    single-configuration search only have the soft limit.

  `RLIMIT_DATA` is used rather than `RLIMIT_AS` because CUDA reserves tens of GB of
  virtual memory that it does not use. It costs one system call per process. The GPU
  member solves slitherlink p02 just the same with the limit (9 s), and with a share of
  300 MB, less than CUDA takes (1 GB), it stops at the first check without errors.
- Exit codes:
  - 0: plan found;
  - 2: invalid input;
  - 3: proven that there is no plan;
  - 4: no plan found: a limit was reached, or the search cannot prove there is none (IW, or
    a temporal task in which repeated states with another timeline were dropped; see
    [Temporal engine](#temporal-engine)).

## Library usage

```python
from pyplanner.frontend import read_domain, read_problem
from pyplanner.planner import solve
from pyplanner.search import portfolio_for_cores

domain = read_domain("domain.pddl")
problem = read_problem("p01.pddl", domain)
result = solve(domain, problem, portfolio_for_cores(2), time_limit=60)
if result.plan is not None:
    print(result.plan.to_ipc())  # or result.plan.to_dict() for the JSON
```
