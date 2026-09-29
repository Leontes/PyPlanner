# Dominios de evaluación

Benchmarks para evaluar el planificador, organizados en **tiers**. Cada tier corresponde a
una iteración del [roadmap](../docs/ROADMAP.md). Al terminar la iteración *N* tienen que
resolverse **todos** los problemas de los tiers 1…*N*. Así, la funcionalidad nueva nunca
puede romper la anterior.

| Tier | Iteración | Dominio | Requisitos declarados | Origen |
|------|-----------|---------|-----------------------|--------|
| 1 | STRIPS | `blocksworld` | `:strips` | IPC 2000 (blocks-strips-untyped) |
| 1 | STRIPS | `gripper` | ninguno (STRIPS implícito) | IPC 1998 (gripper-round-1-strips) |
| 2 | PDDL 1.2 (ADL) | `rubiks-cube` | `:adl` | IPC 2023 classical (sat) |
| 2 | PDDL 1.2 (ADL) | `slitherlink` | `:typing :adl` | IPC 2023 classical (sat) |
| 3 | PDDL 2.1 numérico | `numeric/counters` | `:strips :typing :equality :adl :numeric-fluents` (*) | IPC 2023 numeric |
| 3 | PDDL 2.1 numérico | `numeric/zenotravel` | `:typing :fluents` (*) | IPC 2023 numeric |
| 3 | PDDL 2.1 temporal | `temporal/driverlog` | `:typing :durative-actions :numeric-fluents` (*) | IPC 2002 (driverlog-time-simple-automatic) |
| 3 | PDDL 2.1 temporal | `temporal/zenotravel` | `:durative-actions :typing :numeric-fluents` (*) | IPC 2002 (zenotravel-time-simple-automatic) |
| 4 | PDDL 2.2 | `derived-predicates/psr-middle` | `:adl :derived-predicates` | IPC 2004 (psr-middle-derived-predicates-adl) |
| 4 | PDDL 2.2 | `timed-initial-literals/satellite` | `:strips :equality :typing :fluents :durative-actions :timed-initial-literals` | IPC 2004 (satellite-time-time-windows-strips) |
| 5 | PDDL 3 | `preferences/storage` | `:typing :adl :preferences :constraints` | IPC 2006 (storage-preferences-qualitative) |
| 5 | PDDL 3 | `constraints/trucks` | `:typing :adl :durative-actions :fluents :constraints` | IPC 2006 (trucks-time-constraints) |
| 6 | PDDL+ | `car` | `:typing :durative-actions :fluents :time :negative-preconditions :timed-initial-literals` | SMTPlan (car_nodrag) |
| 6 | PDDL+ | `generator` | `:fluents :durative-actions :duration-inequalities :adl :typing :time :continuous-effects` (*) | SMTPlan (generator_events) |

Cada dominio tiene `domain.pddl` y dos problemas, `p01.pddl` y `p02.pddl`. Son las dos primeras
instancias de la colección original, que suelen ser las más sencillas.

## Convenciones

- Estructura: `tierN-<nombre>/[<subtipo>/]<dominio>/{domain.pddl, pNN.pddl}`.
- Los ficheros se copian **sin modificar**. Si un dominio necesita retoques, se añade como
  dominio nuevo y se explica en esta tabla. La única excepción son las erratas en
  `:requirements` marcadas con (*), que se corrigen en el propio fichero.
- **Toda discordancia de requisitos se trata como una errata del fichero:** se añade a
  mano el requisito que falta, y el modo estricto no hace excepciones.
- **Modo estricto:** el planificador exige que toda construcción que se use tenga su
  requisito declarado. Si no se declara `:requirements`, se supone `:strips`.
- Para añadir problemas basta con copiar más `pNN.pddl`; la batería de tests los
  descubre sola.

## Procedencia

| Colección | Repositorio | Commit |
|-----------|-------------|--------|
| IPC 1998–2006 | [potassco/pddl-instances](https://github.com/potassco/pddl-instances) | `cf19edf` |
| IPC 2023 classical | [ipc2023-classical/ipc2023-dataset](https://github.com/ipc2023-classical/ipc2023-dataset) | `1e29a0e` |
| IPC 2023 numeric | [ipc2023-numeric/ipc2023-dataset](https://github.com/ipc2023-numeric/ipc2023-dataset) | `0799944` |
| PDDL+ | [KCL-Planning/SMTPlan](https://github.com/KCL-Planning/SMTPlan) (`benchmarks/`) | `b085bd7` |

Notas:
- Las IPC 2023 no tienen track temporal, de PDDL 3 ni de PDDL+. Por eso los tiers 1,
  3 (temporal), 4, 5 y 6 salen del archivo histórico de las IPC y de los benchmarks de
  SMTPlan. Los de PDDL 3 son de las IPC 2006: storage combina preferencias con
  restricciones sin tiempo (`sometime`, `at-most-once`) y una métrica con `is-violated`;
  trucks, restricciones con tiempo (`within`) sobre acciones durativas.
- Los benchmarks PDDL+ de ENHSP se han descartado porque usan `:constraint`, una
  extensión propia de ese planificador que no forma parte del estándar.
- (*) En los dominios numéricos de las IPC 2023 la línea `:requirements` venía comentada,
  aunque usan tipos y fluentes numéricos. Se trata como una errata y se ha descomentado
  sin tocar el resto del dominio. En `counters` la línea original tampoco incluía los
  fluentes numéricos, así que además se le ha añadido `:numeric-fluents`.
- (*) Los problemas temporales de driverlog y zenotravel (IPC 2002) tienen
  `(:metric minimize (total-time))`, y la métrica necesita `:numeric-fluents`, que sus
  dominios no declaraban. Se ha añadido a la línea `:requirements` del dominio.
- (*) `generator` usa un efecto continuo (`#t`) en una acción durativa, pero no declaraba
  `:continuous-effects` (solo `:time`, que es para procesos y eventos). Se ha añadido.
- Varios dominios usan funcionalidad de tiers anteriores: por ejemplo, `car` usa
  `:timed-initial-literals` y trucks necesita el motor temporal. Es lo esperado, porque
  cada tier incluye a los anteriores.
