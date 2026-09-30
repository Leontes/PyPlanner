# Motor de búsqueda (iteraciones 1 a 3b)

Cómo está construido el motor y cómo se usa. Las decisiones de fondo (portfolio, lotes,
GPU, técnicas descartadas) están en el [roadmap](ROADMAP.md#motor-de-búsqueda-diseño-y-repertorio).

## Flujo

```
Domain + Problem (AST) ──ground()──► GroundTask ──search() / run_portfolio()──► SearchResult ──► Plan
```

| Paso | Módulo | Qué hace |
|------|--------|----------|
| Grounding | `pyplanner.grounding` | Instancia las acciones alcanzables en la relajación y elimina los hechos estáticos |
| Heurísticas | `pyplanner.heuristics` | Evalúan lotes de estados; implementación de referencia y en PyTorch |
| Búsqueda | `pyplanner.search` | GBFS / A* / WA*, inmediata o perezosa, con acciones útiles y lotes |
| Portfolio | `pyplanner.portfolio` | Varias configuraciones en procesos paralelos; gana la primera |
| Plan | `pyplanner.plan` | Salida IPC o JSON y validación interna (simulación) |
| Fachada | `pyplanner.planner.solve` | Encadena todo lo anterior |

## Normalización ADL (iteración 2)

El grounder (`grounding/grounder.py`) normaliza cada instancia de acción con
`grounding/normalize.py`, de modo que la búsqueda solo ve estructuras conjuntivas:

- **Precondiciones y objetivos → DNF de literales.** Los cuantificadores se expanden sobre
  los objetos de su tipo (`exists` → disyunción, `forall` → conjunción), las negaciones se
  empujan hasta los átomos e `imply` se trata como `(or (not a) b)`. `=` y los hechos
  estáticos se evalúan en el momento; las cláusulas contradictorias y las subsumidas se
  eliminan. Si la DNF supera 10 000 cláusulas se aborta con un error claro (esas fórmulas
  necesitarán axiomas, iteración 4).
- **Precondición disyuntiva:** una `GroundAction` por cláusula, con el mismo nombre y
  argumentos, así que en el plan se ven igual.
- **Efectos → efectos incondicionales + efectos condicionales** (`forall` se expande;
  cada cláusula de la condición de un `when` da un efecto condicional). Si la precondición
  ya garantiza la condición, el efecto pasa a ser incondicional; si la contradice, se
  descarta.
- **Objetivo:** disyunción de cláusulas. Si ninguna es alcanzable, el objetivo queda
  vacío y la tarea se reconoce como irresoluble sin buscar.
- **Alcanzabilidad relajada:** los literales negativos se suponen alcanzables, y un
  efecto condicional cuenta si alguna cláusula de su condición es alcanzable.

## Tarea instanciada y estados

- Un **estado** es un `State(facts, values)`: `facts` es un `int` usado como bitset sobre
  `GroundTask.facts`, y `values` la tupla de valores de las variables numéricas
  (`GroundTask.variables`; vacía en tareas proposicionales, `None` si el valor está
  indefinido). Es una `NamedTuple`: se compara y se usa como clave de diccionario igual
  que antes, con un coste medido de unos 0,3 µs por sucesor. Una acción es aplicable si
  `facts & pre == pre`, `facts & neg == 0` (precondiciones negativas) y se cumplen sus
  condiciones numéricas.
- **Efectos condicionales:** las condiciones se evalúan sobre el estado **anterior**;
  después se borra y se añade (un hecho borrado y añadido a la vez queda cierto). Los
  efectos condicionales de cada acción se indexan por su primer hecho de condición, así
  que al aplicarla solo se revisan los que pueden dispararse. En rubiks-cube cada giro
  tiene cientos de efectos condicionales y solo unos pocos se disparan.
- Los hechos **estáticos** no están en el estado.
- **Generación de sucesores** (`search/successors.py`), con un **orden canónico**:
  1. primero las acciones sin precondiciones positivas;
  2. después, las de cada hecho cierto en orden creciente cuya primera precondición es
     ese hecho;
  3. dentro de cada grupo, por índice.

  No hace falta ordenar, y hay dos implementaciones con el mismo orden: la de referencia
  (`GroundTask.applicable`) y una compilada con Numba, que comprueba las precondiciones
  con palabras de 64 bits. La búsqueda usa la compilada salvo con `--device python`; como
  el orden es el mismo, expande los mismos nodos con las dos.
- Para generar sucesores, las acciones se agrupan por su primera precondición: solo se
  miran las acciones cuya primera precondición es cierta en el estado.
- Hechos y acciones tienen un orden determinista, así que dos ejecuciones iguales dan el
  mismo plan.

## Motor numérico (iteración 3a)

- **Grounding.**
  - Las funciones que ninguna acción modifica son **estáticas**: se sustituyen por su
    valor al instanciar. Una comparación con un valor estático indefinido es falsa.
  - Solo son **variables** del estado las funciones **relevantes**: las que se leen en
    alguna condición o en el objetivo, o en el valor de un efecto sobre otra variable
    relevante (punto fijo). Las que solo aparecen en la métrica, como `total-cost` o
    `total-fuel-used`, no están en el estado: si estuvieran, dos estados iguales con
    distinto coste acumulado serían distintos y la detección de duplicados dejaría de
    funcionar.
  - Las condiciones numéricas se normalizan a `expresión ⋈ 0` con `⋈` ∈ {`>=`, `>`, `=`};
    `(not (= a b))` da dos cláusulas con `>`. Las expresiones lineales se guardan también
    como forma lineal (términos ordenados y constante), que es la que usan los núcleos
    compilados.
- **Semántica de los efectos** (`GroundAction.apply`):
  - los efectos numéricos se evalúan con los valores del estado anterior;
  - varios `increase`/`decrease` sobre la misma variable se suman;
  - cualquier otro conflicto, una división por cero o un valor indefinido hacen que la
    acción no sea aplicable (`apply` devuelve `None` y la búsqueda descarta el sucesor);
  - un valor indefinido hace falsa cualquier comparación.
- **La métrica como coste.** Si la métrica minimiza una expresión lineal de `total-time`
  y de funciones que las acciones solo cambian con `increase`/`decrease` de valor
  constante (tras sustituir los estáticos), cada acción contribuye con una cantidad fija:
  ese es su **coste**, y minimizar el coste del plan minimiza la métrica. Es el caso de
  `:action-costs` y de métricas como la de zenotravel (`total-time` cuenta como la
  longitud del plan). Si no (métricas que se maximizan, no lineales o con costes
  negativos), se busca con coste unitario y el grounder deja una **nota**, que la CLI
  muestra.
- **Salida.** El plan lleva el coste y, cuando los costes representan la métrica, su
  valor (`; metric = …` en IPC, `"metric"` en JSON), que coincide con el `Final value` de
  VAL. Sin métrica utilizable, el comentario IPC dice `(unit cost)`.
- **Heurística numérica: *h^hbd*** (Scala, Haslum, Thiébaux y Ramírez, 2020). Cada
  condición numérica distinta es un nodo más de la relajación. Una acción relajada cuyo
  efecto aumenta una condición lineal en una cantidad constante Δ la logra repitiéndose
  ⌈déficit/Δ⌉ veces (para `=`, avanzando hacia 0), y la arista cuesta repeticiones × coste.
  Las repeticiones dependen del estado. Los efectos que no son incrementos constantes y
  las condiciones no lineales son **optimistas** (una repetición basta), así que la
  heurística nunca declara un callejón sin salida falso. Los efectos que empeoran una
  condición se ignoran, como los borrados. h_FF cuenta cada operador por su coste por el
  máximo número de repeticiones que necesita.

## Motor temporal (iteración 3b)

El motor temporal sigue a POPF (Coles, Coles, Fox y Long, 2010): búsqueda hacia delante
sobre acciones de inicio y fin, con una red temporal simple (STN) que decide cuándo ocurre
cada paso. El código está en `grounding/temporal.py`.

- **Compilación.** Cada acción durativa `a` da dos acciones clásicas y un hecho
  `running@a`:
  - `a@start`: condiciones `at start` y `¬running@a` (una instancia no se solapa consigo
    misma); efectos `at start` y `running@a`;
  - `a@end`: condiciones `at end` y `over all`, y `running@a`; efectos `at end` y
    `¬running@a`.

  El objetivo exige además que no quede ninguna acción en curso. Todo lo demás se
  reutiliza tal cual: grounder, normalización ADL y numérica, sucesores compilados y
  *h^hbd* (Python, Numba y PyTorch) sobre la tarea de inicios y fines. Las acciones
  instantáneas del dominio conviven con las durativas.
- **Estado temporal.** Un `State` (hechos y valores) con una línea temporal adjunta. Se
  compara solo por la parte lógica (memoización al estilo de POPF), así que la detección
  de repetidos funciona como en la planificación clásica. Es incompleta en casos raros:
  un estado repetido con otra planificación en el tiempo se descarta.
- **Línea temporal bajo demanda.** Al generar un sucesor se comprueba su línea temporal,
  pero no se guarda: el estado solo recuerda de dónde viene y la recalcula al expandirse.
  La búsqueda genera muchos más estados de los que expande, y cada línea temporal crece
  con la longitud del plan: guardarlas todas costaba de 10 a 14 KB por estado en las
  tareas grandes (3,5 GB en zenotravel 17 con solo 3000 expansiones), y así, de 2 a 4.
- **Órdenes.** Para cada hecho y variable se guarda el último paso que lo cambió y los que
  lo han leído desde que tiene su valor. Un paso nuevo va, con una separación ε, después
  del último cambio de lo que lee o modifica y de los que leyeron el valor que cambia. Los
  pasos independientes no se ordenan: la red temporal los hace simultáneos, y el plan sale
  en paralelo aunque la búsqueda los eligiera en secuencia.
- **Red temporal.** Restricciones `t_j ≥ t_i + w`: las órdenes (ε) y la duración de cada
  acción (mínima y máxima). Se mantienen los tiempos más tempranos con propagación
  incremental. Solo una duración máxima puede crear un ciclo positivo, y pasa
  necesariamente por el paso nuevo, así que se detecta al añadirlo. Los tiempos más
  tempranos son la planificación del plan.
- **Duraciones.** Fijas (`=`) o con desigualdades; con funciones, se evalúan al empezar (o
  al terminar, con `(at end ...)`). `?duration` en los efectos se sustituye por la duración
  cuando es fija y no depende de funciones que cambian; si no, la decide el programa
  lineal (ver abajo).
- **Invariantes.** Las condiciones `over all` se comprueban tras cada paso en todas las
  acciones en curso. Sus hechos y variables cuentan como leídos por el inicio y por el
  fin, así que los pasos que los cambian quedan fuera del intervalo.
- **Efectos condicionales** con la condición al empezar y el efecto al terminar: un hecho
  auxiliar recuerda, al empezar, si la condición se cumplía. Con condiciones `over all`, el
  hecho es un **vigilante**: se activa al empezar si se cumplen las condiciones y el motor
  lo apaga en cuanto la condición `over all` deja de cumplirse durante la acción.
- **ε = 0,01.** VAL agrupa como simultáneos los sucesos más cercanos que su tolerancia (por
  defecto 0,01), así que con ε = 0,001 (el valor de POPF) solo valida con
  `Validate -t 0.001`. Con 0,01 el plan vale con la configuración por defecto, a cambio de
  0,01 unidades por dependencia.
- **Métrica.** Solo `(:metric minimize (total-time))`: su valor es la duración total. La
  búsqueda cuenta pasos (coste unitario); no optimiza la duración.
- **Salida.** `t: (acción args) [d]` en IPC, con `; makespan = …`; en JSON, `time` y
  `duration` en cada paso y `makespan` en el plan.
- **Programación lineal** (al estilo de COLIN; `grounding/temporal_lp.py`, con HiGHS a
  través de SciPy). Las **variables lineales** son las que cambian con efectos continuos
  (`(increase f (* #t r))`) y las que reciben valores que dependen de `?duration` en
  acciones con duración no fija. Sus condiciones y efectos salen de las acciones de la
  búsqueda y se deciden con un programa lineal por estado:
  - incógnitas: el tiempo de cada paso y el valor de cada variable lineal tras cada cambio;
    entre dos cambios evoluciona según la suma de los ritmos activos;
  - restricciones: la red temporal, las condiciones de cada paso, los efectos,
    `?duration = t_fin − t_inicio` y las invariantes en cada extremo de tramo (una
    condición lineal sobre valores que cambian linealmente se cumple en un tramo si se
    cumple en sus extremos);
  - objetivo: minimizar la suma de los tiempos. Solo se resuelve cuando el paso toca
    variables lineales o retrasa pasos anteriores.

  El estado guarda el valor de cada variable lineal tras su último cambio, que es lo que
  ven la heurística y el objetivo. Las condiciones del objetivo sobre variables lineales se
  comprueban añadiéndolas al programa (se pueden cumplir con otra planificación, p. ej.
  alargando una acción), y la planificación final las incluye. La heurística usa una
  versión de la tarea con los efectos continuos integrados al terminar (ritmo × duración)
  y las condiciones sobre variables lineales. Con variables que cambian de forma continua
  el espacio de estados es infinito: la búsqueda no puede demostrar que no hay plan.
  También: efectos continuos dentro de `forall` (se expanden al instanciar) y dentro de
  `when` con condiciones `at start` (el ritmo se decide al empezar); `?duration` no fijo en
  efectos `at start` (la duración es una incógnita, entre sus límites, enlazada con
  `t_fin − t_inicio` al terminar); duraciones que dependen de variables lineales (sus
  límites son restricciones del programa); y efectos condicionales sobre variables
  lineales con condiciones clásicas. Un dominio de certificación hecho a mano
  (`greenhouse`, en `tests/test_temporal_engine.py`) las reúne todas, validado con VAL: no
  hay benchmarks públicos que las usen.
- **Límites de la programación lineal** (error claro): condiciones de efectos
  condicionales sobre variables lineales e invariantes disyuntivas sobre ellas (el efecto
  dependería de qué solución se elija: haría falta programación lineal entera),
  invariantes y condiciones no lineales, y `?duration` no fijo en el ritmo de un efecto
  continuo (ritmo × tiempo no es lineal).

## Heurísticas

| Nombre | Admisible | Acciones útiles | Numba | PyTorch |
|--------|-----------|-----------------|-------|---------|
| `blind` | sí | no | no | no |
| `goalcount` | no | no | no | no |
| `hmax` | sí | no | sí | sí |
| `hadd` | no | no | sí | sí |
| `ff` | no | sí | sí | sí (costes y plan relajado en tensores) |

**Dispositivos** (`SearchConfig.device`, `--device`):

| Dispositivo | Qué es | Cuándo usarlo |
|-------------|--------|---------------|
| `auto` (por defecto) | `numba` si está instalado; si no, `python` | siempre, salvo que se quiera forzar otro |
| `python` | implementación de referencia en Python puro | tests y máquinas sin Numba |
| `numba` | la referencia compilada con Numba (`pip install -e ".[fast]"`); 12–19 veces más rápida | es la opción rápida para búsquedas normales |
| `cpu`, `cuda` | PyTorch por lotes (`pip install -e ".[gpu]"`) | solo con lotes grandes; en GPU, 4 veces más rápido que Python con lotes de 256 |

Los tres dan **exactamente los mismos valores y acciones útiles** que la referencia; los
tests lo comprueban en los tiers 1 a 3 (numérico), en tareas hechas a mano y, durante el
desarrollo, en tareas aleatorias con muchos empates y costes 0.

- **Interfaz:** `evaluate_batch(states) -> list[Evaluation]`. Cada `Evaluation` lleva el
  valor (`inf` si es un callejón sin salida) y las acciones útiles.
- **Tarea relajada** (`RelaxedTask`, compartida por las dos implementaciones):
  - **Hechos complementarios:** cada hecho que aparece negado tiene un hecho "no-p",
    cierto si p es falso y que hacen cierto las acciones que borran p. Así las negaciones
    pasan a ser positivas **solo dentro de la relajación**, como en FF.
  - **Acciones relajadas:** una por los efectos incondicionales de cada operador y otra
    por cada efecto condicional (con la condición añadida a la precondición), como en
    Fast Downward.
  - **Condiciones numéricas:** un nodo por condición, con aristas cuyas repeticiones
    dependen del estado (*h^hbd*, ver arriba).
  - **Objetivo:** el valor es el mínimo entre cláusulas.
- **Referencia en Python:** Dijkstra generalizado sobre la relajación sin borrados. Se
  detiene al extraer todos los hechos del objetivo, y anota el mejor soportador durante el
  propio recorrido, así que h_FF no tiene que buscarlo después.
- **Numba** (`heuristics/compiled.py`): el mismo Dijkstra y la misma extracción,
  compilados, sobre la tarea relajada en arrays CSR. El lote entero se evalúa en una
  llamada y los estados se decodifican dentro del código compilado. La primera ejecución
  tras instalar compila los núcleos (unos segundos) y los guarda en la caché de disco.
- **PyTorch:** punto fijo sobre todo el lote con `index_add_` y `scatter_reduce`, usando
  listas de aristas acción-hecho y no matrices densas. Calcula en `float64`: con costes
  enteros es exacto; con costes no enteros, el orden de las sumas puede cambiar el último
  bit de algún valor.
- **Desempate (coste, nivel).** Con costes 0 (`:action-costs` los admite), el coste solo
  no basta para elegir soportadores sin ciclos. Los hechos se ordenan por (coste, nivel):
  el nivel de una acción es 1 + el máximo nivel de sus precondiciones, y el de un hecho
  cierto es 0. El mejor soportador es la acción relajada de **menor índice** entre las que
  dan al hecho su (coste, nivel) mínimo. No depende del orden de extracción, así que
  PyTorch lo reproduce sin Dijkstra: el nivel de un hecho es la última iteración del punto
  fijo en que baja su coste (demostración en `TensorRelaxation._fact_costs`).
- **h_FF:** se usa la cláusula del objetivo de menor valor (la primera si hay empate). El
  valor cuenta cada **operador** una sola vez (por su máximo de repeticiones), aunque el
  plan relajado use varios de sus efectos, y se suma en orden de operador en las tres
  implementaciones.
- **h_FF en tensores:** la extracción del plan relajado también se hace para todo el lote
  a la vez, por capas hacia atrás desde los objetivos (ver `heuristics/tensor.py`).

## Configuración de búsqueda

`SearchConfig` define por completo una búsqueda y se valida al crearla. Campos:

- `algorithm`: `gbfs`, `astar`, `wastar` (con `weight`) o `iw` (ver "Búsqueda por
  novedad");
- `heuristic`;
- `preferred_operators`: dos colas alternas, la normal y la de acciones útiles;
- `lazy`: evaluación perezosa; no se puede usar con A*;
- `tie_breaking`: `fifo`, `lifo` o `random`, con `seed`;
- `exploration`: `"type-based"` añade una cola de **exploración por tipos** (Xie et al.,
  2014) que se alterna con las demás: saca un tipo (h, g) al azar y un nodo al azar de
  ese tipo. Evita que la búsqueda voraz se atasque en mesetas y zonas con muchos
  callejones sin salida;
- `novelty`: anchura de la novedad (0, 1 o 2; ver "Búsqueda por novedad");
- `batch_size`;
- `device`: `auto` (por defecto), `python`, `numba`, `cpu` o `cuda` (ver arriba);
- `max_time`, `max_expansions` y `max_memory` (MB de memoria residente del proceso; se
  consulta cada 32 iteraciones leyendo `/proc/self/statm`).

Garantías:

- **Óptimo:** A* con `blind` o `hmax` devuelve planes óptimos, también con lotes. En la
  búsqueda inmediata, un objetivo solo se acepta si es el primero que sale de la cola en
  su lote.
- **Completo:** todas las configuraciones lo son salvo IW(k). Los callejones sin salida que
  se descartan son estados desde los que el objetivo es inalcanzable incluso en la
  relajación. IW poda estados por su novedad: si agota la cola habiendo podado alguno,
  termina con `failed` (sin plan, pero sin demostrar que no lo haya), nunca con
  `unsolvable`.
- **Cancelable:** `search(..., stop=evento)` consulta el evento periódicamente.

Configuraciones predefinidas (`pyplanner configs`):

| Nombre | Configuración | Notas |
|--------|---------------|-------|
| `gbfs-ff-pref-tb` | GBFS + h_FF + acciones útiles + exploración por tipos | la de 1 núcleo hasta la iteración 3b |
| `gbfs-ff-pref` | GBFS + h_FF + acciones útiles | |
| `lazy-ff-pref` | GBFS perezosa + h_FF + acciones útiles | evalúa muchos menos estados |
| `lazy-ff-pref-tb` | la anterior + exploración por tipos (semilla 1) | |
| `gbfs-ff` | GBFS + h_FF | |
| `gbfs-add` | GBFS + h_add | |
| `wastar-add-3` | WA* (w = 3) + h_add | |
| `gbfs-goalcount-rnd` | GBFS + objetivos pendientes, desempate aleatorio | muy barata; aporta diversidad |
| `bfws-ff-pref` | BFWS: GBFS con prioridad (novedad, h_FF) + acciones útiles | la de 1 núcleo: la mejor en las pruebas de la 3b |
| `bfws-ff` | BFWS con h_FF | |
| `bfws-goalcount` | BFWS con objetivos pendientes | sin heurística de relajación |
| `iw1`, `iw2` | IW(1), IW(2) | incompletas; solo para problemas de anchura baja |
| `astar-hmax` | A* + h_max | óptima |
| `astar-blind` | A* ciega | óptima; coste uniforme |

## Búsqueda por novedad

Lipovetzky y Geffner (2012, 2017); código en `search/novelty.py` y `search/engine.py`.

- **Novedad** de un estado respecto a los ya vistos: 1 si hace cierto un átomo que ningún
  estado anterior tenía, 2 si forma un par de átomos nunca visto junto, 3 en otro caso.
  Los átomos son los hechos ciertos; la falsedad de los hechos que aparecen negados en
  alguna precondición, condición u objetivo (si el objetivo pide una luz apagada, apagarla
  es algo nuevo: sin esto, IW(2) no resolvía `lights`); y, para las variables numéricas,
  los pares (variable, valor). En las tareas temporales se excluyen los hechos internos
  `running@a`: con miles de acciones durativas, hacían la novedad cara y poco
  discriminante (de 33 a 29 instancias resueltas).
- **Bitsets:** los átomos de un estado forman un entero. Para los pares, cada átomo guarda
  el bitset de los átomos vistos junto a él, así que medir y registrar un estado cuesta
  O(átomos del estado) operaciones sobre enteros, en C.
- **BFWS** (`algorithm="gbfs"` con `novelty`): la prioridad es (novedad, h), con la
  novedad medida entre los estados del mismo valor h. Entre estados igual de buenos para
  la heurística, se expanden antes los que exploran algo nuevo. Es completa: no poda.
- **IW(k)** (`algorithm="iw"`): búsqueda en anchura que descarta los estados de novedad
  mayor que k. Resuelve de inmediato los problemas de anchura ≤ k (los que se pueden
  resolver pasando solo por estados nuevos), pero es incompleta.
- Solo con evaluación inmediata: la novedad de un estado se mide al generarlo.

## Portfolio

**Qué búsquedas se lanzan.** Depende del número de núcleos (`-j N`). Hay un portfolio
predefinido para 1, 2, 4, 5 y 6 núcleos (`PORTFOLIOS` en `search/config.py`), y cada uno
contiene al anterior:

| Núcleos | Búsquedas | Qué se añade |
|---------|-----------|--------------|
| 1 | `bfws-ff-pref` | BFWS, la configuración individual más robusta y rápida; se ejecuta en el propio proceso |
| 2 | + `lazy-ff-pref` | la más rápida en tareas grandes y fáciles |
| 4 | + `lazy-ff-pref-tb`, `gbfs-goalcount-rnd` | la perezosa con exploración, y la más barata y aleatoria |
| 5 | + `gbfs-ff-pref-tb` | GBFS con exploración por tipos, la más rápida en slitherlink |
| 6 | + `wastar-add-3` | WA*, que tiende a planes más cortos: más diversidad con CPU de sobra |

- Con otro número de núcleos se usa el portfolio más grande que quepa: con 3, el de 2;
  con más de 6, el de 6. Nunca hay más búsquedas que núcleos.
- **Por defecto se usan 6 núcleos**, para la máxima robustez, pero limitados a los que
  tenga la máquina (`os.sched_getaffinity`). En un sistema mononúcleo se usa el
  portfolio de 1 sin necesidad de indicarlo.
- Con `-c` se eligen las configuraciones a mano. Si se piden más configuraciones que
  núcleos, es un error.

**Cómo se ejecutan.**

- Un proceso por configuración; gana el primer plan y el resto se cancelan.
- Se usa `fork` cuando es seguro: sin CUDA, sin PyTorch cargado en el padre y con un
  solo hilo. En cualquier otro caso se usa `spawn`.
- Si un miembro falla, informa con estado `ERROR` y un mensaje, pero no tumba el
  portfolio.
- **Precompilación de Numba:** antes de lanzar los hijos, el padre compila (o carga de la
  caché) los núcleos de Numba si algún miembro los va a usar. Sin esto, un hijo cancelado
  a mitad de compilar dejaba la caché incompleta, y cada ejecución pagaba la compilación
  y la espera del cierre. Con `fork`, los hijos heredan los núcleos ya compilados.

## Uso desde la terminal

```bash
pyplanner configs                                   # configuraciones y portfolios
pyplanner solve dominio.pddl problema.pddl          # portfolio de 6 núcleos (o menos)
pyplanner solve -j 1 dominio.pddl problema.pddl     # un solo núcleo
pyplanner solve -j 2 -t 60 dominio.pddl p.pddl      # 2 núcleos, 60 s como máximo
pyplanner solve -m 4000 dominio.pddl p.pddl         # 4000 MB en total para las búsquedas
pyplanner solve -c astar-hmax dominio.pddl p.pddl   # plan óptimo
pyplanner solve -c gbfs-add --device cuda --batch-size 64 dominio.pddl p.pddl
pyplanner solve -f json -o plan.json dominio.pddl p.pddl
```

- El plan sale por la salida estándar y las estadísticas por la de error; `-q` las
  suprime.
- **Límite de memoria** (`-m`, o `memory_limit` en `solve`): se reparte a partes iguales
  entre los miembros del portfolio. Por defecto es el 80% de la memoria disponible, así
  que una búsqueda larga nunca agota la de la máquina (una búsqueda voraz puede generar
  cientos de miles de estados por segundo). Al llegar al límite, la búsqueda termina con
  `limit_reached` y lo explica en el detalle.
- Códigos de salida:
  - 0: plan encontrado;
  - 2: entrada no válida;
  - 3: se ha demostrado que no hay plan;
  - 4: no hay plan dentro de los límites.

## Uso como librería

```python
from pyplanner.frontend import read_domain, read_problem
from pyplanner.planner import solve
from pyplanner.search import portfolio_for_cores

domain = read_domain("domain.pddl")
problem = read_problem("p01.pddl", domain)
result = solve(domain, problem, portfolio_for_cores(2), time_limit=60)
if result.plan is not None:
    print(result.plan.to_ipc())  # o result.plan.to_dict() para el JSON
```

## Rendimiento medido

Medido en la máquina de desarrollo (GTX 970, 12 núcleos) con gripper de las IPC 1998,
`instance-20`: 42 bolas, 172 hechos y 340 acciones instanciadas.

**Coste de evaluar un estado** (1024 estados de paseos aleatorios):

| Heurística | Python | PyTorch CPU (lote 1024) | PyTorch GPU (lote 64) | PyTorch GPU (lote 1024) |
|------------|--------|-------------------------|-----------------------|-------------------------|
| h_add | 164 µs | 15 µs | 21 µs | **5.7 µs** |
| h_FF | 269 µs | 28 µs | 51 µs | **9.7 µs** |

**Búsqueda completa** (tiempo total del proceso):

| Configuración | Plan | Tiempo |
|---------------|------|--------|
| `gbfs-ff-pref` | 165 | 8.0 s |
| `gbfs-ff-pref`, CPU, lote 64 | 125 | 10.9 s |
| `gbfs-ff-pref`, GPU, lote 64 | 125 | 7.6 s |
| `lazy-ff-pref` | 165 | 0.47 s |
| `gbfs-add` | 167 | 1.1 s |
| `gbfs-add`, CPU, lote 64 | 163 | 5.0 s |
| `gbfs-add`, GPU, lote 64 | 163 | 4.1 s |
| portfolio de 5 núcleos | 129 (ganó `gbfs-goalcount-rnd`) | 0.52 s |

Conclusiones:

- **Evaluar por lotes con tensores sale muy a cuenta:** con lotes grandes, h_add es 11
  veces más rápida en CPU y 29 en GPU, y h_FF (con la extracción del plan relajado también
  en tensores) 10 y 28 veces.
- **Los lotes no aceleran por sí solos la búsqueda completa en tareas de este tamaño:**
  - importar PyTorch e inicializar CUDA cuesta unos 2 s;
  - con lotes, la GBFS expande *k* nodos antes de volver a mirar la heurística, así que
    se guía peor y expande más.

  Compensan cuando la evaluación domina el tiempo total, es decir, con tareas grandes o
  heurísticas caras. Por eso los lotes son una opción y no el valor por defecto.
- **El portfolio aporta robustez de verdad:** una configuración muy barata ganó a todas
  las demás, y además con un plan mejor. Es la diversidad de la que hablábamos.

### Iteración 2: slitherlink

slitherlink p02 (tablero de 4×6, 306 hechos, 1920 acciones con 2400 precondiciones
negativas) es el caso más duro de los tiers 1 y 2. Lecciones:

- **La heurística domina el tiempo:** más del 90%, y más de la mitad de los estados
  evaluados son callejones sin salida. Dos cambios en el Dijkstra de referencia duplicaron
  el ritmo de expansión:
  - precalcular el número de precondiciones de cada acción;
  - anotar el mejor soportador durante el recorrido, en lugar de buscarlo después.
- **La búsqueda voraz pura se atasca.** `gbfs-ff-pref` necesitaba unas 49 000 expansiones
  (unos 80 s al ritmo medido). Con lotes (*k-best-first search*) el número de expansiones
  variaba entre 1 400 y 82 000 según el tamaño de lote y la instancia: suerte con el
  orden, no una mejora fiable.
- **La exploración por tipos sí es robusta:** con 4 semillas resuelve p02 siempre, en
  17–24 s y con 9 000–14 000 expansiones. Por eso es la base de los portfolios desde la
  iteración 2.

### Iteración 2: heurísticas compiladas con Numba

Medido sobre 3000 estados reales de una búsqueda en slitherlink p02 (45% de callejones sin
salida; los hermanos de cada expansión dan lotes de unos 2,8 estados):

| Opción | µs/estado (h_FF) | Frente a Python |
|--------|------------------|-----------------|
| Referencia en Python | ~600 | 1× |
| Comprobación de alcanzabilidad previa + h_FF solo en estados vivos | 613 | 1,0× |
| PyTorch CPU / GPU, lotes de 3 | 892 / 1524 | 0,7× / 0,4× |
| PyTorch GPU, lotes de 256 | 142 | 4,3× |
| **Numba** | **~42** | **~19×** |

- **La comprobación previa no gana nada.** Los callejones sin salida son *más baratos* que
  los estados vivos (unos 490 µs frente a 760 µs), así que lo que ahorra en unos lo
  pierde en otros.
- **PyTorch no sirve con los lotes pequeños de la búsqueda:** domina el coste fijo de
  cada llamada.
- **Numba es 12–19 veces más rápido** según el dominio (gripper, rubiks-cube,
  slitherlink), con resultados idénticos a la referencia.

Búsqueda completa en slitherlink p02 (CLI, tiempo total del proceso):

| | Python | Numba (`auto`) |
|--|--------|----------------|
| `-j 1` | 22,3 s | **3,1 s** |
| `-j 5` | 23,6 s | **3,6 s** |

El test de tiers completo (tiers 1 y 2, 4 portfolios, validación con VAL) pasó de 97 s a
18 s. La búsqueda expande exactamente los mismos nodos: la ganancia es pura velocidad.
Ahora la heurística ya no domina el tiempo total; el siguiente cuello de botella es la
propia búsqueda en Python (generar sucesores y gestionar las colas).

### Iteración 2: NumPy y generación de sucesores

Se estudió sustituir las listas y los enteros de Python por arrays de NumPy. Medido en el
código del planificador:

| Operación | Python | NumPy |
|-----------|--------|-------|
| Acceso elemento a elemento en un bucle (patrón del Dijkstra de referencia) | 28 ns | 104 ns (3,6× más lento) |
| Aplicar una acción y calcular el hash del estado | 240 ns (entero bitset) | 1098 ns (array; 5× más lento) |
| Memoria por estado | 68 B | 418 B |
| Acciones aplicables (1920 acciones, vectorizado) | 70 µs | 80 µs |

**Conclusión: ahora mismo no nos beneficia usarlo de forma general.** NumPy gana cuando
un bucle entero corre en C, y eso ya lo aprovechamos dentro de los núcleos de Numba y de
PyTorch. En el código que va elemento a elemento es más lento. Además, el estado como
entero de Python es mejor que cualquier array: las operaciones de bits y el hash son una
sola operación en C sobre un solo objeto.

No es una decisión cerrada. NumPy es ya una dependencia y está disponible. Si en alguna
iteración aparece una parte concreta que se beneficie de operaciones vectoriales, se usará
ahí, de forma localizada; por ejemplo, el cálculo numérico de la iteración 3.

El perfil sí mostró que, con la heurística ya compilada, la **generación de sucesores**
se llevaba el 46% del tiempo: ordenar los candidatos de cada expansión era el 18%.
Resultados en slitherlink p02 (tiempo de búsqueda, sin arranque del proceso):

| Paso | Tiempo | µs/expansión |
|------|--------|--------------|
| Punto de partida (heurística con Numba) | 2,31 s | 210 |
| Orden canónico sin ordenar | 1,99 s | 177 |
| Generador de sucesores compilado con Numba | 1,53 s | 137 |
| Extracción de h_FF recorriendo solo el plan relajado y acciones útiles compactas | **1,44 s** | **129** |

Con la CLI (tiempo total del proceso), slitherlink p02 queda en **2,4 s con 1 núcleo y
2,5 s con 5**, frente a los 22–24 s de la versión con la heurística en Python. Ahora la
generación de sucesores es el 5% del tiempo y la heurística compilada unos 35 µs por
estado (unas tres cuartas partes del total). Bajar más exigiría cambios de algoritmo
(heurísticas incrementales, menos evaluaciones), no de implementación.

### Iteración 3a: motor numérico

**Llamar a funciones de Numba con muchos arrays es caro.** La primera versión compilada
del Dijkstra numérico era 4 veces más lenta que la de la iteración 2 (157 µs por estado
frente a 41 en slitherlink p02). La causa era una función auxiliar (`_offer`) que recibía
14 argumentos, 9 de ellos arrays, y se llamaba por cada efecto de cada acción disparada.
El generador de sucesores tenía el mismo problema con la comprobación de precondiciones.
Escribiendo ese código dentro del bucle (y unificando añadidos y aristas numéricas en una
sola lista CSR), el Dijkstra bajó a 50 µs y los sucesores volvieron a su coste anterior.
Las funciones pequeñas con pocos argumentos (las del montículo) no tienen ese problema.

**El desempate (coste, nivel) cuesta un 17% en el núcleo de Numba** (un nivel más por
hecho y por acción). Se probó un montículo indexado con *decrease-key* en lugar de
entradas obsoletas: es igual de rápido, pero su memoria está acotada por el número de
hechos, así que se queda. Las reservas de memoria por estado cuestan menos de 1 µs y no
merece la pena reutilizarlas.

**PyTorch:** calcular los niveles en un segundo punto fijo duplicaba el tiempo. Se
obtienen gratis del punto fijo de costes (la última iteración en que baja cada coste). Las
aristas de añadido ya no multiplican por 1: solo las numéricas llevan multiplicador.

Búsqueda en slitherlink p02 (`gbfs-ff-pref-tb`, tiempo de búsqueda, mismas expansiones):

| Dispositivo | Iteración 2 | Iteración 3a |
|-------------|-------------|--------------|
| Numba | 1,55 s | 1,83 s |
| PyTorch CPU, lote 64 | 6,9 s | 9,5 s |
| PyTorch GPU, lote 64 | 2,3 s | 2,9 s |

En PyTorch, el resto de la diferencia es `float64`, que mueve el doble de memoria en cada
`scatter` y `gather`: es el precio de la exactitud.

### Iteración 3b: motor temporal

Instancias de driverlog y zenotravel (IPC 2002, *time-simple*) hasta la 17, con
`gbfs-ff-pref-tb` y `lazy-ff-pref` (Numba, 60 s): se resuelven 47 de 52. Las instancias
del tier 3 tardan milisegundos.

- **La heurística domina el tiempo** (más del 80% en driverlog 15). Aplicar un paso
  (órdenes, red temporal e invariantes) cuesta unos 5 µs.
- **Se corrigió un derroche en h_FF compilada:** convertía a lista el array entero de
  acciones útiles (lote × hechos, unos 830 000 elementos por lote en driverlog 15) en
  lugar de solo la parte usada. Venía de la iteración 2; con tareas grandes costaba 0,7 s.
- **Se descartó exigir la invariante ya en el inicio** (cuando los efectos `at start` no
  la tocan). Evita generar inicios que la rompen, pero empeoró la búsqueda: 46 resueltos
  frente a 47 y un 20% más de tiempo total.

### Iteración 3b: búsqueda por novedad y portfolios

Tiers 1 a 3 y driverlog y zenotravel temporales hasta la 17: 34 instancias, 60 s y
2,5 GB por búsqueda. Número de instancias resueltas y tiempo total (las no resueltas
cuentan 60 s):

| Configuración | Resueltas | Tiempo |
|---------------|-----------|--------|
| `bfws-ff-pref` | 33 | 165 s |
| `gbfs-ff-pref-tb` | 33 | 249 s |
| `lazy-ff-pref` | 32 | 144 s |
| `lazy-ff-pref-tb` | 31 | 202 s |
| `wastar-add-3` | 28 | 451 s |
| `gbfs-goalcount-rnd` | 26 | 469 s |
| `bfws-ff` / `bfws-goalcount` | 30 / 26 | (medidas aparte) |
| `iw1` / `iw2` | 6 / 9 | incompletas: solo anchura baja |

- **BFWS con acciones útiles es la mejor configuración individual**, y sin acciones
  útiles pierde casi toda la ventaja. Ni la exploración por tipos ni la novedad 1 (más
  barata) la mejoran.
- **Portfolios:** para cada número de núcleos se eligió, entre estas configuraciones, la
  combinación con más instancias resueltas y menos tiempo (el del miembro más rápido de
  cada instancia). BFWS entra en todos: con 1 núcleo, de 249 s a 165 s; con 2, de 119 s a
  107 s; con 4 y 5, de 117 s a 101 s. WA* sale del de 5 (empataba con `gbfs-ff-pref-tb`,
  que es la más rápida en slitherlink).
- **Memoria.** La primera tanda, con 10 procesos, agotó la memoria de la máquina: una
  búsqueda llegó a 10 GB en menos de 3 minutos. De ahí la línea temporal bajo demanda y el
  límite de memoria (ver "Motor temporal" y "Uso desde la terminal").

