# Motor de búsqueda

[English](../en/search-engine.md) · **Español**

Cómo está construido el motor y cómo se usa.

## Flujo

```
Domain + Problem (AST) ──ground()──► GroundTask ──search() / run_portfolio()──► SearchResult ──► Plan
```

| Paso | Módulo | Qué hace |
|------|--------|----------|
| Grounding | `pyplanner.grounding` | Instancia las acciones alcanzables en la relajación y elimina los hechos estáticos |
| Heurísticas | `pyplanner.heuristics` | Evalúan lotes de estados; implementación de referencia y en PyTorch |
| Búsqueda | `pyplanner.search` | GBFS / A* / WA*, inmediata o perezosa, con acciones útiles y lotes |
| Portfolio | `pyplanner.portfolio` | Varias configuraciones en procesos paralelos; gana la primera (con preferencias, la mejor dentro del límite de tiempo) |
| Plan | `pyplanner.plan` | Salida IPC o JSON y validación interna (simulación) |
| Fachada | `pyplanner.planner.solve` | Encadena todo lo anterior |

## Normalización ADL

El grounder (`grounding/grounder.py`) normaliza cada instancia de acción con
`grounding/normalize.py`, de modo que la búsqueda solo ve estructuras conjuntivas:

- **Precondiciones y objetivos → DNF de literales.** Los cuantificadores se expanden sobre
  los objetos de su tipo (`exists` → disyunción, `forall` → conjunción), las negaciones se
  empujan hasta los átomos e `imply` se trata como `(or (not a) b)`. `=` y los hechos
  estáticos se evalúan en el momento; las cláusulas contradictorias y las subsumidas se
  eliminan. Si la DNF supera 10 000 cláusulas se aborta con un error claro (esas fórmulas
  hay que escribirlas con predicados derivados).
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
- **Candidatos:** solo se normalizan las asignaciones que hacen alcanzables los átomos
  necesarios de la precondición: los de la conjunción exterior y, de cada disyunción de
  átomos, los de alguno de sus disyuntos. Las igualdades de la conjunción exterior juntan
  los parámetros iguales o los fijan a una constante. El punto fijo es **semi-ingenuo**:
  en cada vuelta, un esquema solo busca los candidatos que usan algún hecho nuevo (ver
  `grounding/grounder.py`).

## Tarea instanciada y estados

- Un **estado** es un `State(facts, values)`: `facts` es un `int` usado como bitset sobre
  `GroundTask.facts`, y `values` la tupla de valores de las variables numéricas
  (`GroundTask.variables`; vacía en tareas proposicionales, `None` si el valor está
  indefinido). Es una `NamedTuple`: se compara y se usa como clave de diccionario como una
  tupla, con un coste medido de unos 0,3 µs por sucesor. Una acción es aplicable si
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

## Motor numérico

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
  condición se ignoran, como los borrados. Asignar una constante a la única variable de
  una condición la logra solo si ese valor la cumple (si no, solo sirve para definir la
  variable cuando no tiene valor). Un incremento variable (`increase x y`) solo cuenta si
  es positivo, y sus repeticiones se cuentan con el valor del estado (ver "Ritmos que fijan
  las acciones"). h_FF cuenta cada operador por su coste por el máximo número de
  repeticiones que necesita.

## Motor temporal

El motor temporal sigue a POPF (Coles, Coles, Fox y Long, 2010): búsqueda hacia delante
sobre acciones de inicio y fin, con una red temporal simple (STN) que decide cuándo ocurre
cada paso. El código está en `grounding/temporal.py`.

- **Compilación.** Cada acción durativa `a` da dos acciones clásicas y un hecho
  `running@a`:
  - `a@start`: condiciones `at start`, los literales `over all` que sus efectos de inicio
    no pueden tocar (la **guarda de inicio**, ver abajo) y `¬running@a` (una instancia
    no se solapa consigo misma); efectos `at start` y `running@a`;
  - `a@end`: condiciones `at end` y `over all`, y `running@a`; efectos `at end` y
    `¬running@a`.

  El objetivo exige además que no quede ninguna acción en curso. Todo lo demás se
  reutiliza tal cual: grounder, normalización ADL y numérica, sucesores compilados y
  *h^hbd* (Python, Numba y PyTorch) sobre la tarea de inicios y fines. Las acciones
  instantáneas del dominio conviven con las durativas.
- **Guarda de inicio.** Un literal `over all` que los efectos de inicio no pueden tocar
  (otro predicado, o argumentos que por sus tipos no pueden ser los mismos objetos; ni
  derivado) vale justo después de empezar lo mismo que antes de empezar: si no se cumple, el
  inicio rompe al instante su propia invariante y se descarta. Exigirlo ya en `a@start` no
  cambia la semántica, pero el grounding deja de instanciar inicios que nunca pueden
  terminar. En satellite, `take_image` tiene todas sus condiciones en `over all`: sin la
  guarda, su inicio se instanciaría para cada satélite, dirección, instrumento y modo, y
  p17 tendría 57 916 hechos (casi todos `running@a`) y 957 MB; con ella, 11 262 y 138 MB,
  con las mismas acciones. En las preferencias temporales de la IPC 2006, storage p06 y
  pathways p08 agotarían 3 GB al instanciar. En el motor discretizado, la guarda excluye
  además lo que cambian los eventos, que se disparan en el mismo instante.

  La guarda va en la precondición de `a@start`, así que la ven la búsqueda (deja de
  generar inicios que la invariante descartaría) y la heurística (pide las condiciones
  `over all` ya al empezar, como POPF y TFD). Cambia la trayectoria de casi todas las
  búsquedas. Se probó a quitarla de la relajación: con el mismo orden de sucesores, BFWS
  hacía exactamente las mismas búsquedas, pero con las preferencias temporales el
  portfolio resolvía 3 instancias menos (con planes algo mejores).
- **Estado temporal.** Un `State` (hechos y valores) con una línea temporal adjunta. Se
  compara solo por la parte lógica (memoización al estilo de POPF), así que la detección
  de repetidos funciona como en la planificación clásica. Es incompleta: un estado
  repetido con otra planificación en el tiempo se descarta, y con plazos (TIL,
  restricciones con tiempo, duraciones máximas) esa otra quizá llegaba al objetivo. Por
  eso, si la búsqueda agota la cola tras descartar algún repetido, termina con `failed`
  (no ha encontrado plan), no con `unsolvable` (no lo hay): trucks p01 con `lazy-ff-pref`
  agotaba la cola y decía que no tenía solución, y la tiene.
- **Línea temporal bajo demanda.** Al generar un sucesor se comprueba su línea temporal,
  pero no se guarda: el estado solo recuerda de dónde viene y la recalcula al expandirse.
  La búsqueda genera muchos más estados de los que expande, y cada línea temporal crece
  con la longitud del plan: guardarlas todas costaba de 10 a 14 KB por estado en las
  tareas grandes (3,5 GB en zenotravel 17 con solo 3000 expansiones), y así, de 2 a 4.
- **Órdenes.** Para cada hecho y variable se guarda el último paso que lo cambió y los que
  lo han leído desde entonces. Un paso nuevo va, con una separación ε, después del último
  cambio de lo que lee o modifica y de los que leyeron lo que escribe. Los pasos
  independientes no se ordenan: la red temporal los hace simultáneos, y el plan sale en
  paralelo aunque la búsqueda los eligiera en secuencia. Dos detalles para que VAL acepte
  los planes, sobre todo con TIL:
  - una escritura que no cambia el valor (añadir un hecho ya cierto) también se separa de
    los lectores, porque VAL no admite que en un mismo instante una acción lea un hecho y
    otra lo escriba. Solo cuentan los efectos incondicionales y los condicionales que se
    disparan;
  - los hechos `running@a` cuentan como los demás: un nuevo inicio de la misma acción va
    después de su fin anterior también en el tiempo, no solo en la secuencia.
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
- **Métrica.** Solo `(:metric minimize (total-time))`, salvo con preferencias (ver
  "Preferencias y búsqueda anytime"): su valor es la duración total. La
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

## Predicados derivados

- **Grounding.** Cada regla `:derived` se instancia como un esquema más del punto fijo
  de alcanzabilidad: su condición hace de precondición y su cabeza, de añadido. Cada
  cláusula alcanzable de la condición (en DNF) da un `Axiom` (hecho derivado, condición,
  estrato). Los predicados derivados nunca son estáticos.
- **Estado.** Los hechos derivados están en el mismo bitset que los demás.
  `GroundTask.apply` los recalcula en cada sucesor con `AxiomEvaluator`
  (`pyplanner.grounding.axioms`), y el estado inicial ya los trae calculados. Por eso el
  atajo de la búsqueda que aplica la acción directamente se desactiva si hay axiomas.
- **Evaluación por estratos.** El estrato de un predicado es el menor que deja los
  predicados que usa en positivo en su estrato o antes, y los que niega, estrictamente
  antes. En cada estrato se calcula el punto fijo mínimo con contadores, como el
  *AxiomEvaluator* de Fast Downward:
  - la parte de la condición que no cambia dentro del estrato se comprueba una vez;
  - los hechos del propio estrato (reglas recursivas) bajan un contador;
  - los axiomas se agrupan por su primer hecho externo positivo, y solo se revisan los de
    los hechos ciertos.

  La negación es de mundo cerrado: un hecho derivado es falso si ningún axioma lo hace
  cierto. En psr cuesta unos 50 µs por estado.
- **Heurísticas.** Cada axioma es una acción relajada de coste 0. Todas comparten un
  operador ficticio de coste 0 (índice `len(task.actions)`) que nunca es una acción útil,
  así que los núcleos de Numba y PyTorch no cambian. Las condiciones que **niegan** un
  hecho derivado se dan por ciertas en la relajación, como hace Fast Downward por defecto.
  Su complementario no lo añade ninguna acción: depende de que dejen de cumplirse todos
  sus axiomas, y tratarlo como un hecho más declararía callejones sin salida que no lo
  son. Es una aproximación: en psr, `(forall (?b) (not (affected ?b)))` no cuenta en la
  heurística.
- **Motor temporal.** Los axiomas se recalculan tras cada paso, antes de comprobar
  invariantes y vigilantes. Ninguna acción escribe un hecho derivado, así que leer uno
  cuenta, para las órdenes al estilo de POPF, como leer todos los hechos y variables de
  los que depende de forma transitiva. Es correcto, aunque puede ordenar más de lo
  necesario. Un predicado derivado que depende de una variable lineal (con efectos
  continuos o `?duration` no fijo) da un error claro: el programa lineal no puede decidir
  su valor.

## Literales iniciales temporizados

- **Compilación.** Cada instante distinto *t_k* de los TIL es una acción clásica `til@k` de
  coste 0 que aplica sus literales. Se encadenan con hechos `til-done@k`, así que la
  búsqueda los aplica en orden y cuando le conviene. Se hace a nivel de AST, como las
  acciones de inicio y fin: así el grounder trata como no estáticos los hechos que solo
  cambian los TIL (`visible` en satellite) y la heurística los ve como acciones relajadas.
  Un problema con TIL es siempre una tarea temporal, aunque el dominio no tenga acciones
  durativas.
- **Instante fijo.** En la red temporal, el paso de `til@k` tiene una arista desde el
  origen con peso *t_k* y otra de vuelta con peso −*t_k*. Si las órdenes lo obligan a ir
  más tarde, el sucesor se descarta. Si un retraso propagado por una duración máxima llega
  a mover el origen, también.
- **Plazos, como en OPTIC.** Todo paso que la secuencia pone antes de un TIL tiene que
  ocurrir no más tarde que él: arista de vuelta al origen con peso −*t* del próximo TIL
  pendiente. Sin esto, el estado seguiría creyendo cierto un hecho que un TIL ya ha borrado,
  porque el tiempo pasa aunque la búsqueda no aplique el TIL. Las órdenes al estilo de POPF
  hacen el resto: un TIL que borra un hecho queda después de los pasos que lo leyeron.
- **Plan.** Los TIL no aparecen en el plan de salida.
- **Memoización y padres.** En una tarea temporal, dos estados iguales pueden tener distinta
  línea temporal. Al encontrar un camino mejor a un estado conocido, la búsqueda inmediata
  solo cambia su padre si todavía no lo ha expandido, y entonces guarda también el estado
  del camino nuevo, con su línea temporal. Con los TIL (coste 0) esto pasa a menudo, y
  cambiar el padre de un nodo ya expandido daba planes que no cumplían los plazos al
  repetirlos desde el principio.
- **Plazos en la heurística** (`pyplanner.heuristics.deadlines`). En las tareas con TIL,
  la heurística va envuelta en `DeadlineHeuristic`, que descarta los estados que no pueden
  cumplir los plazos y deja el valor y las acciones útiles a la heurística base:
  - **Instantes más tempranos.** Un Dijkstra sobre la relajación, como h_max, donde el
    coste es el tiempo absoluto:
    - los hechos ciertos lo son desde el último paso que los cambió;
    - el fin de una acción durativa va, como pronto, tras su inicio más la duración
      mínima (las acciones en curso, desde su inicio real);
    - los TIL pendientes ocurren en su instante.
  - **Ventanas.** Los hechos que solo cambian los TIL solo son ciertos en los intervalos
    que marcan el estado y los TIL pendientes. Una precondición sobre ellos retrasa la
    acción hasta el primer intervalo que la cumple, y una condición `over all` exige que
    la acción entera quepa.
  - **Corrección.** Todo son cotas inferiores, así que nunca se descarta un estado con
    solución.
  - **Implementación.** Las entradas de cada estado se preparan con NumPy. El Dijkstra
    tiene una versión de referencia en Python y otra en Numba
    (`compiled_deadlines`), con el mismo resultado; para en cuanto alcanza el objetivo.
    Cuesta más o menos lo mismo que h_FF compilada (de 1 a 2 veces).
  - **Línea temporal.** Necesita la de cada estado evaluado: la calcula sin guardarla
    (`TemporalState.peek_timeline`), para no perder el ahorro de memoria de no guardarlas.
  - **Aplazada en las búsquedas inmediatas.** Evaluar todos los sucesores con la
    comprobación doblaba el tiempo en los casos en que no ahorraba expansiones: la
    mayoría de los sucesores nunca se expanden. La búsqueda inmediata la pide a la
    heurística (`Heuristic.defer_dead_end_check`) y la hace al sacar cada nodo de la
    cola. Un nodo descartado así no gasta el turno de su cola (`retry`), así que el orden
    de los nodos vivos es el mismo que si se hubiera descartado al generarlo, y la
    búsqueda expande los mismos nodos. La perezosa ya evalúa al sacar de la cola.
  - **Sin TIL pendientes** no se comprueba nada: ya no hay plazos y la relajación de la
    heurística base detecta los mismos callejones.
  - **Plan relajado que respeta los tiempos:** es h_TFF (ver la sección siguiente), una
    heurística aparte que incluye esta comprobación.

## Plan relajado temporal: h_TFF

h_FF no ve el tiempo: en satellite, su plan relajado usa antenas cuya ventana de
visibilidad ya ha pasado, y sus acciones útiles llevan a callejones sin salida. h_TFF
(`tff`; `heuristics/temporal_ff.py`) extrae un plan relajado que respeta los tiempos, como
el grafo de planificación temporal de POPF y OPTIC (Coles, Coles, Fox y Long, 2010):

- **Instantes más tempranos:** los de la comprobación de plazos (el Dijkstra de arriba, con
  duraciones mínimas, TIL pendientes y ventanas).
- **Soportador de cada hecho:** la acción relajada de menor índice entre las que le dan su
  (instante, nivel) mínimo, con el nivel de h_FF (0 si es cierto en el estado, 1 + el
  máximo de las precondiciones). No depende del orden del cálculo, y el plan relajado no
  tiene ciclos.
- **Cláusula del objetivo:** la que antes se cumple (con sus ventanas); si ninguna puede
  cumplirse, el estado es un callejón sin salida. Son los mismos callejones que la
  comprobación de plazos, así que h_TFF no va envuelta en ella.
- **Valor y acciones útiles**, como en h_FF: el coste de los operadores del plan relajado
  (por sus repeticiones numéricas) y los operadores del plan con todas sus precondiciones
  ciertas, también las de los hechos con ventanas.
- **Tres implementaciones con el mismo resultado:** la referencia en Python (Dijkstra por
  (instante, nivel)), Numba y PyTorch. En PyTorch, los instantes y los niveles son dos
  puntos fijos sobre el lote. Los niveles se calculan solo sobre las aristas *ajustadas*
  (las que dan al hecho su instante exacto): el truco de h_FF en PyTorch (la última
  iteración en que bajó el coste) no vale con máximos, porque una precondición con
  holgura puede alcanzarse por un camino más corto que no es el más temprano. Ajustar a
  las ventanas usa los intervalos de cada estado en un tensor relleno.
- **Solo con TIL.** Sin plazos, lo único que cambia frente a h_FF es elegir los soportadores
  por instante en lugar de por coste, y eso da peores planes relajados (como h_max frente
  a h_add): zenotravel temporal pasaba de 17 a 13 instancias resueltas y driverlog de 16 a
  14–15. Así que en las tareas sin TIL (temporales o no) `tff` crea h_FF, con sus
  envoltorios, y sus configuraciones sirven para cualquier tarea. Quitar a h_FF solo las
  acciones que no caben en el tiempo, manteniendo los soportadores por coste, no sirve:
  casi nunca son las que elige (en satellite, las mismas expansiones que h_FF).
- Cuesta más o menos lo mismo que h_FF compilada (satellite p02: 0,04 s por 1500 estados
  cada una, más la preparación de las entradas).

## Restricciones de trayectoria

Las restricciones duras de PDDL 3 (`:constraints` del dominio y del problema) son
**monitores** del motor (`pyplanner.grounding.monitors`). Cada restricción instanciada (los
`forall` se expanden sobre los objetos) es un pequeño autómata con hechos propios en el
estado, que se actualiza tras cada paso con el estado nuevo, como los axiomas pero con
memoria.

- **Sin tiempo:** `always` descarta el paso que la incumple; `sometime` tiene un hecho
  `done` que exige el objetivo; `at-most-once` recuerda si φ se ha cumplido y si ha dejado
  de cumplirse; `sometime-before` recuerda si ψ se ha cumplido en un estado anterior;
  `sometime-after` deja φ pendiente hasta que se cumpla ψ, y el objetivo exige que no
  quede nada pendiente; `at end` se añade al objetivo.
- **Con tiempo:** usan **marcas** de instante. En una tarea temporal, cada instante es un
  TIL sin literales (su hecho `til-done@k` es la marca): por la regla de OPTIC, los pasos
  anteriores a un TIL pendiente ocurren no más tarde que él, y la heurística de plazos
  entiende las ventanas que crean. En una tarea clásica, el instante es el
  número de pasos (como en VAL, la acción k ocurre en el instante k): un contador de pasos,
  saturado en el último instante que importa, enciende las marcas.
  - `within t φ`: `done` si φ se cumple antes de la marca;
  - `hold-after` y `hold-during`: descartan el paso que incumple φ con las marcas
    adecuadas;
  - `always-within t φ ψ`: en clásico, un contador de los pasos que lleva pendiente; en
    temporal, la línea temporal recuerda el paso que disparó φ y, cuando se cumple ψ, la
    red temporal recibe la arista "ψ no más tarde que t después de φ", como una duración
    máxima.
- **Órdenes en el motor temporal.** Un paso que puede cambiar algo vigilado lee y escribe
  los hechos del monitor (y lee sus marcas), así que esos pasos quedan ordenados en el
  tiempo como en la secuencia: el resultado de un monitor depende del orden.
- **Heurística.** Las salidas buenas de los monitores (`done`, quitar lo pendiente) son
  reglas relajadas de coste 0 con el operador ficticio de los axiomas. Las que solo
  descartan estados no cuentan, así que la relajación sigue siendo optimista.
- **Coste.** Un monitor solo se reevalúa si el paso cambia algo de sus fórmulas o de sus
  marcas.
- **VAL** valida `within`, `hold-during`, `always-within` y los operadores sin tiempo,
  pero su `hold-after` no funciona (falla con planes válidos o se cae): esos planes se
  comprueban con la simulación propia.

## Preferencias y búsqueda anytime

Una preferencia es una condición deseable: cada incumplimiento suma a la métrica el
coeficiente de su `(is-violated p)`, su **peso**. El grounder las compila a costes de
acción (Keyder y Geffner, 2009; `pyplanner.grounding.preferences`), así que la búsqueda
solo minimiza costes y el coste de un plan es su métrica.

- **Pesos.** Salen de la forma lineal de la métrica, que tiene que minimizarse. Las
  preferencias que no aparecen en ella (o pesan 0) no cuentan y no se compilan. Sin
  métrica, ninguna cuenta (queda una nota). Una métrica no lineal o un peso negativo se
  rechazan con un error claro.
- **Preferencias del objetivo y de las restricciones** (se evalúan al final del plan).
  Cada instancia (una por asignación de los `forall` que la rodean; `is-violated` cuenta
  las incumplidas) se resuelve en una **fase final**:
  - `_end` (coste 0) se aplica con el objetivo duro cumplido; después ninguna acción del
    dominio es aplicable;
  - cada instancia se resuelve con `_collect` (coste 0) si se cumple o con `_forgo` (coste
    su peso) si no;
  - se resuelven en un orden fijo, así que la fase final no multiplica los caminos.
- **Restricciones-preferencia.** Son monitores **blandos**: donde uno duro descartaría el
  paso, uno blando marca su hecho `violated`, borra los demás y deja de actualizarse, y ese
  paso cuesta además su peso (una **penalización** de la transición, que la búsqueda suma
  a g): incumplida una vez, ya no se puede evitar. Las que solo se incumplen a mitad del
  plan (`always`, `at-most-once`, `sometime-before`, `hold-after`, `hold-during`) no pasan
  por la fase final; las que se deciden al final (`sometime`, `within`, lo pendiente de
  `sometime-after` y `always-within`) se resuelven en ella según `done` o lo pendiente.
  `(preference p (at end φ))` es como una del objetivo.
- **Restricciones triviales.** Las que se cumplen en cualquier plan porque, con los hechos
  estáticos y los inalcanzables, sus fórmulas quedan en cierto o en falso (`always` de algo
  siempre cierto, `sometime-before` de algo imposible...) se descartan, duras o blandas.
  Un `forall` sobre `always`, `hold-*` o las que tienen disparador se instancia como una
  acción: solo con las asignaciones que hacen alcanzables
  los átomos necesarios para incumplirla o dispararla. En storage, la restricción de cajas
  incompatibles en zonas contiguas recorre solo las zonas conectadas: en p20 pasa de 2,8
  millones de instancias a las 37 000 que importan, y el grounding de 197 s y 2,7 GB a 6 s
  y 350 MB.
- **Preferencias de las precondiciones** (cuentan en cada aplicación). La acción se
  desdobla según qué instancias cumple: la variante que incumple unas cuesta además sus
  pesos y exige que se incumplan, así que el coste es exacto. Con más de 256 variantes por
  acción, error.
- **Coste de los pasos.** Si la métrica solo mide preferencias, las acciones costarían 0 y
  la heurística no distinguiría un plan corto de uno largo. Cada acción del dominio cuesta
  además una milésima del menor peso: se prefiere cumplir a renunciar, y entre dos planes
  igual de buenos, el más corto. La métrica que se informa lo descuenta.
- **El plan** no incluye la fase final (sus acciones empiezan por `_`, que no puede
  empezar un nombre PDDL). Informa de la métrica y de las instancias incumplidas
  (`; violated preferences: ...` en IPC, `violated` en el JSON).
- **Heurística.** La fase final es parte de la relajación: h_FF suma los pesos de las
  preferencias que no puede cumplir y el coste de los pasos de las que sí.

**Búsqueda anytime** (`pyplanner.search.anytime`). Con preferencias, el primer plan rara
vez es el mejor, y buscar el óptimo de una vez no es viable en tareas grandes. Como LAMA,
se encadenan búsquedas, cada una con la **cota** del mejor plan encontrado:

1. la configuración elegida con **coste unitario**, sin cota: un plan cualquiera, cuanto
   antes (su coste real se calcula después). Con los costes reales, una búsqueda voraz
   persigue las preferencias y puede no terminar: en storage, sin objetivo duro, explora
   sin fin estados donde parecen alcanzables;
2. la configuración elegida con los costes reales y el **incumbente** (ver abajo);
3. WA* con h_FF y acciones útiles, con pesos 5, 3, 2 y 1 (perezosa o inmediata, con la
   exploración y el desempate de la elegida);
4. A* con h_max, que es admisible: si agota la cola, el mejor plan es **óptimo**.

Tras cada plan se pasa a la siguiente; la última se repite mientras mejore. Termina al
agotar el tiempo (60 s por defecto, `-t` para cambiarlo), al demostrar el óptimo o al
alcanzar el límite de memoria.

- **Incumbente.** Cada estado expandido en el que se cumple el objetivo duro es un plan
  posible: desde él, la fase final cuesta exactamente los pesos de las preferencias que no
  cumple. Si ese plan mejora la cota, la búsqueda lo apunta, baja la cota hasta él y
  sigue; al terminar sin otro plan, devuelve el incumbente. Una búsqueda voraz que avanza
  cumpliendo preferencias va dejando así planes cada vez mejores, aunque nunca entre por sí
  misma en la fase final.
- **Poda por cota.** La búsqueda descarta los nodos con g (que incluye las
  penalizaciones) igual o mayor que la cota; con una heurística admisible, también los de
  g + h. Si una búsqueda inmediata con reapertura (A* o WA*) agota la cola, no hay plan
  mejor que la cota.
- **Óptimo** quiere decir mínimo coste de búsqueda: métrica más el coste de los pasos. Es
  el de menor métrica salvo que uno mejor necesite mil pasos más por cada unidad del menor
  peso.
- **En el portfolio**, cada miembro hace su búsqueda anytime empezando por su
  configuración, informa de cada plan mejor y todos comparten la mejor cota
  (`multiprocessing.Value`). Termina cuando se agota el tiempo, cuando todos han terminado
  o cuando uno demuestra el óptimo.
- **Storage** (IPC 2006, preferencias cualitativas): p01 y p02 llegan al óptimo demostrado
  (métricas 0 y 1) en menos de un segundo, con cualquier portfolio. Las métricas coinciden
  con las de VAL.
- **Métricas que se maximizan:** maximizar es minimizar la
  negación. Los costes son los de la negación y la métrica del plan se informa con su
  signo: `(maximize (+ (* 9.9 (- 1 (is-violated d1))) …))` es minimizar el peso 9,9 de
  `d1`. Si así algún peso o coste queda negativo, el error (o el coste unitario) es el de
  siempre.
- **En tareas temporales:** la tarea de inicios y fines se compila igual (fase final,
  monitores blandos con las marcas de instante de las restricciones de trayectoria,
  variantes de las acciones), con estos cambios:
  - las preferencias de una condición `at start` o `at end` de una acción durativa son
    preferencias de la precondición de su acción de inicio o de fin; las de `over all`
    no se admiten (error claro);
  - las acciones de la fase final no son pasos de la red temporal (no alargan el plan), y
    `_end` exige además que no quede ninguna acción durativa en curso;
  - el orden de los pasos que cambian algo que vigila un monitor se calcula con un índice
    de hechos y variables vigilados hacia sus monitores, calculado una vez, en lugar de
    recorrer todos los monitores para cada acción: en storage p10 (4139 monitores y 9546
    acciones), el grounding tarda 1,3 s en lugar de 26 s;
  - `total-time` en una métrica con preferencias es la duración del plan, que no es un
    coste de acción: la búsqueda no la optimiza (nota) y la métrica del plan la suma con
    su peso. Sin preferencias, la métrica de una tarea temporal sigue siendo solo
    `(minimize (total-time))`.
- **Límites.** VAL calcula mal las preferencias del objetivo si el problema tiene una
  sección `(:constraints (and))` vacía; un `sometime-before` con una comparación numérica
  estricta (`(> x 0)`) le parece incumplido siempre; y en pathways (IPC 2006) da p0a por
  incumplida pero no suma su peso a la métrica.

## Procesos y eventos

Los dominios con procesos y eventos (PDDL+) van a un motor con el **tiempo discretizado**
(`pyplanner.grounding.discrete`), al estilo de ENHSP, DiNo o UPMurphi. Las tareas
temporales sin `:time` siguen en el motor temporal, que es más preciso, salvo las que este
no admite por sus efectos continuos (ritmos no lineales, como el repostaje de
generator-nonlinear, donde el caudal crece con el tiempo que lleva repostando): esas pasan
al motor discretizado, con una nota. Si tampoco las admite, el error es el del motor
temporal.

- **Pasos de la búsqueda:** las acciones instantáneas, el inicio y el fin de cada acción
  durativa, y una acción `_wait` que deja pasar δ unidades de tiempo (1 por defecto;
  `--delta` en la CLI, `delta` en `solve`). La espera se acorta si antes una acción
  durativa llega a una cota de su duración, para que pueda terminar justo a tiempo. Hay
  además una espera hasta el siguiente cambio (`_wait-until`, ver "Ritmos que fijan las
  acciones").
- **Integración exacta** (`pyplanner.grounding.flows`): durante una espera, las variables
  siguen la dinámica de los procesos activos. Se integra con series de Taylor, que en
  dinámicas polinómicas (car: la velocidad crece con la aceleración y la distancia con la
  velocidad; generator: un caudal que crece con el cuadrado del tiempo) **terminan** y dan
  los mismos valores que VAL. Si la serie no termina, Runge-Kutta 4 con 100 subpasos por
  unidad. Los valores se redondean a 9 decimales: las sumas de pasos no arrastran ruido y
  las igualdades como `(= (v) 0)` se pueden cumplir.
- **Eventos en su instante exacto.** Tras cada cambio, los eventos se disparan en cascada.
  Si durante una espera se cumple la precondición de un evento (o cambia qué procesos están
  activos), el instante se busca por bisección sobre el flujo exacto, como hace VAL, y la
  espera sigue hasta completar δ. Sin esto, un evento se detectaría al final de la espera:
  en generator, un depósito quedaría con combustible negativo y el generador recibiría de
  más, y la búsqueda acabaría aprovechando ese error (5 de 8 instancias de
  generator-events; 8 de 8 con el instante exacto).
- **Invariantes:** las condiciones `over all` de las acciones durativas en curso se
  comprueban tras cada cambio (no en mitad de una espera).
- **Una acción instantánea** va seguida de una espera de ε = 0,01: VAL no admite dos
  acciones en el mismo instante, y durante ese ε los procesos siguen actuando, igual que en
  VAL.
- **Acciones durativas:** se compilan en un inicio (pone en marcha un reloj y fija las
  cotas de su duración), un proceso `a@run` (el reloj y los efectos continuos) y un fin
  (el reloj dentro de la duración). Las condiciones `over all` son invariantes.
- **Coste:** δ por espera y unos 2ε por acción (ε por el tiempo que pasa y ε por ser un
  paso más; sin el segundo, A* devolvía planes con decenas de acciones inútiles que hacían
  pasar el tiempo de ε en ε). Es 1/64, para que las sumas de costes sean exactas, y la
  espera hasta el siguiente cambio cuesta lo que avanza. La métrica que se informa es la
  duración del plan.
- **Heurística:** en la relajación, la espera es una acción con un efecto condicional por
  proceso (sus ritmos × δ); *h^hbd* cuenta las repeticiones si el ritmo es constante, como
  el reloj de una acción durativa. Los eventos son reglas relajadas de coste 0 sobre
  hechos. Lo que la relajación no ve, las invariantes y los ritmos que fijan las acciones,
  lo añaden las dos secciones siguientes.
- **Novedad:** las variables que cambian de forma continua no cuentan. Toman un valor
  nuevo en cada espera, así que todo estado parecía nuevo y la tabla crecía sin fin (en
  generator-events p06, 85 000 átomos y 2 GB en 20 s). Limitar sus valores distintos (16,
  64, 256, 1024) en lugar de ignorarlas da resultados que dependen del límite sin ninguna
  tendencia.
- **Validación con un paso más fino:** el plan se vuelve a simular con δ/10. Si falla (una
  invariante que se rompe en mitad de una espera, por ejemplo), `solve` busca otra vez con
  δ/2, hasta 3 veces, dentro del límite de tiempo. Es más estricto que VAL, que solo
  comprueba las invariantes en los *happenings*: en generator p02, un plan que VAL daba
  por válido dejaba la gasolina en −1,08 entre dos *happenings*. Cada espera se recorre a
  trozos de δ/10 para comprobar los instantes de en medio, pero termina en el estado de la
  búsqueda: el redondeo de miles de trozos se acumularía, y un plan que agota el
  combustible justo al final de una espera larga quedaría en −10⁻⁹.
- **Sin plan:** agotar la búsqueda no demuestra nada (con un δ menor podría haber plan):
  termina con `failed`.
- **Literales iniciales temporizados:** son eventos de un reloj global, `til@clock`, que
  avanza a ritmo 1, así que se disparan en su instante exacto. Una espera que termina justo en el instante de un TIL deja pasar además ε: VAL
  no admite que una acción lea un hecho en el mismo instante en que un TIL lo cambia.
- **Restricciones de trayectoria:** los monitores de las tareas sin procesos. Los
  instantes de `within`, `hold-after` y `hold-during` son eventos del mismo reloj (TIL sin
  literales). Como VAL, se comprueban en los *happenings*: tras cada acción, y tras una
  espera solo si un evento o un TIL han cambiado los hechos. El final de una espera en la
  que no pasa nada no es un *happening*: comprobarlas también ahí daba por buenos planes
  que VAL rechaza (un `within` que se cumple en mitad de una espera). La espera hasta el
  siguiente cambio también pasa por los monitores.
- **Efectos condicionales en las acciones durativas.** Con condiciones `over all`, un
  vigilante como el del motor temporal: se activa al empezar si se cumplen sus condiciones
  y se apaga en cuanto la condición `over all` deja de cumplirse.
  Durante una espera, los hechos solo cambian con los eventos, en su instante exacto, y ahí
  se comprueba: un evento que rompe la condición apaga el vigilante aunque otro la
  recupere antes del final de la espera. Un efecto continuo condicional se decide al
  empezar (como en el motor temporal, solo puede depender de condiciones `at start`): un
  hecho `rate@a@k` recuerda si la condición se cumplía, y un proceso aparte, `a#k@run`, aplica el
  ritmo mientras la acción sigue en curso.
- **Preferencias:** como en las tareas sin procesos, compiladas en costes. La fase final
  (`_end`, `_collect`, `_forgo`) va detrás de las esperas y queda fuera del tiempo: no deja pasar ε ni dispara eventos. `_end` exige además que no haya acciones
  durativas en curso. La métrica ya no es la duración: cada espera cuesta lo que un paso.
  Las restricciones-preferencia son los monitores blandos de las preferencias, también en
  los *happenings*.
- **Límites** (error claro): `always-within` con procesos y eventos.

## Heurística con procesos y eventos

La relajación ignora lo que empeora las condiciones, así que no ve las invariantes: para
ella, el generador puede funcionar 1000 unidades con 960 de combustible y basta con
esperar. Las búsquedas voraces esperarían hasta que la invariante se rompiera y ya fuera
tarde para repostar. Lo que falta lo añade `pyplanner.heuristics.projection`:

- **Duraciones en la relajación.** Las cotas de la duración que no dependen de nada que
  cambie se comparan con el reloj tal cual, sin una función intermedia que la relajación
  no sabría cumplir. Asignar una constante (el reloj a 0 al empezar) no cuenta como lograr
  cualquier condición sobre esa variable (ver "Motor numérico"). Los relojes valen 0 desde
  el estado inicial. Con esto, h(inicial) en generator es de unas 1000 esperas (sin ello,
  0,04).
- **Proyección** (`DiscreteTask.shortfall`): se simula el paso del tiempo sin hacer nada
  hasta el final obligatorio (la duración mínima) de cada acción durativa en curso con
  invariante. Si la invariante se rompe antes, cuenta lo que faltaba, y la heurística le
  suma esos faltantes. En la simulación, cada acción en curso termina al llegar a su
  duración máxima (o a la mínima, si no tiene máxima) o al romperse su propia invariante:
  se aplican los efectos de su fin aunque no se cumplan sus condiciones, para que deje de
  actuar también un proceso que dependa de ellos (un repostaje se detiene al llenarse el
  depósito). Sin máxima ni ruptura, una acción seguiría para siempre: repostar sin parar
  tapaba cualquier faltante.
- **Anticipación.** Si solo contaran las acciones en curso, empezar el generador subiría el
  valor de golpe (de 1000 a 1040 en generator-toricelli p02), y la búsqueda perezosa se
  quedaría explorando los estados anteriores: empezar y terminar un repostaje en ε da un
  estado nuevo cada vez (93 000 estados sin pasar del estado inicial). Por eso también se
  simula empezar cada acción durativa con invariante y duración mínima que la base da como
  útil, y el valor es el mayor de los faltantes.
- **Acciones útiles.** Las de la base no arreglan un faltante (para la relajación basta con
  esperar), y la búsqueda perezosa, que pone a los sucesores el valor del padre, bajaría
  esperando hasta que repostar ya no sirviera. Al expandir un estado con un faltante, las
  útiles son las que lo reducen. Hacen falta tantas simulaciones como acciones
  candidatas (las que cambian valores o hechos de las condiciones de procesos y eventos),
  así que se calculan solo en los estados que se expanden (`Heuristic.helpful`), no en
  todos los que se evalúan.
- **Rupturas que no duran.** La bisección de una espera solo ve los cambios que siguen al
  final del tramo, que en una espera de δ es lo previsto (la validación con δ/10 descubre
  el resto). La proyección avanza tramos de cientos de unidades. En generator-events, con
  los depósitos enganchados tarde, el combustible baja de 0 y se recupera (el caudal crece
  con el cuadrado del tiempo), y una proyección por tramos no lo vería. Por eso comprueba
  entre cada dos instantes en que se anula alguna de las condiciones numéricas en juego. Con un flujo
  exacto, una condición polinómica es un polinomio en el tiempo: sus coeficientes son sus
  derivadas a lo largo del flujo (`Flow.crossings`), que se calculan una vez por condición,
  y sus raíces, con fórmula hasta grado 2. Con Runge-Kutta, solo la bisección.
- **Rendimiento.** El flujo redondea solo las variables que cambian, y los valores tras una
  acción o un evento, solo los que cambian (los demás son los mismos objetos, ya
  redondeados). La comprobación de cambios durante una espera evalúa una vez lo del estado
  de partida, y solo los procesos con condiciones numéricas (los hechos no cambian mientras
  pasa el tiempo). Así, la búsqueda de generator-toricelli p09 con `bfws-ff-pref` tarda la
  mitad (de 13,1 a 6,2 s con el perfilador).

## Ritmos que fijan las acciones

En algunos dominios, el ritmo de un proceso lo fijan las acciones: por ejemplo, una
economía con un proceso de producción cuyo ritmo depende de los trabajadores que se ponen o
se quitan, donde llegar al objetivo exige encadenar acciones y esperas de miles de unidades
de tiempo. Sin estos mecanismos, h_FF trataría la espera como si bastara una para cualquier
cantidad, la única acción útil sería esperar y la búsqueda no avanzaría:

- **Incrementos variables en la relajación** (`pyplanner.heuristics.relaxation`): el efecto
  de la espera, ritmo × δ, no es constante, y *h^hbd* lo trataba de forma optimista: una
  espera bastaría para cualquier cantidad, aunque el ritmo fuera 0. La arista exige que
  el cambio sea positivo (un nodo más de la relajación, logrado por las acciones que
  suben el ritmo) y cuenta las esperas con el ritmo del estado o, si es menor, con el
  menor que le da una sola acción: ⌈déficit / max(ritmo·δ, ρ)⌉. Así, poner en marcha un
  ritmo no sube el valor y acelerarlo lo baja, y las acciones útiles pasan a ser las que
  ponen en marcha la producción.
- **Esperar hasta el siguiente cambio** (`_wait-until`, `DiscreteTask.wait_until`):
  avanza hasta el primer instante en que se cumple una condición numérica de una acción o
  un evento aplicables por sus hechos o del objetivo, o una acción durativa llega a una
  cota de su duración. Los instantes salen de las raíces exactas del flujo
  (`Flow.crossings`); una invariante que se rompe por el camino invalida el paso. Juntar
  150 unidades a ritmo constante es un paso en lugar de 150. Cuesta lo que
  avanza (`GroundTask.transition`), así que A* sigue minimizando la duración, y puede
  mejorarla: en el depósito de los tests, de 2,02 a 1,67, porque no espera pasos enteros
  de δ. No está en la relajación: es útil cuando lo es `_wait`.
- **Costes exactos.** Con repeticiones de cientos o miles, unas sumas de costes de 0,02
  arrastrarían ruido de redondeo, y en CUDA (que suma en otro orden) los empates entre
  soportadores se romperían de otra forma: h_FF daría otro valor que la referencia. El
  coste de una acción instantánea es 1/64 (una fracción binaria cercana a 2ε) y las sumas
  son exactas: Python, Numba y PyTorch coinciden en todos los estados.
- **Calidad.** La búsqueda voraz encuentra antes planes que esperan hasta la siguiente
  condición, y no siempre son los más cortos: en car-nodrag, acelerar, esperar hasta
  d ≥ 30 y frenar dura 15,50 (frenando antes, en la rejilla de δ, 11,03).

## Heurísticas

| Nombre | Admisible | Acciones útiles | Numba | PyTorch |
|--------|-----------|-----------------|-------|---------|
| `blind` | sí | no | no | no |
| `goalcount` | no | no | no | no |
| `hmax` | sí | no | sí | sí |
| `hadd` | no | no | sí | sí |
| `ff` | no | sí | sí | sí (costes y plan relajado en tensores) |
| `tff` | no | sí | sí | sí (instantes, niveles y plan relajado en tensores); en tareas sin TIL es `ff` |

**Dispositivos** (`SearchConfig.device`, `--device`):

| Dispositivo | Qué es | Cuándo usarlo |
|-------------|--------|---------------|
| `auto` (por defecto) | `numba` si está instalado; si no, `python` | siempre, salvo que se quiera forzar otro |
| `python` | implementación de referencia en Python puro | tests y máquinas sin Numba |
| `numba` | la referencia compilada con Numba (`pip install -e ".[fast]"`); 12–19 veces más rápida | es la opción rápida para búsquedas normales |
| `cpu`, `cuda` | PyTorch por lotes (`pip install -e ".[gpu]"`) | solo con lotes grandes; en GPU, 4 veces más rápido que Python con lotes de 256 |
| `hybrid` | cada lote en Numba o en la GPU según su tamaño, con un umbral calibrado al empezar | con lotes adaptativos y tareas grandes (ver "Lotes adaptativos") |

Los tres dan **exactamente los mismos valores y acciones útiles** que la referencia; los
tests lo comprueban en los problemas de prueba y en tareas hechas a mano, también
aleatorias, con muchos empates y costes 0.

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

- `algorithm`: `gbfs`, `astar`, `wastar` (con `weight`), `iw` (ver "Búsqueda por
  novedad"), `ehc` o `mrw` (ver "Búsquedas locales");
- `heuristic`;
- `preferred_operators`: dos colas alternas, la normal y la de acciones útiles;
- `lazy`: evaluación perezosa; no se puede usar con A*. Las acciones de un nodo con la
  misma prioridad van en una sola entrada de la cola, que reserva sus desempates seguidos:
  salen en el mismo orden que una entrada por sucesor, sin su memoria (con desempate
  aleatorio, una por sucesor);
- `tie_breaking`: `fifo`, `lifo` o `random`, con `seed`;
- `exploration`: `"type-based"` añade una cola de **exploración por tipos** (Xie et al.,
  2014) que se alterna con las demás: saca un tipo (h, g) al azar y un nodo al azar de
  ese tipo. Evita que la búsqueda voraz se atasque en mesetas y zonas con muchos
  callejones sin salida;
- `landmarks`: añade la heurística de landmarks (h_LM) con sus propias colas, que se
  alternan con las de `heuristic`, al estilo de LAMA (ver "Landmarks"); solo con GBFS sin
  novedad;
- `novelty`: anchura de la novedad (0, 1 o 2; ver "Búsqueda por novedad");
- `batch_size`; con `adaptive_batch`, el mínimo de un lote adaptativo que se duplica
  (hasta `max_batch_size`) tras `plateau_patience` lotes sin mejorar la mejor h (ver
  "Lotes adaptativos");
- `device`: `auto` (por defecto), `python`, `numba`, `cpu` o `cuda` (ver arriba);
- `max_time`, `max_expansions` y `max_memory` (MB de memoria residente del proceso; se
  consulta cada 32 iteraciones leyendo `/proc/self/statm`). En los miembros del portfolio
  y en la CLI hay además un límite duro (ver "Uso desde la terminal").

Garantías:

- **Óptimo:** A* con `blind` o `hmax` devuelve planes óptimos, también con lotes. En la
  búsqueda inmediata, un objetivo solo se acepta si es el primero que sale de la cola en
  su lote.
- **Completo:** todas las configuraciones lo son salvo IW(k) y los paseos aleatorios. Los
  callejones sin salida que se descartan son estados desde los que el objetivo es
  inalcanzable incluso en la relajación. IW poda estados por su novedad: si agota la cola
  habiendo podado alguno, termina con `failed` (sin plan, pero sin demostrar que no lo
  haya), nunca con `unsolvable`. Los paseos aleatorios no terminan sin plan: los para el
  límite de tiempo.
- **Cancelable:** `search(..., stop=evento)` consulta el evento periódicamente.

Configuraciones predefinidas (`pyplanner configs`):

| Nombre | Configuración | Notas |
|--------|---------------|-------|
| `gbfs-ff-pref-tb` | GBFS + h_FF + acciones útiles + exploración por tipos | |
| `gbfs-ff-pref` | GBFS + h_FF + acciones útiles | |
| `lazy-ff-pref` | GBFS perezosa + h_FF + acciones útiles | evalúa muchos menos estados |
| `lazy-ff-pref-tb` | la anterior + exploración por tipos (semilla 1) | |
| `gbfs-ff-lm-pref` | GBFS + h_FF y landmarks en colas alternadas + acciones útiles | ver "Landmarks" |
| `lazy-ff-lm-pref` | la anterior, perezosa | |
| `lazy-ff-lm-pref-tb` | la anterior + exploración por tipos (semilla 1) | la mejor individual en las tareas clásicas |
| `gbfs-tff-pref`, `lazy-tff-pref`, `lazy-tff-pref-tb`, `bfws-tff-pref`, `gbfs-tff-pref-tb`, `gbfs-tff-pref-hybrid` | las de h_FF con h_TFF | en tareas sin TIL, iguales que las de h_FF |
| `gbfs-ff` | GBFS + h_FF | |
| `gbfs-add` | GBFS + h_add | |
| `wastar-add-3` | WA* (w = 3) + h_add | |
| `gbfs-goalcount-rnd` | GBFS + objetivos pendientes, desempate aleatorio | muy barata; aporta diversidad |
| `bfws-ff-pref` | BFWS: GBFS con prioridad (novedad, h_FF) + acciones útiles | la más robusta (resuelve todos los problemas de prueba); en los portfolios, con `tff` (`bfws-tff-pref`, la de 1 núcleo) |
| `bfws-ff` | BFWS con h_FF | |
| `bfws-goalcount` | BFWS con objetivos pendientes | sin heurística de relajación |
| `iw1`, `iw2` | IW(1), IW(2) | incompletas; solo para problemas de anchura baja |
| `ehc-tff-pref` | *enforced hill-climbing* (FF) + h_TFF + acciones útiles | si se atasca, sigue con GBFS; fuera de los portfolios |
| `mrw-tff`, `mrw-tff-pref` | paseos aleatorios (Arvand), puros o con acciones útiles (MHA) | incompletas; fuera de los portfolios |
| `astar-hmax` | A* + h_max | óptima |
| `astar-blind` | A* ciega | óptima; coste uniforme |

## Búsquedas locales

Para dar diversidad al portfolio, dos búsquedas que no mantienen una cola global, sino
que avanzan desde el mejor estado encontrado (`pyplanner.search.local`):

- ***Enforced hill-climbing*** (`algorithm="ehc"`; Hoffmann y Nebel, 2001): desde el
  estado actual, una búsqueda en anchura hasta el primer estado de h menor (o un
  objetivo), que pasa a ser el actual. Con acciones útiles, la búsqueda en anchura solo
  usa las de cada estado y, si así no encuentra salida, lo intenta con todas. Si tampoco
  (un callejón que la heurística no reconoce), sigue con GBFS con la misma heurística y el
  tiempo que queda, como FF: así es completa.
- **Paseos aleatorios** (`algorithm="mrw"`; Arvand, Nakhost y Müller, 2009): desde el
  estado actual, paseos de 10 pasos al azar, y solo se evalúa el estado al que llega cada
  uno (en lotes). En cuanto uno mejora la menor h, la búsqueda salta a él; tras 2000
  paseos sin mejorarla, salta al mejor igualmente, y tras 7 saltos así vuelve a empezar
  desde el estado inicial. Los paseos se alargan (×1,5 cada 300 sin mejorar). Con
  acciones útiles, cada paso las elige con probabilidad proporcional a exp(Q/10), donde Q
  cuenta las veces que la acción ha sido útil en los estados evaluados (MHA). Es
  incompleta.

Las dos dejan ciclos en el plan (volver a un estado ya visitado); se quitan al final si el
plan sin ellos sigue siendo válido y no cuesta más. Con preferencias, la búsqueda
*anytime* pasa de los paseos aleatorios directamente a WA*: no terminan sin plan, así que
con la cota se quedarían con todo el tiempo.

No están en los portfolios: en las 97 instancias medidas que el portfolio de 7 núcleos no
resolvía no aportan ninguna.

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

## Landmarks

Código en `heuristics/landmarks.py` y `search/engine.py`. Al estilo de LAMA (Richter y
Westphal, 2010).

- **Qué son.** Un landmark es un hecho que se cumple en algún momento de todo plan. Se
  calculan los de la relajación sin borrados, que también lo son de la tarea, sobre los
  hechos de `RelaxedTask`: los de la tarea, los complementarios de los que se niegan y las
  condiciones numéricas. Así también hay landmarks negativos y numéricos.
- **Cálculo**, una vez, desde el estado inicial: el punto fijo
  `LM(p) = {p} ∪ ⋂ₐ ⋃_{q ∈ pre(a)} LM(q)`, con `a` las acciones relajadas que consiguen `p`
  (Zhu y Givan; Keyder, Richter y Helmert, 2010). Los landmarks del objetivo son la unión
  de los de sus hechos; con un objetivo disyuntivo, los comunes a sus cláusulas. En las
  colecciones medidas tarda como mucho 0,13 s.
- **Ordenaciones:** naturales (`l` va antes que `l'` si `l ∈ LM(l')`) y
  codiciosas-necesarias (`l` es precondición de todas las acciones que pueden conseguir
  `l'` por primera vez, así que tiene que cumplirse justo antes).
- **h_LM depende del camino**, como en LAMA. Cada nodo lleva sus landmarks *aceptados*:
  uno se acepta cuando se cumple y ya lo están sus predecesores. El valor es lo que falta
  por aceptar más lo aceptado que hay que volver a conseguir: los del objetivo que no se
  cumplen y los que tienen que cumplirse justo antes de otro aún sin aceptar. Si se llega
  a un estado ya visto por otro camino, se queda con los aceptados por los dos.
- **Colas alternadas.** Con `landmarks=True`, la búsqueda tiene dos colas más (la normal y
  la de acciones útiles) ordenadas por h_LM, que se turnan con las de h_FF. Las acciones
  útiles de un nodo son las de h_FF más las que consiguen alguno de los landmarks
  siguientes (los que faltan con todos sus predecesores aceptados); estas se calculan al
  expandir el nodo, no al evaluarlo.
- Si el objetivo no es alcanzable ni en la relajación, h_LM es un callejón sin salida.
- **Coste:** del 5 % al 15 % del tiempo de búsqueda (perfilado en barman, openstacks,
  parking y visit-all).
- Solo con GBFS sin novedad, inmediata o perezosa.

## Lotes adaptativos y evaluación híbrida

Código en `search/engine.py` y `heuristics/hybrid.py`.

- **Lote adaptativo.** El tamaño de lote vuelve al mínimo en cuanto mejora la mejor h
  vista, y se duplica tras 8 lotes seguidos sin mejorarla (hasta 256). Mientras la
  heurística guía, la búsqueda es la clásica; en las mesetas expande muchos nodos de golpe
  y evalúa juntos todos sus sucesores.
- **Evaluación híbrida** (`device="hybrid"`). Numba y PyTorch dan exactamente los mismos
  valores, así que cada lote puede ir a cualquiera sin cambiar la búsqueda. El coste de
  Numba por estado no depende del lote pero crece con la tarea; el de la GPU tiene un
  coste fijo alto por llamada y crece menos. Al empezar se miden los dos en la propia
  tarea (paseo aleatorio) y se fija el tamaño a partir del cual la GPU es más rápida; si
  no lo es nunca, todo va a Numba. Coste por estado de h_FF (µs) con una GTX 970:

  | Tarea | Numba (cualquier lote) | GPU, lote 64 | GPU, lote 256 |
  |-------|------------------------|--------------|---------------|
  | slitherlink p02 (1920 acciones) | 45–59 | 253 | 158 |
  | driverlog 15 (5184) | 550 | 445 | 280 |
  | zenotravel 15 (20 160) | 1765 | 861 | 756 |

  PyTorch en CPU no gana nunca a Numba.
- **Trozos.** Los lotes de PyTorch se evalúan por trozos de como mucho 5·10⁷ elementos
  (estados × relaciones de la tarea relajada): en Windows, la GPU que dibuja la pantalla
  tiene un vigilante que aborta cualquier cálculo de más de unos 2 s
  (`cudaErrorLaunchTimeout`), y los lotes de miles de estados de una tarea grande lo
  superaban.
- **Una sola búsqueda por GPU.** Dos procesos compartiendo la GPU pierden toda la ventaja
  (zenotravel 14: 17,5 s con dos, 3,4 s con uno).
- **Portfolio mixto.** Los miembros con GPU arrancan con `spawn` y los demás con `fork`;
  si ya hay un plan, a los que siguen arrancando se les termina enseguida. Un miembro con
  GPU cuesta unas décimas en las tareas pequeñas (gripper p01: 1,1 s frente a 0,9 s).

Resultados (driverlog y zenotravel temporales 12–17, un proceso, 60 s):

| Configuración | Tiempo total | Destacados |
|---------------|--------------|------------|
| `gbfs-ff-pref` (lote 1, Numba) | — | zenotravel 16: 34,2 s |
| `gbfs-ff-pref-ab` (adaptativo, Numba) | 176 s | zenotravel 16: 25,2 s |
| `gbfs-ff-pref-hybrid` (adaptativo, híbrido) | 136 s | zenotravel 16: 14,9 s; driverlog 17: 18,2 s |

- **El lote adaptativo es neutro para la búsqueda** (con paciencia 4 perdía instancias;
  con 8, no). Su valor es que habilita la GPU, que recorta un 30–55% en las tareas grandes.
- **En el portfolio aporta poco en estas tareas** (la búsqueda perezosa ya resuelve
  zenotravel en pocos segundos), pero es el miembro con GPU de los portfolios de 6 y 7
  (ver "Portfolio").

## Portfolio

**Qué búsquedas se lanzan.** Depende del número de núcleos (`-j N`). Hay un portfolio
predefinido para cada número de núcleos de 1 a 7 (`PORTFOLIOS` en `search/config.py`), y
cada uno contiene al anterior:

| Núcleos | Búsquedas | Qué se añade |
|---------|-----------|--------------|
| 1 | `bfws-tff-pref` | BFWS, la más robusta: resuelve todos los problemas de prueba y da los mejores planes con preferencias; se ejecuta en el propio proceso |
| 2 | + `lazy-ff-lm-pref-tb` | la perezosa con landmarks y exploración por tipos, la mejor individual en las tareas clásicas |
| 3 | + `gbfs-goalcount-rnd` | la más barata y aleatoria |
| 4 | + `wastar-add-3` | WA*, que da los mejores planes con preferencias |
| 5 | + `lazy-ff-pref` | la perezosa sin landmarks |
| 6 | + `gbfs-tff-pref-hybrid` | lote adaptativo y evaluación híbrida CPU/GPU, el único miembro con GPU: casi empata con GBFS con exploración por tipos (una instancia menos, un 1 % más rápido) y se queda para las tareas grandes, donde la GPU rinde |
| 7 | + `gbfs-tff-pref-tb` | GBFS con exploración por tipos |

- Elegidos midiendo en 475 instancias del catálogo PDDLDomains, clásicas y no clásicas.
  Cada uno resuelve todos los problemas de prueba. Del de 2 al de 5 son la mejor
  combinación posible de su tamaño; el de 6 y el de 7, por el miembro con GPU, quedan a una
  instancia de la mejor.
- **Los miembros con h_FF usan `tff`** (h_TFF en las tareas con TIL y h_FF, con la misma
  búsqueda, en las demás), salvo `lazy-ff-pref`: con h_FF resuelve satellite p10 y p14, y
  con h_TFF no. La perezosa con landmarks no tiene versión con `tff`. En las tareas con
  TIL (satellite y trucks, 22 instancias), el portfolio de 1 núcleo pasa de 6 resueltas a
  8 y los de 6 y 7 de 11 a 12; en las demás no cambia nada.
- Con más de 7 núcleos se usa el de 7. Nunca hay más búsquedas que núcleos.
- Los portfolios de 1 a 5 núcleos no usan la GPU: funcionan igual en cualquier máquina y
  no pagan el arranque de PyTorch y CUDA en las tareas pequeñas. Sin GPU, el miembro
  híbrido del de 6 y del de 7 evalúa todo con Numba.
- **Por defecto se usan 6 núcleos**, para la máxima robustez, pero limitados a los que
  tenga la máquina (`os.sched_getaffinity`). En un sistema mononúcleo se usa el
  portfolio de 1 sin necesidad de indicarlo.
- Con `-c` se eligen las configuraciones a mano. Si se piden más configuraciones que
  núcleos, es un error.

**Cómo se ejecutan.**

- Un proceso por configuración; gana el primer plan y el resto se cancelan. Con
  preferencias, el portfolio es *anytime* (ver
  [Preferencias y búsqueda anytime](#preferencias-y-búsqueda-anytime)): se queda con el
  mejor plan dentro del límite de tiempo.
- Se usa `fork` cuando es seguro: sin CUDA, sin PyTorch cargado en el padre y con un
  solo hilo (también del sistema: un programa que embebe Python suele tener muchos). En
  cualquier otro caso se usa `spawn`, y cada hijo vuelve a importar el script principal:
  si este llama al planificador fuera de `if __name__ == "__main__":`, el hijo lanza un
  error claro en lugar de repetir el trabajo del script, y el padre informa de la causa.
- Si un miembro falla, informa con estado `ERROR` y un mensaje, pero no tumba el
  portfolio. Eso incluye no poder arrancarlo y terminar sin informar (se detecta al
  momento, con el código de salida).
- Los miembros de `spawn` leen la tarea de un fichero temporal, serializada una vez para
  todos. Por la tubería de arranque de cada hijo no sería seguro: CPython mantiene abierto
  el otro extremo mientras escribe, y si el hijo muere al arrancar y la tarea no cabe en la
  tubería (64 KB; slitherlink p02 ocupa 589 KB), el padre se quedaría esperando para
  siempre, aunque hubiera límite de tiempo.
- **Precompilación de Numba:** antes de lanzar los hijos, el padre compila (o carga de la
  caché) los núcleos de Numba si algún miembro los va a usar. Sin esto, un hijo cancelado
  a mitad de compilar dejaría la caché incompleta, y cada ejecución pagaría la compilación
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
pyplanner solve -t 30 dominio.pddl preferencias.pddl  # el mejor plan en 30 s
```

- El plan sale por la salida estándar y las estadísticas por la de error; `-q` las
  suprime.
- **Límite de memoria** (`-m`, o `memory_limit` en `solve`): se reparte a partes iguales
  entre los miembros del portfolio. Por defecto es el 80% de la memoria disponible, así
  que una búsqueda larga nunca agota la de la máquina (una búsqueda voraz puede generar
  cientos de miles de estados por segundo). Al llegar al límite, la búsqueda termina con
  `limit_reached` y lo explica en el detalle.
- **Límite duro** (`pyplanner.search.memory`). El límite anterior se consulta cada 32
  iteraciones y no ve lo que crece de golpe entre dos consultas ni el grounding. Por eso
  hay además un límite duro con
  `RLIMIT_DATA`: al pasarlo, cualquier reserva falla y la búsqueda termina con
  `limit_reached`.
  - **Miembros del portfolio:** cada uno se lo pone en su proceso al arrancar, algo por
    encima de su parte (el mayor entre su parte y lo que ya ocupa, más 512 MB o el 25 %).
    Antes de ponérselo carga PyTorch y CUDA, si los usa: importar PyTorch son unos 630 MB.
  - **CLI:** se pone el total a sí misma antes del grounding. Si se agota al instanciar,
    termina con el código 4.
  - **Como librería**, no se limita al proceso que llama: el grounding y la búsqueda con
    una sola configuración solo tienen el límite blando.

  Se usa `RLIMIT_DATA` y no `RLIMIT_AS` porque CUDA reserva decenas de GB de memoria
  virtual que no usa. Cuesta lo que una llamada al sistema por proceso. El miembro con GPU
  resuelve slitherlink p02 igual con el límite (9 s), y con una parte de 300 MB, menos de lo
  que ocupa CUDA (1 GB), para en la primera comprobación sin errores.
- Códigos de salida:
  - 0: plan encontrado;
  - 2: entrada no válida;
  - 3: se ha demostrado que no hay plan;
  - 4: no se ha encontrado plan: se ha llegado a un límite, o la búsqueda no puede
    demostrar que no lo haya (IW, o una tarea temporal en la que se han descartado estados
    repetidos con otra línea temporal; ver
    [Motor temporal](#motor-temporal)).

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
