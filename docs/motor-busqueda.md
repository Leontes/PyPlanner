# Motor de búsqueda (iteraciones 1 a 6)

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
| Portfolio | `pyplanner.portfolio` | Varias configuraciones en procesos paralelos; gana la primera (con preferencias, la mejor dentro del límite de tiempo) |
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
  condición se ignoran, como los borrados. Desde la 6c, asignar una constante a la única
  variable de una condición la logra solo si ese valor la cumple (si no, solo sirve para
  definir la variable cuando no tiene valor). h_FF cuenta cada operador por su coste por el
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
  paralelo aunque la búsqueda los eligiera en secuencia. Dos detalles corregidos en la
  iteración 4, cuando los TIL llevaron a la búsqueda a planes que VAL rechazaba:
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

## Predicados derivados (iteración 4)

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

## Literales iniciales temporizados (iteración 4)

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
    (`TemporalState.peek_timeline`), para no perder el ahorro de memoria de la 3b.
  - **Aplazada en las búsquedas inmediatas.** Evaluar todos los sucesores con la
    comprobación doblaba el tiempo en los casos en que no ahorraba expansiones: la
    mayoría de los sucesores nunca se expanden. La búsqueda inmediata la pide a la
    heurística (`Heuristic.defer_dead_end_check`) y la hace al sacar cada nodo de la
    cola. Un nodo descartado así no gasta el turno de su cola (`retry`), así que el orden
    de los nodos vivos es el mismo que si se hubiera descartado al generarlo, y la
    búsqueda expande los mismos nodos. La perezosa ya evalúa al sacar de la cola.
  - **Sin TIL pendientes** no se comprueba nada: ya no hay plazos y la relajación de la
    heurística base detecta los mismos callejones.
  - **Pendiente** (trabajo opcional, ver el roadmap). Extraer además un plan relajado que
    respete los tiempos y usarlo como heurística.

## Restricciones de trayectoria (iteración 5b)

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
  anteriores a un TIL pendiente ocurren no más tarde que él, y la heurística de plazos de
  la iteración 4 entiende las ventanas que crean. En una tarea clásica, el instante es el
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

## Preferencias y búsqueda anytime (iteración 5c)

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
  siempre cierto, `sometime-before` de algo imposible...) se descartan, duras o blandas. Un `forall` sobre `always`, `hold-*` o las que tienen
  disparador se instancia como una acción: solo con las asignaciones que hacen alcanzables
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
- **Storage** (tier 5): p01 y p02 llegan al óptimo demostrado (métricas 0 y 1) en menos de
  un segundo, con cualquier portfolio. Las métricas coinciden con las de VAL. Resultados
  en `benchmarks/preferences/storage` en
  [Rendimiento medido](#iteración-5c-preferencias-y-búsqueda-anytime).
- **Límites.** Las preferencias en tareas temporales todavía no se admiten (error claro).
  VAL calcula mal las preferencias del objetivo si el problema tiene una sección
  `(:constraints (and))` vacía.

## Procesos y eventos (iteración 6b)

Los dominios con procesos y eventos (PDDL+) van a un motor con el **tiempo discretizado**
(`pyplanner.grounding.discrete`), al estilo de ENHSP, DiNo o UPMurphi. Las tareas
temporales sin `:time` siguen en el motor de la 3b, que es más preciso, salvo las que este
no admite por sus efectos continuos (ritmos no lineales, como el repostaje de
generator-nonlinear, donde el caudal crece con el tiempo que lleva repostando): esas pasan
al motor discretizado, con una nota. Si tampoco las admite, el error es el de la 3b.

- **Pasos de la búsqueda:** las acciones instantáneas, el inicio y el fin de cada acción
  durativa, y una acción `_wait` que deja pasar δ unidades de tiempo (1 por defecto;
  `--delta` en la CLI, `delta` en `solve`). La espera se acorta si antes una acción
  durativa llega a una cota de su duración, para que pueda terminar justo a tiempo.
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
  espera sigue hasta completar δ. Sin esto, un evento se detectaba al final de la espera:
  en generator, un depósito quedaba con combustible negativo y el generador recibía de
  más, y la búsqueda acababa aprovechando ese error (5 de 8 instancias de
  generator-events; 8 de 8 con el instante exacto).
- **Invariantes:** las condiciones `over all` de las acciones durativas en curso se
  comprueban tras cada cambio (no en mitad de una espera).
- **Una acción instantánea** va seguida de una espera de ε = 0,01: VAL no admite dos
  acciones en el mismo instante, y durante ese ε los procesos siguen actuando, igual que en
  VAL.
- **Acciones durativas:** se compilan en un inicio (pone en marcha un reloj y fija las
  cotas de su duración), un proceso `a@run` (el reloj y los efectos continuos) y un fin
  (el reloj dentro de la duración). Las condiciones `over all` son invariantes.
- **Coste:** δ por espera y 2ε por acción (ε por el tiempo que pasa y ε por ser un paso
  más; sin el segundo, A* devolvía planes con decenas de acciones inútiles que hacían
  pasar el tiempo de ε en ε). La métrica que se informa es la duración del plan.
- **Heurística:** en la relajación, la espera es una acción con un efecto condicional por
  proceso (sus ritmos × δ); *h^hbd* cuenta las repeticiones si el ritmo es constante, como
  el reloj de una acción durativa. Los eventos son reglas relajadas de coste 0 sobre
  hechos. Lo que la relajación no ve (las invariantes) lo añade la 6c.
- **Novedad:** las variables que cambian de forma continua no cuentan. Toman un valor
  nuevo en cada espera, así que todo estado parecía nuevo y la tabla crecía sin fin (en
  generator-events p06, 85 000 átomos y 2 GB en 20 s). Limitar sus valores distintos (16,
  64, 256, 1024) en lugar de ignorarlas da resultados que dependen del límite sin ninguna
  tendencia.
- **Validación con un paso más fino:** el plan se vuelve a simular con δ/10. Si falla (una
  invariante que se rompe en mitad de una espera, por ejemplo), `solve` busca otra vez con
  δ/2, hasta 3 veces, dentro del límite de tiempo. Es más estricto que VAL, que solo
  comprueba las invariantes en los *happenings*: en generator p02, un plan que VAL daba
  por válido dejaba la gasolina en −1,08 entre dos *happenings*.
- **Sin plan:** agotar la búsqueda no demuestra nada (con un δ menor podría haber plan):
  termina con `failed`.
- **Límites** (error claro): literales iniciales temporizados, restricciones de
  trayectoria y preferencias con procesos y eventos; efectos condicionales que dependen de
  condiciones `over all` y efectos continuos condicionales en las acciones durativas.

## Heurística con procesos y eventos (iteración 6c)

La relajación ignora lo que empeora las condiciones, así que no ve las invariantes: para
ella, el generador puede funcionar 1000 unidades con 960 de combustible y basta con
esperar. Las búsquedas voraces esperaban hasta que la invariante se rompía y ya era tarde
para repostar. La 6c añade lo que falta (`pyplanner.heuristics.projection`):

- **Duraciones en la relajación.** Las cotas de la duración que no dependen de nada que
  cambie se comparan con el reloj tal cual, sin una función intermedia que la relajación
  no sabía cumplir. Asignar una constante (el reloj a 0 al empezar) ya no cuenta como
  lograr cualquier condición sobre esa variable (ver "Motor numérico"). Los relojes valen
  0 desde el estado inicial. Con esto, h(inicial) en generator pasa de 0,04 a unas 1000
  esperas.
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
  quedaba explorando los estados de antes: empezar y terminar un repostaje en ε da un
  estado nuevo cada vez (93 000 estados sin pasar del estado inicial). Por eso también se
  simula empezar cada acción durativa con invariante y duración mínima que la base da como
  útil, y el valor es el mayor de los faltantes.
- **Acciones útiles.** Las de la base no arreglan un faltante (para la relajación basta con
  esperar), y la búsqueda perezosa, que pone a los sucesores el valor del padre, bajaba
  esperando hasta que repostar ya no servía. Ahora, al expandir un estado con un faltante,
  las útiles son las que lo reducen. Hacen falta tantas simulaciones como acciones
  candidatas (las que cambian valores o hechos de las condiciones de procesos y eventos),
  así que se calculan solo en los estados que se expanden (`Heuristic.helpful`), no en
  todos los que se evalúan.
- **Rupturas que no duran.** La bisección de una espera solo ve los cambios que siguen al
  final del tramo, que en una espera de δ es lo previsto (la validación con δ/10 descubre
  el resto). La proyección avanza tramos de cientos de unidades. En generator-events, con
  los depósitos enganchados tarde, el combustible baja de 0 y se recupera (el caudal crece
  con el cuadrado del tiempo), y la proyección no lo veía. Ahora comprueba entre cada dos
  instantes en que se anula alguna de las condiciones numéricas en juego. Con un flujo
  exacto, una condición polinómica es un polinomio en el tiempo: sus coeficientes son sus
  derivadas a lo largo del flujo (`Flow.crossings`), que se calculan una vez por condición,
  y sus raíces, con fórmula hasta grado 2. Con Runge-Kutta, solo la bisección.
- **Rendimiento.** Sobre 1114e1d: el flujo redondea solo las variables que cambian, y los
  valores tras una acción o un evento, solo los que cambian (los demás son los mismos
  objetos, ya redondeados). La comprobación de cambios durante una espera evalúa una vez
  lo del estado de partida, y solo los procesos con condiciones numéricas (los hechos no
  cambian mientras pasa el tiempo). La búsqueda de generator-toricelli p09 con
  `bfws-ff-pref` pasa de 13,1 a 6,2 s con el perfilador.

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
| `hybrid` | cada lote en Numba o en la GPU según su tamaño, con un umbral calibrado al empezar | con lotes adaptativos y tareas grandes (ver "Lotes adaptativos") |

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
- `landmarks`: añade la heurística de landmarks (h_LM) con sus propias colas, que se
  alternan con las de `heuristic`, al estilo de LAMA (ver "Landmarks"); solo con GBFS sin
  novedad;
- `novelty`: anchura de la novedad (0, 1 o 2; ver "Búsqueda por novedad");
- `batch_size`; con `adaptive_batch`, el mínimo de un lote adaptativo que se duplica
  (hasta `max_batch_size`) tras `plateau_patience` lotes sin mejorar la mejor h (ver
  "Lotes adaptativos");
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
| `gbfs-ff-lm-pref` | GBFS + h_FF y landmarks en colas alternadas + acciones útiles | ver "Landmarks" |
| `lazy-ff-lm-pref` | la anterior, perezosa | |
| `lazy-ff-lm-pref-tb` | la anterior + exploración por tipos (semilla 1) | la mejor individual en las tareas clásicas |
| `gbfs-ff` | GBFS + h_FF | |
| `gbfs-add` | GBFS + h_add | |
| `wastar-add-3` | WA* (w = 3) + h_add | |
| `gbfs-goalcount-rnd` | GBFS + objetivos pendientes, desempate aleatorio | muy barata; aporta diversidad |
| `bfws-ff-pref` | BFWS: GBFS con prioridad (novedad, h_FF) + acciones útiles | la de 1 núcleo: la más robusta (resuelve todos los tiers) |
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

## Landmarks

Trabajo opcional (punto 2); código en `heuristics/landmarks.py` y `search/engine.py`. Al
estilo de LAMA (Richter y Westphal, 2010).

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
  no lo es nunca, todo va a Numba. Coste por estado de h_FF (µs) en la máquina de
  desarrollo (GTX 970):

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
- **En el portfolio aporta poco en estas pruebas** (la búsqueda perezosa ya resuelve
  zenotravel en pocos segundos), pero es la mejor sexta configuración: con las 7
  candidatas medidas en las 34 instancias, el mejor portfolio de 6 es el de 5 más el
  híbrido (93,8 s frente a 100,9 s), y WA* pasa al de 7. Con los landmarks se rehicieron
  los portfolios, y el híbrido sigue en el de 6 y el de 7 (ver "Portfolio").

## Portfolio

**Qué búsquedas se lanzan.** Depende del número de núcleos (`-j N`). Hay un portfolio
predefinido para cada número de núcleos de 1 a 7 (`PORTFOLIOS` en `search/config.py`), y
cada uno contiene al anterior:

| Núcleos | Búsquedas | Qué se añade |
|---------|-----------|--------------|
| 1 | `bfws-ff-pref` | BFWS, la más robusta: resuelve todos los tiers y da los mejores planes con preferencias; se ejecuta en el propio proceso |
| 2 | + `lazy-ff-lm-pref-tb` | la perezosa con landmarks y exploración por tipos, la mejor individual en las tareas clásicas |
| 3 | + `gbfs-goalcount-rnd` | la más barata y aleatoria |
| 4 | + `wastar-add-3` | WA*, que da los mejores planes con preferencias |
| 5 | + `lazy-ff-pref` | la perezosa sin landmarks |
| 6 | + `gbfs-ff-pref-hybrid` | lote adaptativo y evaluación híbrida CPU/GPU, el único miembro con GPU: casi empata con GBFS con exploración por tipos (una instancia menos, un 1 % más rápido) y se queda para las tareas grandes, donde la GPU rinde |
| 7 | + `gbfs-ff-pref-tb` | GBFS con exploración por tipos |

- Elegidos en el trabajo opcional de los landmarks con 475 instancias, clásicas y no
  clásicas (ver "Rendimiento medido"). Cada uno resuelve todos los tiers. Del de 2 al de
  5 son la mejor combinación posible de su tamaño; el de 6 y el de 7, por el miembro con
  GPU, quedan a una instancia de la mejor.
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
  [Preferencias y búsqueda anytime](#preferencias-y-búsqueda-anytime-iteración-5c)): se
  queda con el mejor plan dentro del límite de tiempo.
- Se usa `fork` cuando es seguro: sin CUDA, sin PyTorch cargado en el padre y con un
  solo hilo (también del sistema: un programa que embebe Python suele tener muchos). En
  cualquier otro caso se usa `spawn`, y cada hijo vuelve a importar el script principal:
  si este llama al planificador fuera de `if __name__ == "__main__":`, el hijo lanza un
  error claro en lugar de repetir el trabajo del script, y el padre informa de la causa.
- Si un miembro falla, informa con estado `ERROR` y un mensaje, pero no tumba el
  portfolio. Eso incluye no poder arrancarlo (antes, el error se quedaba en el hilo que
  arranca los de `spawn`) y terminar sin informar (se detecta al momento, con el código
  de salida).
- Los miembros de `spawn` leen la tarea de un fichero temporal, serializada una vez para
  todos. Antes viajaba por la tubería de arranque de cada hijo, y CPython mantiene abierto
  el otro extremo mientras escribe: si el hijo moría al arrancar y la tarea no cabía en la
  tubería (64 KB; slitherlink p02 ocupa 589 KB), el padre se quedaba esperando para
  siempre, aunque hubiera límite de tiempo.
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
pyplanner solve -t 30 dominio.pddl preferencias.pddl  # el mejor plan en 30 s
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
  - 4: no se ha encontrado plan: se ha llegado a un límite, o la búsqueda no puede
    demostrar que no lo haya (IW, o una tarea temporal en la que se han descartado estados
    repetidos con otra línea temporal; ver
    [Motor temporal](#motor-temporal-iteración-3b)).

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

### Iteración 4: predicados derivados

psr-middle p01 y p02 (IPC 2004): unos 400 axiomas en un solo estrato y 28 acciones;
grounding en 0,25 s. Todas las configuraciones predefinidas los resuelven en
milisegundos, salvo IW(1), que es incompleta. Los tiempos de la CLI en los tiers 1 y 2 son
los mismos que antes de añadir los axiomas (medidos en las mismas condiciones).

### Iteración 4: literales iniciales temporizados

satellite p01 y p02 (IPC 2004, ventanas de visibilidad de las antenas). p01 se resuelve
en milisegundos. En p02 las ventanas restringen de verdad: con 60 s y 2,5 GB por búsqueda,
`bfws-ff` tarda 0,6 s, `bfws-ff-pref-hybrid` 2,6 s, `gbfs-ff-pref` 3,7 s y
`bfws-ff-pref` 8,2 s. `gbfs-ff`, `lazy-ff-pref`, `gbfs-goalcount-rnd` y las A* no
terminan. El portfolio por defecto tarda unos 12 s. La mayoría de los nodos están entre el
cierre de una antena y el de la otra, donde la heurística no ve que queda poco tiempo.

**Plazos en la heurística** (parte A), medidos en `benchmarks/timed-initial-literals/
satellite` (p01–p20, 60 s y 2,5 GB por búsqueda, planes validados con VAL):

| Configuración | Sin plazos | Con plazos |
|---------------|------------|------------|
| `gbfs-ff-pref` | 4 (973 s) | 5 (917 s) |
| `bfws-ff-pref` | 3 (1035 s) | 4 (1028 s) |
| `bfws-ff` | 3 (1039 s) | 4 (1007 s) |
| `lazy-ff-pref` | 3 (1025 s) | 6 (859 s) |
| `gbfs-ff-pref-tb` | 4 (999 s) | 4 (990 s) |
| Alguna de las cinco | 5 (p01–p05) | 8 (además p06, p10 y p14) |

- La comprobación descubre muchos callejones (en p07, 24 000 frente a 3 000 sin plazos) y
  cuesta más o menos lo mismo que h_FF compilada.
- No mejora todos los casos pequeños: cambia el recorrido de la búsqueda, y en algunos
  (`bfws-ff` en p02, de 0,8 a 6,1 s) va peor.
- Por encima de p06 las búsquedas se ahogan: cada expansión genera unos 270 sucesores en
  p07 y h_FF no orienta con ventanas. Las perezosas agotan los 2,5 GB. Lo que queda por
  ganar es de la parte B (plan relajado temporal), que está en el trabajo opcional.

### Iteración 5c: preferencias y búsqueda anytime

storage con preferencias cualitativas (IPC 2006, `benchmarks/preferences/storage`,
p01–p20), 60 s y 2,5 GB por búsqueda, 3 procesos. Todos los planes son válidos para VAL y
su métrica coincide con la de VAL:

| | `lazy-ff-pref` | `bfws-ff-pref` |
|---|---|---|
| Con plan | 20 de 20 | 20 de 20 |
| Óptimo demostrado | p01–p03 | p01–p03 |
| Métrica total | 30 233 | 31 515 |

- **Grounding.** La restricción de cajas incompatibles en zonas contiguas es un `forall`
  sobre dos cajas y dos zonas: en p20, 2,8 millones de instancias, casi todas triviales.
  Instanciarlas todas agotó la memoria de WSL en la primera medida. Con el join y el
  descarte de las triviales quedan 37 000, y el grounding de p20 pasa de 197 s y 2,7 GB a
  6 s y 350 MB. El índice del join acelera también el grounding de los demás tiers
  (slitherlink p02, de 0,37 a 0,19 s) con el mismo resultado.
- **Primer plan e incumbente.** Con los costes reales desde el principio, desde p04
  `lazy-ff-pref` no encontraba plan en 60 s y `bfws-ff-pref` solo hasta p07. La primera
  búsqueda con coste unitario da siempre un plan, pero el trivial (todo incumplido), y WA*
  no lo mejora en 60 s. El incumbente sí: en p03–p06, p08 y p10, la métrica total pasa
  de 1987 a 1309 (`lazy-ff-pref`) y a 985 (`bfws-ff-pref`); p04, de 81 a 5.
- **Lo que queda.** En las instancias grandes la métrica sigue alta: cada estado tiene
  miles de hechos (las marcas de las restricciones-preferencia) y las búsquedas expanden
  poco (`bfws-ff-pref`, unas 600 expansiones en p20). En p10, la mitad del tiempo de
  `bfws-ff-pref` es la novedad (un OR de enteros de miles de bits por átomo) y un tercio
  h_FF compilada.
- **Las mismas búsquedas en los tiers 1 a 5** (sin preferencias) cuestan lo mismo que
  antes de la 5c (+1 %, dentro del ruido): la cota, la penalización y el incumbente se
  comprueban con un solo booleano por sucesor cuando no se usan.

### Iteración 6c: heurística con procesos y eventos

Colecciones PDDL+ de `benchmarks/`, 60 s y 2 GB por búsqueda, 4 procesos. Instancias
resueltas y tiempo total (las no resueltas cuentan 60 s), antes (1114e1d, la 6b) y
después de la 6c. Todos los planes de la 6c son válidos para VAL:

| Colección | `bfws-ff-pref` 6b | `bfws-ff-pref` 6c | `lazy-ff-pref` 6b | `lazy-ff-pref` 6c |
|---|---|---|---|---|
| car-nodrag (10) | 10, 36,0 s | 10, 26,1 s | 10, 28,7 s | 10, 19,6 s |
| generator-events (8) | 8, 31,0 s | 8, 9,8 s | 4, 263,8 s | 8, 9,9 s |
| generator-nonlinear (8) | 8, 75,3 s | 8, 7,5 s | 4, 271,6 s | 8, 7,5 s |
| generator-toricelli (9) | 5, 342,0 s | 9, 26,2 s | 1, 482,2 s | 9, 15,7 s |
| generator-linear (8, motor de la 3b) | 8, 8,5 s | 8, 7,6 s | 8, 59,1 s | 8, 57,3 s |

- **generator**: con la proyección, las búsquedas recorren las 1000 esperas casi sin
  desviarse (unas 1000 expansiones). En generator-toricelli p08, el primer plan de
  `bfws-ff-pref` falla con δ/10 y se encuentra otro con δ = 0,5.
- **car-nodrag** no tiene invariantes: hace las mismas búsquedas, un 27-32 % más rápidas
  por los cambios de rendimiento de las esperas.
- **generator-linear** va al motor de la 3b y hace las mismas búsquedas; la lentitud de
  `lazy-ff-pref` ya estaba.
- **Las mismas búsquedas en los tiers 1 a 5** (52 búsquedas con `lazy-ff-pref` y
  `bfws-ff-pref`, el mejor de 3): 51 expanden lo mismo, y la otra (satellite p02 con
  `lazy-ff-pref`) llega al límite de tiempo en las dos versiones. El coste es el mismo
  (mediana +0,9 %; en slitherlink p02, 7,3-7,6 s en las dos con varias medidas).

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

### Trabajo opcional: landmarks y portfolios

Dos conjuntos, 60 s y 1,5 GB por búsqueda, con límite duro de memoria por proceso:

- **Clásicas:** 340 instancias, las 20 primeras de 17 colecciones: las 14 de *satisficing*
  de la IPC 2011 y recharging-robots, ricochet-robots y slitherlink de la IPC 2023.
- **No clásicas:** 135 instancias, los tiers y las colecciones de `benchmarks/`, sin
  repetir, más driverlog y zenotravel temporales hasta la 17.

Las clásicas y las temporales grandes no están en git (ver
[benchmarks/README.md](../benchmarks/README.md)). Instancias resueltas y tiempo total (las no
resueltas cuentan 60 s):

| Configuración | Clásicas | No clásicas | Total | Tiempo |
|---------------|----------|-------------|-------|--------|
| `lazy-ff-lm-pref-tb` | **203** | 114 | **317** | 12984 s |
| `gbfs-ff-lm-pref` | 193 | 116 | 309 | 13149 s |
| `bfws-ff-pref` | 187 | 117 | 304 | 14120 s |
| `lazy-ff-lm-pref` | 189 | 112 | 301 | 13847 s |
| `gbfs-ff-pref-tb` | 156 | 117 | 273 | 15484 s |
| `gbfs-ff-pref-hybrid` | 149 | 116 | 265 | 15976 s |
| `lazy-ff-pref-tb` | 137 | 116 | 253 | 16090 s |
| `lazy-ff-pref` | 127 | 118 | 245 | 16749 s |
| `wastar-add-3` | 126 | 82 | 208 | 18638 s |
| `gbfs-goalcount-rnd` | 120 | 72 | 192 | 18074 s |

- **Los landmarks ganan mucho en las clásicas.** La búsqueda perezosa resuelve 75
  instancias más y pierde 13 (con exploración por tipos, 73 más y 7 menos), sobre todo en
  openstacks (20), barman (17), visit-all (16–18) y parking (7–11). BFWS sigue siendo la
  mejor en floor-tile, no-mystery, ricochet-robots, slitherlink y transport.
- **En las no clásicas, casi todas resuelven casi todo.** Trucks p01 (tier 5) solo lo
  resuelven en 60 s `bfws-ff-pref`, `gbfs-ff-lm-pref` y el miembro con GPU, así que cada
  portfolio necesita uno de los tres.
- **La GPU no destaca en las temporales grandes.** Driverlog: 122 s frente a 104 s de BFWS;
  zenotravel: 93 s frente a 70 s de la búsqueda perezosa.

**Portfolios.** Por cada número de núcleos, la combinación con más instancias resueltas y
menos tiempo (el del miembro más rápido de cada instancia), con la condición de resolver
todos los tiers. Storage es la suma de las métricas de las preferencias (menos es mejor),
con el mejor plan de los miembros en cada problema:

| Núcleos | Antes: resueltas · tiempo · storage | Ahora: resueltas · tiempo · storage |
|---------|-------------------------------------|-------------------------------------|
| 1 | 304 · 14120 s · 20117 (2 sin plan) | igual |
| 2 | 314 · 13551 s · 28667 | 352 · 11357 s · 27094 |
| 3 | (usaba el de 2) | 366 · 10331 s · 27094 |
| 4 | 357 · 10911 s · 25984 | 371 · 10022 s · 20172 |
| 5 | 360 · 10638 s · 25975 | 375 · 9758 s · 20053 |
| 6 | 361 · 10388 s · 25975 | 376 · 9499 s · 20053 |
| 7 | 369 · 9976 s · 20041 | 378 · 9356 s · 20045 |

- La cadena que empieza por BFWS es la mejor combinación de cada tamaño, del 2 al 7
  (377 con 6 núcleos y 379 con 7, sin el miembro con GPU). Con
  1 núcleo, `gbfs-ff-lm-pref` resolvería 5 instancias más, pero con planes mucho peores en
  storage (33586).
- El miembro con GPU entra en el de 6 y el de 7 en lugar de `gbfs-ff-pref-tb` y
  `lazy-ff-lm-pref`: una instancia menos en cada uno y un 1 % más rápidos.
- Es una sola muestra: `lazy-ff-lm-pref-tb` desempata al azar (semilla 1).

**Problemas encontrados por el camino:**

- **Grounding sin memoria.** 9 instancias fallan en todas las configuraciones al
  instanciar: 8 de recharging-robots y scanalyzer p18. Recharging-robots p05 tiene 1,39
  millones de acciones (casi todas de `recharge`, por todos los pares de robots, lugares y
  niveles de batería) y tarda 391 s y 5,7 GB solo en el grounding: unos 4 KB por acción.
  Folding y labyrinth (IPC 2023) tardan 118 s y 33 s en el grounding. Pendiente.
- **VAL sin memoria.** El script de medida validaba en el proceso de la búsqueda, y VAL
  heredaba su límite duro: en storage p20 (2,3 GB) moría sin veredicto y el plan contaba
  como inválido. Ahora VAL se ejecuta al final, sin búsquedas en marcha, y si termina sin
  veredicto el plan queda sin validar. Todos los planes de estas medidas son válidos.
- **Memoria.** WSL se cayó durante la primera medida; el único proceso sin límite duro
  era el miembro con GPU. Ahora el script le pone `RLIMIT_DATA`, y la memoria usada no
  pasó de 8,4 GB de 16.
