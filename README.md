# PyPlanner

Planificador PDDL en Python basado en **búsqueda heurística hacia delante**, pensado para
integrarse en otros proyectos mediante JSON. No pretende batir el estado del arte, sino
ser un planificador al día que se pueda entender, controlar y modificar.

## Estado

El planificador crece por iteraciones, y cada una amplía el lenguaje PDDL soportado (ver
[docs/ROADMAP.md](docs/ROADMAP.md)):

| Versión | Iteración | Lenguaje | Estado |
|---------|-----------|----------|--------|
| v0.1 | 1 | STRIPS | ✅ |
| v0.2 | 2 | PDDL 1.2, subconjunto ADL: tipos, precondiciones negativas y disyuntivas, cuantificadores, igualdad y efectos condicionales | ✅ |
| v0.3 | 3 | PDDL 2.1: fluentes numéricos (con `:action-costs`) y acciones durativas | ✅ |
| v0.4 | 4 | PDDL 2.2: predicados derivados y literales iniciales temporizados | ✅ |
| v0.5 | 5 | PDDL 3: restricciones de trayectoria, preferencias y funciones de objetos | ✅ |
| v0.6 | 6 | PDDL+: procesos y eventos | ✅ |

Las seis iteraciones de la planificación original están completas. Queda como trabajo
opcional: técnicas de búsqueda adicionales, preferencias en tareas temporales, un límite
de memoria duro y un estudio de la integración de HTN.

**Modo estricto:** toda construcción tiene que tener su requisito declarado en
`:requirements`. Una entrada que usa algo todavía no soportado se rechaza con un error
claro.

## Características

- **Parser en dos fases** unidas por JSON: PDDL → JSON (con un esquema versionado) → objetos
  internos. Otro sistema puede pasarle las tareas directamente en JSON
  ([formato](docs/formato-json.md)).
- **Grounding** con análisis de alcanzabilidad relajada y normalización ADL (DNF,
  cuantificadores expandidos, efectos condicionales).
- **Fluentes numéricos y costes de acción:** estado con valores numéricos, funciones
  estáticas sustituidas, métrica como coste de acción cuando es posible y heurísticas
  numéricas (*h^hbd*).
- **Planificación temporal:** acciones durativas divididas en inicio y fin, órdenes al
  estilo de POPF sobre una red temporal y programación lineal (al estilo de COLIN) para
  los efectos continuos.
- **PDDL 2.2:** predicados derivados (axiomas evaluados por estratos en cada estado) y
  literales iniciales temporizados, con plazos y ventanas en la heurística para descartar
  los estados que ya no llegan a tiempo.
- **PDDL 3:** restricciones de trayectoria (los diez operadores modales, también con
  tiempo) vigiladas por monitores en el motor; preferencias compiladas a costes con los
  pesos de la métrica, con búsqueda *anytime* que devuelve el mejor plan dentro del límite
  de tiempo; y funciones de objetos (PDDL 3.1) compiladas a predicados.
- **PDDL+:** procesos y eventos con el tiempo discretizado al estilo de ENHSP (una espera
  de δ, 1 por defecto), integración exacta con series de Taylor (Runge-Kutta si la serie
  no termina), eventos en su instante exacto y validación del plan con un paso diez veces
  más fino (si falla, se busca otra vez con δ/2). Las tareas temporales con efectos
  continuos no lineales también van a este motor.
- **Heurísticas:** blind, goalcount, h_max, h_add y h_FF con acciones útiles; con procesos,
  además, la proyección de las invariantes de las acciones en curso.
- **Búsquedas:** GBFS, A* y WA*, con evaluación inmediata o perezosa, acciones útiles,
  exploración por tipos, búsqueda por novedad (IW y BFWS), evaluación por lotes y búsqueda
  *anytime* (con preferencias).
- **Rendimiento:** heurísticas y generación de sucesores compiladas con Numba, con
  resultados idénticos a la implementación de referencia en Python. También hay evaluación
  por lotes en GPU con PyTorch.
- **Portfolio paralelo:** conjuntos de búsquedas predefinidos para 1, 2, 4, 5, 6 y 7 núcleos;
  gana la primera que encuentra un plan (con preferencias, el mejor plan que encuentren
  todas en el tiempo dado, compartiendo la cota).
- **Planes** en formato IPC (el que lee VAL) o en JSON.

## Instalación

Requiere Python 3.12 o superior.

```bash
git clone git@github.com:Leontes/PyPlanner.git
cd PyPlanner
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[fast]"        # recomendado: incluye Numba
```

| Extra | Qué añade | Cuándo |
|-------|-----------|--------|
| (ninguno) | el planificador en Python, con jsonschema y SciPy (programación lineal del motor temporal) | siempre funciona |
| `fast` | Numba: heurísticas y sucesores compilados, unas 10 veces más rápido | recomendado |
| `gpu` | PyTorch: evaluación de heurísticas por lotes en CPU o GPU | experimentación con lotes grandes |
| `dev` | pytest, ruff, mypy | desarrollo |

Para GPUs antiguas (Maxwell, como la GTX 970) solo sirven las builds de PyTorch para
CUDA 12.6:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Uso desde la terminal

```bash
# Buscar un plan: el plan sale por la salida estándar y las estadísticas por la de error
pyplanner solve dominio.pddl problema.pddl                 # portfolio de 6 núcleos (o menos)
pyplanner solve -j 1 -t 60 dominio.pddl problema.pddl       # un solo núcleo, 60 s
pyplanner solve -c astar-hmax dominio.pddl problema.pddl    # plan óptimo
pyplanner solve -f json -o plan.json dominio.pddl problema.pddl
pyplanner solve -t 30 dominio.pddl preferencias.pddl       # con preferencias: el mejor en 30 s
pyplanner solve --delta 0.5 dominio.pddl problema.pddl     # PDDL+: paso de tiempo de 0,5
pyplanner configs                                            # configuraciones y portfolios

# Fase 1: PDDL → JSON
pyplanner parse dominio.pddl -o dominio.json

# Fases 1 y 2: comprueba que un dominio y un problema son válidos (.pddl o .json)
pyplanner check dominio.pddl problema.pddl
```

Códigos de salida de `solve`:

- 0: plan encontrado;
- 2: entrada no válida;
- 3: se ha demostrado que no hay plan;
- 4: no se ha encontrado plan (dentro de los límites, o con una búsqueda que no puede
  demostrar que no lo haya, como IW, una temporal que descarta estados repetidos o una con
  el tiempo discretizado).

## Uso como librería

```python
from pyplanner.frontend import read_domain, read_problem
from pyplanner.planner import solve
from pyplanner.search import portfolio_for_cores

domain = read_domain("domain.pddl")  # también acepta .json
problem = read_problem("p01.pddl", domain)
result = solve(domain, problem, portfolio_for_cores(2), time_limit=60)
if result.plan is not None:
    print(result.plan.to_ipc())  # o result.plan.to_dict() para el JSON
```

Para integrarlo en otro sistema:

- **Entrada en JSON:** `pyplanner.loader.load_domain` y `load_problem` reciben el JSON
  directamente y lo validan (estructura, requisitos y semántica).
- **Salida en JSON:** el plan es un documento con `"kind": "plan"`.
- **Errores:** todos los errores de entrada heredan de `pyplanner.errors.PDDLError`.
- **Procesos:** el portfolio crea procesos, así que el código que llama a `solve` debe ir
  dentro de `if __name__ == "__main__":`. Con el arranque `spawn` (si PyTorch está cargado,
  si el proceso tiene varios hilos o para los miembros con GPU), cada proceso vuelve a
  importar el script principal; sin esa guarda, el planificador lo detecta y devuelve
  `ERROR` con la causa.
- **Python embebido** (en un programa en C++, por ejemplo): `spawn` lanza
  `sys.executable`, que entonces es el programa anfitrión. Antes de llamar a `solve`, hay
  que indicarle el intérprete con `multiprocessing.set_executable(...)`.
- **Límite de tiempo:** sin `time_limit`, cada búsqueda sigue hasta encontrar plan,
  demostrar que no lo hay o agotar la memoria; en un problema difícil eso puede ser mucho
  tiempo. Desde otro programa conviene pasarlo siempre.

## Documentación

- [docs/ROADMAP.md](docs/ROADMAP.md): iteraciones, decisiones de diseño y repertorio de
  técnicas.
- [docs/formato-json.md](docs/formato-json.md): el contrato JSON de entrada y salida.
- [docs/motor-busqueda.md](docs/motor-busqueda.md): cómo funciona el motor, sus
  configuraciones y las medidas de rendimiento.
- [dominios/README.md](dominios/README.md): los benchmarks y su procedencia.
- [benchmarks/README.md](benchmarks/README.md): colecciones más grandes para medir el
  rendimiento (`scripts/benchmark.py`), fuera de los tests.

## Estructura

```
src/pyplanner/
  pddl/         fase 1: lexer, parser PDDL → JSON y conversor inverso
  schema/       esquema JSON (contrato entre fases y con otros sistemas)
  loader/       fase 2: JSON → AST, con validación
  grounding/    instanciación, normalización ADL y tarea instanciada
  heuristics/   heurísticas: referencia en Python, Numba y PyTorch
  search/       búsqueda, configuraciones, portfolios y generación de sucesores
  portfolio/    ejecución paralela de configuraciones
  plan/         planes: formatos IPC y JSON, validación interna
  planner.py    fachada: de dominio y problema a plan
  cli.py        interfaz de línea de comandos
tests/          tests (pytest)
dominios/       benchmarks PDDL organizados por tiers
docs/           documentación
scripts/        utilidades (compilar VAL)
```

## Desarrollo

```bash
pip install -e ".[dev,fast]"
scripts/build_val.sh     # compila VAL en tools/bin/Validate (necesita git, cmake y g++)
pytest                   # tests
ruff check . && ruff format --check . && mypy
```

- **Tiers:** los benchmarks de `dominios/` están organizados en tiers, uno por iteración.
  Los tests resuelven todos los problemas de los tiers soportados con cada portfolio y
  validan los planes con [VAL](https://github.com/KCL-Planning/VAL). Si VAL no está
  compilado, esos tests se saltan.
- **Equivalencia:** las implementaciones con Numba y PyTorch tienen tests que comprueban
  que dan exactamente los mismos resultados que la referencia en Python.
