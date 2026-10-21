# PyPlanner

[English](README.md) · **Español**

Planificador PDDL en Python basado en **búsqueda heurística hacia delante**, pensado para
integrarse en otros proyectos mediante JSON. No pretende batir el estado del arte, sino
ser un planificador al día que se pueda entender, controlar y modificar.

## Lenguaje soportado

PDDL de STRIPS a PDDL+, incluido PDDL 3:

| Versión de PDDL | Qué incluye |
|-----------------|-------------|
| STRIPS | acciones con precondiciones, añadidos y borrados |
| PDDL 1.2 | subconjunto ADL: tipos, precondiciones negativas y disyuntivas, cuantificadores, igualdad y efectos condicionales |
| PDDL 2.1 | fluentes numéricos (con `:action-costs`) y acciones durativas, con efectos continuos |
| PDDL 2.2 | predicados derivados y literales iniciales temporizados |
| PDDL 3 | restricciones de trayectoria, preferencias (también en tareas temporales y con procesos) y funciones de objetos |
| PDDL+ | procesos y eventos |

**Modo estricto:** toda construcción tiene que tener su requisito declarado en
`:requirements`. Una entrada que usa algo todavía no soportado se rechaza con un error
claro.

## Características

- **Parser en dos fases** unidas por JSON: PDDL → JSON (con un esquema versionado) → objetos
  internos. Otro sistema puede pasarle las tareas directamente en JSON
  ([formato](docs/es/formato-json.md)).
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
  pesos de la métrica (también las que se maximizan), con búsqueda *anytime* que devuelve
  el mejor plan dentro del límite de tiempo, también en tareas temporales y con procesos y
  eventos; y funciones de objetos (PDDL 3.1) compiladas a predicados.
- **PDDL+:** procesos y eventos con el tiempo discretizado al estilo de ENHSP (una espera
  de δ, 1 por defecto), integración exacta con series de Taylor (Runge-Kutta si la serie
  no termina), eventos en su instante exacto y validación del plan con un paso diez veces
  más fino (si falla, se busca otra vez con δ/2). Admite acciones durativas (con efectos
  condicionales y continuos condicionales), literales iniciales temporizados,
  restricciones de trayectoria y preferencias. Las tareas temporales con efectos
  continuos no lineales también van a este motor.
- **Heurísticas:** blind, goalcount, h_max, h_add y h_FF con acciones útiles; con literales
  iniciales temporizados, h_TFF (un plan relajado que respeta los tiempos y las ventanas);
  con procesos, además, la proyección de las invariantes de las acciones en curso.
- **Búsquedas:** GBFS, A* y WA*, con evaluación inmediata o perezosa, acciones útiles,
  exploración por tipos, búsqueda por novedad (IW y BFWS), landmarks en colas alternadas
  (al estilo de LAMA), evaluación por lotes y búsqueda *anytime* (con preferencias).
  También *enforced hill-climbing* (FF) y paseos aleatorios (Arvand), fuera de los
  portfolios porque no resuelven nada que no resuelvan ya.
- **Rendimiento:** heurísticas y generación de sucesores compiladas con Numba, con
  resultados idénticos a la implementación de referencia en Python. También hay evaluación
  por lotes en GPU con PyTorch.
- **Portfolio paralelo:** conjuntos de búsquedas predefinidos para 1 a 7 núcleos;
  gana la primera que encuentra un plan (con preferencias, el mejor plan que encuentren
  todas en el tiempo dado, compartiendo la cota).
- **Planes** en formato IPC (el que lee VAL) o en JSON.

## Instalación

Requiere Python 3.12 o superior.

```bash
git clone https://github.com/Leontes/PyPlanner.git
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

## Idioma

La ayuda de la línea de comandos, los mensajes y los errores del planificador están en
español. Los identificadores (subcomandos, opciones, nombres de las configuraciones,
claves del JSON) están en inglés.

## Uso

### Desde la terminal

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

### Como librería

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
- **Memoria:** cada miembro del portfolio tiene un límite duro (`RLIMIT_DATA`) algo por
  encima de su parte de `memory_limit`; al pasarlo, su búsqueda termina con
  `limit_reached`. El proceso que llama a `solve` no se limita: el grounding y una búsqueda
  con una sola configuración solo tienen el límite blando. La CLI sí se limita a sí misma.
- **Límite de tiempo:** sin `time_limit`, cada búsqueda sigue hasta encontrar plan,
  demostrar que no lo hay o agotar la memoria; en un problema difícil eso puede ser mucho
  tiempo. Desde otro programa conviene pasarlo siempre.

## Estructura

| Ruta | Qué hay |
|------|---------|
| `src/pyplanner/pddl/` | fase 1: lexer, parser PDDL → JSON y conversor inverso |
| `src/pyplanner/schema/` | esquema JSON (contrato entre fases y con otros sistemas) |
| `src/pyplanner/loader/` | fase 2: JSON → AST, con validación |
| `src/pyplanner/grounding/` | instanciación, normalización ADL y tarea instanciada |
| `src/pyplanner/heuristics/` | heurísticas: referencia en Python, Numba y PyTorch |
| `src/pyplanner/search/` | búsqueda, configuraciones, portfolios y generación de sucesores |
| `src/pyplanner/portfolio/` | ejecución paralela de configuraciones |
| `src/pyplanner/plan/` | planes: formatos IPC y JSON, validación interna |
| `src/pyplanner/planner.py` | fachada: de dominio y problema a plan |
| `src/pyplanner/cli.py` | interfaz de línea de comandos |
| `tests/` | tests (`pytest`); `dataset.toml`: los problemas de prueba |
| `vendor/PDDLDomains/` | submódulo: el catálogo de dominios PDDL de los problemas de prueba |
| `scripts/` | compilación de VAL y medidas de rendimiento (`benchmark.py`) |
| `docs/es/`, `docs/en/` | documentación, en español y en inglés |

`tools/` (VAL compilado en local), `salidas/` (salidas locales) y `privado/` (documentos de
uso privado) no están en git.

## Documentación

- [Formato JSON](docs/es/formato-json.md): el contrato JSON de entrada y salida.
- [Motor de búsqueda](docs/es/motor-busqueda.md): cómo funciona el motor y sus
  configuraciones.
- [Problemas de prueba](docs/es/pruebas.md): los problemas de los tests, del catálogo
  PDDLDomains y agrupados por versión de PDDL.

## Desarrollo

```bash
git clone --recursive https://github.com/Leontes/PyPlanner.git   # con el catálogo
git submodule update --init vendor/PDDLDomains                    # o, en un clon ya hecho
pip install -e ".[dev,fast]"
scripts/build_val.sh     # compila VAL en tools/bin/Validate (necesita git, cmake y g++)
pytest                   # tests
ruff check . && ruff format --check . && mypy
```

- **Problemas de prueba:** los tests usan problemas del catálogo
  [PDDLDomains](https://github.com/Leontes/PDDLDomains), en el submódulo
  `vendor/PDDLDomains` y fijado en un commit, con al menos dos colecciones por versión de
  PDDL, de STRIPS a PDDL+ ([problemas de prueba](docs/es/pruebas.md)). Los tests resuelven
  todos con cada portfolio y validan los planes con
  [VAL](https://github.com/KCL-Planning/VAL). Si VAL no está compilado, esos tests se
  saltan.
- **Medir el rendimiento:** `scripts/benchmark.py` mide configuraciones sobre una colección
  o un dataset del catálogo, con límites de tiempo y de memoria por búsqueda, y valida los
  planes con VAL (`python scripts/benchmark.py --help`).
- **Equivalencia:** las implementaciones con Numba y PyTorch tienen tests que comprueban
  que dan exactamente los mismos resultados que la referencia en Python.

## Licencia

El código y la documentación se distribuyen con la [licencia MIT](LICENSE) (© 2026 José
Segura). Los dominios y problemas del catálogo PDDLDomains (submódulo) tienen sus propias
condiciones: ver su `LICENCIAS.md`.
