# PyPlanner

Planificador PDDL en Python.

## Estructura

```
src/pyplanner/       Código del planificador
tests/               Tests (pytest)
dominios/            Benchmarks PDDL por tiers (ver dominios/README.md)
docs/                Documentación y notas
```

## Puesta en marcha

Requiere Python 3.12 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Validador de planes (VAL)

Los tests validan los planes con [VAL](https://github.com/KCL-Planning/VAL). Se compila en
local (necesita `git`, `cmake` y `g++`):

```bash
scripts/build_val.sh          # deja el ejecutable en tools/bin/Validate
```

## Uso

```bash
pyplanner dominio.pddl problema.pddl
# o bien
python -m pyplanner dominio.pddl problema.pddl
```

## Desarrollo

```bash
pytest               # tests
ruff check .         # linter
ruff format .        # formateo
mypy                 # comprobación de tipos
```
