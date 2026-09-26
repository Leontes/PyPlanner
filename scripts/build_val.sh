#!/usr/bin/env bash
# Descarga y compila VAL (https://github.com/KCL-Planning/VAL), el validador de planes PDDL.
# Deja el ejecutable en tools/bin/Validate. Requisitos: git, cmake y g++.
#
# Uso: scripts/build_val.sh [ref]   (ref: rama, tag o commit; por defecto, master)
set -euo pipefail

REF="${1:-master}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/tools/VAL"
BUILD="$SRC/build"
BIN="$ROOT/tools/bin"

if [ ! -d "$SRC/.git" ]; then
    git clone https://github.com/KCL-Planning/VAL.git "$SRC"
fi
git -C "$SRC" fetch --quiet origin
git -C "$SRC" checkout --quiet "$REF"

cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --target Validate -j"$(nproc)"

# VAL enlaza con rpath $ORIGIN: libVAL.so tiene que estar junto al ejecutable.
mkdir -p "$BIN"
cp "$BUILD/bin/Validate" "$BUILD/bin/libVAL.so" "$BIN/"
echo "VAL compilado en $BIN/Validate (commit $(git -C "$SRC" rev-parse --short HEAD))"
