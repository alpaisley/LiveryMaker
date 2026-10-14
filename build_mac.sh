#!/bin/bash
# Builds "Livery Maker.app" in ./dist.  Run once from this folder:   bash build_mac.sh
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v $c >/dev/null 2>&1; then
    v=$($c -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')
    if [ "$v" -ge 310 ]; then PY=$c; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/macos/ (or: brew install python), then run this again."
  exit 1
fi
echo "Using $($PY --version)"

$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "Installing libraries (first time only, a few minutes)..."
pip install -r requirements.txt pyinstaller

echo "Building the app..."
rm -rf build dist 2>/dev/null || { sleep 1; rm -rf build dist 2>/dev/null; } || true   # PyInstaller --noconfirm replaces them anyway
pyinstaller --noconfirm --windowed --name "Livery Maker" \
  --icon livery_maker/icon.png \
  --add-data "livery_maker/fonts:livery_maker/fonts" \
  --add-data "livery_maker/icon.png:livery_maker" \
  --add-data "livery_maker/engine/stations:livery_maker/engine/stations" \
  --collect-submodules psd_tools \
  --collect-submodules OpenGL.platform \
  --hidden-import OpenGL.arrays.numpymodule \
  --osx-bundle-identifier uk.co.alypeely.liverymaker \
  launcher.py

echo
echo "Done:  $(pwd)/dist/Livery Maker.app"
echo "Drag it into Applications if you like, then double-click to open."
