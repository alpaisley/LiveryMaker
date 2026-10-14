#!/bin/bash
# Run Livery Maker straight from the source (after build_mac.sh has created .venv once).
cd "$(dirname "$0")"
source .venv/bin/activate
python launcher.py
