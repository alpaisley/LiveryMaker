@echo off
rem Run Livery Maker straight from the source (after build_windows.bat has created .venv once).
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python launcher.py
