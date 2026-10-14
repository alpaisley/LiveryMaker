@echo off
rem Builds "Livery Maker.exe" in .\dist\Livery Maker\   Double-click this file, or run it from a Command Prompt.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
  )
)
if not defined PY (
  echo.
  echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/windows/
  echo and tick "Add python.exe to PATH" on the first screen of the installer. Then run this again.
  echo.
  pause
  exit /b 1
)
for /f "delims=" %%v in ('%PY% --version') do echo Using %%v

if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment in .venv ...
  %PY% -m venv .venv || goto :fail
)
call ".venv\Scripts\activate.bat" || goto :fail

echo Installing libraries (first time only, a few minutes) ...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt pyinstaller || goto :fail

echo Building the app ...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller --noconfirm --windowed --name "Livery Maker" ^
  --icon livery_maker\icon.png ^
  --add-data "livery_maker\fonts;livery_maker\fonts" ^
  --add-data "livery_maker\icon.png;livery_maker" ^
  --add-data "livery_maker\engine\stations;livery_maker\engine\stations" ^
  --collect-submodules psd_tools ^
  --collect-submodules OpenGL.platform ^
  --hidden-import OpenGL.arrays.numpymodule ^
  launcher.py || goto :fail

echo.
echo Done:  %CD%\dist\Livery Maker\Livery Maker.exe
echo Copy the whole "Livery Maker" folder wherever you like (keep the .exe with its _internal folder),
echo then double-click "Livery Maker.exe". Right-click it and choose "Send to > Desktop" for a shortcut.
echo.
pause
exit /b 0

:fail
echo.
echo Something went wrong. The messages above say what. Fix it and run this again.
echo.
pause
exit /b 1
