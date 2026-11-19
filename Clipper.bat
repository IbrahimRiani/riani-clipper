@echo off
rem Claude Clipper: abre la interfaz en tu navegador (solo en tu PC).
title Claude Clipper
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No encuentro el entorno .venv. Sigue los pasos de instalacion del README.
  pause
  exit /b 1
)
echo Abriendo Claude Clipper en tu navegador... (cierra esta ventana para apagarlo)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
