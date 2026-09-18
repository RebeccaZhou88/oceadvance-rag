@echo off
REM ===== OCECopilotAdvanceRAG Backend Launcher =====
REM Double-click this file, or run "runbackend.bat" from cmd/PowerShell
REM Frontend is served automatically by uvicorn -> http://127.0.0.1:8000/

setlocal

cd /d "%~dp0backend"

REM src layout: tell Python where to find the "app" package
set PYTHONPATH=src

echo [runbackend] PYTHONPATH=%PYTHONPATH%
echo [runbackend] cwd=%CD%
echo [runbackend] Starting uvicorn ... open http://127.0.0.1:8000/

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

endlocal
