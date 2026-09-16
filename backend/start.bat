@echo off
REM ── AURA Backend Startup Script ───────────────────────────────────────────
REM Run this from anywhere: double-click or call from the repo root.
REM It activates the venv (if present) and launches the Flask server.

REM Change to the backend directory (same folder as this script)
cd /d "%~dp0"

REM ── Check .env exists ─────────────────────────────────────────────────────
if not exist ".env" (
    echo.
    echo [WARNING] backend\.env not found!
    echo  Copy .env.example to .env and fill in your ROBOFLOW_API_KEY.
    echo  Roboflow detection models will be SKIPPED without it.
    echo.
)

REM ── Activate virtual environment if it exists ─────────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No venv found - using system Python.
    echo        If packages are missing, run: pip install -r requirements.txt
)

REM ── Start Flask server ────────────────────────────────────────────────────
echo [INFO] Starting AURA backend on http://localhost:5000 ...
echo [INFO] Test UI: http://localhost:5000/test
echo.
python app.py
pause
