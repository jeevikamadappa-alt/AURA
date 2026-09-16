# ── AURA Backend Startup Script (PowerShell) ──────────────────────────────
# Run from anywhere: .\backend\start.ps1  OR  cd backend; .\start.ps1

# Change to the backend directory (same folder as this script)
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   AURA Incident Detection Backend" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor Cyan

# ── Check .env exists ─────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "[WARNING] backend\.env not found!" -ForegroundColor Yellow
    Write-Host "  Copy .env.example to .env and fill in your ROBOFLOW_API_KEY." -ForegroundColor Yellow
    Write-Host "  Roboflow detection models will be SKIPPED without it." -ForegroundColor Yellow
    Write-Host ""
}

# ── Activate virtual environment if it exists ─────────────────────────────
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "[INFO] No venv found - using system Python." -ForegroundColor Yellow
    Write-Host "       If packages are missing, run: pip install -r requirements.txt" -ForegroundColor Yellow
}

# ── Start Flask server ────────────────────────────────────────────────────
Write-Host ""
Write-Host "[INFO] Starting AURA backend on http://localhost:5000 ..." -ForegroundColor Green
Write-Host "[INFO] Test UI: http://localhost:5000/test" -ForegroundColor Cyan
Write-Host ""
python app.py
