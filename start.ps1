# SportsVision AI - PowerShell Launcher
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         SportsVision AI - Launch Services" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '.\venv\Scripts\uvicorn.exe' backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

Write-Host "Starting Next.js Frontend on http://localhost:3000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location frontend; npm run dev"

Write-Host ""
Write-Host "Services launched in new windows:" -ForegroundColor Yellow
Write-Host "  - Frontend Dashboard : http://localhost:3000" -ForegroundColor White
Write-Host "  - Backend API        : http://localhost:8000" -ForegroundColor White
Write-Host "  - API Documentation  : http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
