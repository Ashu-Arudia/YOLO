@echo off
echo ========================================================
echo         SportsVision AI - Launch Services
echo ========================================================
echo.

echo Starting FastAPI Backend (Port 8000)...
start "SportsVision Backend" cmd /k ".\venv\Scripts\uvicorn.exe backend.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting Next.js Frontend (Port 3000)...
start "SportsVision Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both services are launching in separate windows!
echo - Web Dashboard: http://localhost:3000
echo - API Backend:   http://localhost:8000
echo - Swagger Docs:  http://localhost:8000/docs
echo.
pause
