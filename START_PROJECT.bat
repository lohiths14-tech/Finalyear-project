@echo off
title Launch Crop Disease Intelligence Platform
echo ========================================================
echo Starting Backend and Frontend Servers...
echo ========================================================

start "Backend Server (FastAPI :8000)" cmd /k "cd /d %~dp0backend && ..\.venv311\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

start "Frontend Dashboard (Vite :5173)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers are launching!
echo Backend Swagger API: http://127.0.0.1:8000/docs
echo Frontend Dashboard:   http://localhost:5173
echo.
