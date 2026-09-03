@echo off
setlocal enabledelayedexpansion
echo ============================================
echo   DiaFact - Starting All Services
echo ============================================

cd /d "%~dp0"

:: --------------------------------------------------------------------------
:: Artifacts. The service will not start without them. They are BUILT from the
:: published experiment rather than copied from a hard-coded drive, and
:: prepare_artifacts.py aborts unless the model reproduces the paper's metrics.
::
:: Point EXP_V2 at your copy of "Final Results\exp_v2" if it is elsewhere.
:: --------------------------------------------------------------------------
if "%EXP_V2%"=="" set "EXP_V2=..\..\TEHI 2026\exp_v2\exp_v2\Final Results\exp_v2"

if not exist "ml_service\artifacts\model.json" (
  echo.
  echo Artifacts missing - building them from: %EXP_V2%
  pushd ml_service
  python prepare_artifacts.py --source "%EXP_V2%"
  if errorlevel 1 (
    echo.
    echo FAILED to build artifacts. Set EXP_V2 to your exp_v2 results folder, e.g.
    echo    set "EXP_V2=D:\thesis paper\TEHI 2026\exp_v2\exp_v2\Final Results\exp_v2"
    popd
    pause
    exit /b 1
  )
  popd
)

echo.
echo Cleaning up existing processes on 5001 / 3001 / 5173...
for %%P in (5001 3001 5173) do (
  for /f "tokens=5" %%a in ('netstat -aon ^| find ":%%P" ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
)

echo.
echo [1/3] Starting ML Service (Flask :5001)...
start "DiaFact ML Service" cmd /k "cd /d ""%~dp0ml_service"" && python app.py"

echo Waiting for the ML service to load the model...
set /a tries=0
:waitml
timeout /t 2 /nobreak >nul
set /a tries+=1
curl -s -o nul http://127.0.0.1:5001/health && goto mlok
if !tries! lss 30 goto waitml
echo   WARNING: ML service did not answer /health in 60s - continuing anyway.
goto mlnext
:mlok
echo   ML service is up.
:mlnext

echo.
echo [2/3] Starting Backend (Node :3001)...
start "DiaFact Backend" cmd /k "cd /d ""%~dp0backend"" && npm install --silent && node index.js"
timeout /t 4 /nobreak >nul

echo.
echo [3/3] Starting Frontend (React :5173)...
start "DiaFact Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm install --silent && npm run dev"

echo.
echo ============================================
echo   All services started.
echo   Frontend:   http://localhost:5173
echo   Backend:    http://localhost:3001/api/health
echo   ML Service: http://localhost:5001/health
echo.
echo   Seed demo patients (after all three are up):
echo     cd backend ^&^& npm run seed
echo ============================================
pause
