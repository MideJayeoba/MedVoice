@echo off
REM Start VoiceMedAI React frontend (Zero-Text UI)
cd /d "%~dp0\..\frontend"

if not exist node_modules (
  echo Installing frontend dependencies...
  call npm install
  if errorlevel 1 exit /b 1
)

echo Starting frontend at http://localhost:5173
echo LAN access: use the Network URL shown below on patient phones
call npm run dev -- --host
