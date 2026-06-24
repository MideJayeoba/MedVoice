@echo off
REM CHEW system setup — start VoiceMedAI inference server (Windows PHC)
cd /d "%~dp0\.."

if exist .venv312\Scripts\activate.bat (
    call .venv312\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)


set VOICEMED_ASR_MODE=auto
echo Starting VoiceMedAI on http://0.0.0.0:8000
echo Patients: open http://localhost:5173 after starting the frontend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
