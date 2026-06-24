#!/usr/bin/env bash
# CHEW system setup — start VoiceMedAI inference server on LAN (Use Case: System Setup)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d ".venv312" ]; then
  source .venv312/bin/activate
elif [ -d ".venv" ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export VOICEMED_ASR_MODE="${VOICEMED_ASR_MODE:-auto}"
echo "Starting VoiceMedAI on http://0.0.0.0:8000"
echo "Patients: open http://<server-ip>:5173 on any browser on the LAN"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
