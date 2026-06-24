# VoiceMedAI — Voice-based Medical AI Assistant

Locally hosted, **voice-only** medical guidance system for Primary Healthcare Centres (PHCs) in Southwest Nigeria — designed for low-literacy users who speak Nigerian-accented English, Yoruba-English code-switching, or Nigerian Pidgin.

Based on the System Analysis & Design specification: zero-text UI, LAN-only operation, modular ASR → idiom mapping → RAG → LLM → TTS pipeline.

## Features (functional requirements)

| ID | Feature |
|----|---------|
| FR-01 | Voice capture via browser microphone — no text input |
| FR-02 | ASR for Nigerian-accented speech (Whisper when installed; demo mode otherwise) |
| FR-03 | Nigerian medical idiom → clinical terminology mapping |
| FR-04 | Evidence-grounded PHC guidance from local knowledge base |
| FR-05 | Spoken responses only — no on-screen text |
| FR-06 | Runs on local network — no internet required |
| FR-07 | Escalation alerts for chest pain, obstetric emergencies, severe infection, etc. |
| FR-08 | Culturally appropriate register with Southwest Nigerian honorifics |
| FR-09 | Multiple LAN clients supported |
| FR-10 | Voice error feedback when pipeline fails |

## Architecture

```
React (Presentation)  →  FastAPI (Orchestrator)  →  Models (ASR, RAG, LLM, TTS)
     MediaRecorder          /transcribe /reason /speak
     Zero-text UI           Idiom + RAG + Rules/LLM
```

## Quick start

### Backend (inference server)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
# Installs Whisper (real speech-to-text) + Windows voice output
scripts\start_server.bat          # or: uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Check the server is really working: open `http://127.0.0.1:8000/health` — you should see `"asr": {"mode": "whisper"}` (not `"demo"`).

### Frontend (client devices on LAN)

From the **project root** (recommended):

```bash
npm install --prefix frontend
npm run dev
```

Or on Windows, double-click / run:

```bash
scripts\start_frontend.bat
```

Or from the `frontend` folder:

```bash
cd frontend
npm install
npm run dev -- --host
```

> If you see `ENOENT: no such file or directory, package.json`, you are in the wrong folder — use one of the commands above.

Open `http://<server-ip>:5173` from any phone or PC on the same Wi‑Fi. Tap the microphone, speak your concern, tap again when finished — the system responds with voice only.

### API endpoints (Section 6.3)

- `POST /transcribe` — multipart audio → `{ transcript }`
- `POST /reason` — `{ query }` → `{ guidance, escalate, normalized_query }`
- `POST /speak` — `{ text }` → `audio/wav`
- `POST /consult` — full pipeline in one request
- `GET /health` — server status

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VOICEMED_ASR_MODE` | `auto` | `demo`, `whisper`, or `auto` |
| `HF_MODEL_ID` | `openai/whisper-tiny` | HuggingFace ASR model |
| `LOCAL_LLM_MODEL_PATH` | `./models/llm.gguf` | Optional llama.cpp GGUF model |
| `VITE_API_URL` | *(empty)* | Frontend API base; set to `http://server-ip:8000` for production build |

## Data files

- `data/idioms.json` — Nigerian Pidgin / colloquial → clinical terms
- `data/phc_knowledge.json` — PHC condition guidance for RAG
- `data/escalation_keywords.json` — high-risk symptom flags

## Production LAN deployment

1. Run backend on the PHC mini-PC: `scripts/start_server.sh`
2. Build frontend: `cd frontend && npm run build && npx serve dist -l 5173 --host`
3. Point patient devices to `http://<phc-server-ip>:5173`
4. First voice query downloads the Whisper `tiny` model (~75 MB); watch the backend terminal for `ASR transcript:` logs
5. Optional: place BioMistral GGUF in `models/` for llama.cpp LLM upgrade

## Project structure

```
backend/           FastAPI orchestrator + services
frontend/          React 18 zero-text voice UI
data/              Idiom lexicon, PHC knowledge, escalation rules
scripts/           start_server.sh / .bat for CHEW setup
asr/               ASR fine-tuning experiments (future)
llm/               Local LLM adapters (future)
```

## Disclaimer

This system provides **preliminary health guidance** for PHC settings — not a diagnosis. High-risk symptoms trigger an explicit directive to seek immediate in-person care.
