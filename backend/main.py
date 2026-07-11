"""VoiceMedAI — FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.database.db import init_db
from backend.routers import admin, auth, consult
from backend.services.pipeline import get_system_status, preload_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown hooks."""
    init_db()
    status = get_system_status()
    logger.info("VoiceMedAI starting 🏥 %s", status)
    preload_models()
    yield
    logger.info("VoiceMedAI shutdown complete")


app = FastAPI(
    title="VoiceMedAI",
    description="Voice Medical AI Assistant — PHC system for Ondo State",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — explicit allowlist (NDPR: no wildcard origin with credentials).
# Set FRONTEND_ORIGINS as a comma-separated list in production, e.g.
#   FRONTEND_ORIGINS=https://medvoice.vercel.app,https://voicemed.example.ng
# ---------------------------------------------------------------------------
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("FRONTEND_ORIGINS", _default_origins).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=[
        "X-VoiceMed-Transcript",
        "X-VoiceMed-Escalate",
        "X-VoiceMed-Guidance",
        "X-VoiceMed-ConversationId",
        "X-VoiceMed-Triage",
    ],
)


# ---------------------------------------------------------------------------
# Security headers on every response
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), geolocation=()"
    # HSTS only meaningful behind TLS (Render/production)
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


app.include_router(admin.router, tags=["Admin"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(consult.router, tags=["Consult"])
