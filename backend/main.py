"""VoiceMedAI — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-VoiceMed-Transcript",
        "X-VoiceMed-Escalate",
        "X-VoiceMed-Guidance",
        "X-VoiceMed-ConversationId",
    ],
)

app.include_router(admin.router, tags=["Admin"])
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(consult.router, tags=["Consult"])
