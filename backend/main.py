"""VoiceMedAI — FastAPI application entry point.

This file is intentionally lean: it creates the app, configures middleware,
registers routers, and handles startup/shutdown.
All business logic, schemas, and endpoint handlers live in their own modules.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database.db import init_db
from backend.routers import admin as admin_router
from backend.routers import auth as auth_router
from backend.routers import consult as consult_router
from backend.services.pipeline import get_system_status, preload_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialise SQLite database (creates tables if they don't exist)
    init_db()

    # 2. Log system component status
    logger.info("VoiceMedAI starting — %s", get_system_status())



    # 4. Preload heavy models in background threads
    preload_models()

    yield

    logger.info("VoiceMedAI shutdown complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VoiceMedAI — Voice-based Medical AI Assistant",
    description="Locally hosted PHC voice assistant for Southwest Nigeria",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
    expose_headers=[
        "X-VoiceMed-Transcript",
        "X-VoiceMed-Guidance",
        "X-VoiceMed-Escalate",
        "X-VoiceMed-ConversationId",
    ],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router.router, prefix="/auth", tags=["Authentication"])
app.include_router(consult_router.router, tags=["Consultation"])
app.include_router(admin_router.router, tags=["Admin"])
