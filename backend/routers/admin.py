"""Admin and diagnostic router — /health /status /admin/*."""

import logging

from fastapi import APIRouter
from backend.services.pipeline import get_system_status

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    summary="Quick liveness check",
)
def health_check() -> dict:
    status = get_system_status()
    return {"status": "ok", "service": "voicemed-ai", **status}


@router.get(
    "/status",
    summary="Full component readiness diagnostic",
)
def full_status() -> dict:
    return get_system_status()
