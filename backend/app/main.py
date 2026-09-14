"""
SupportGraph AI — FastAPI Application Entry Point

Provides:
  GET /health  → Health check
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.core.config import settings
    from app.core.logging import configure_logging, get_logger
except ModuleNotFoundError:
    from backend.app.core.config import settings  # type: ignore[no-redef]
    from backend.app.core.logging import configure_logging, get_logger  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Configure logging before anything else
# ---------------------------------------------------------------------------
configure_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan handler (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: runs startup logic, then yields, then shutdown."""
    logger.info(
        "Starting %s v%s in %s mode",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    logger.info("Raw data directory: %s", settings.raw_data_path)
    logger.info("Interim data directory: %s", settings.interim_data_path)
    yield
    logger.info("Shutting down %s", settings.app_name)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "An Agentic Customer Support Intelligence System. "
        "Analyzes real customer-support conversations, classifies intents, "
        "retrieves historically similar resolutions, and decides escalation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow all origins in development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    summary="Health Check",
    tags=["System"],
    response_description="Service health status",
)
async def health() -> dict[str, str]:
    """
    Returns the health status of the service.

    This endpoint is used by load balancers, orchestration systems,
    and monitoring dashboards to verify the service is running.
    """
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "service": "supportgraph-ai",
    }
