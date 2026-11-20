"""
SportsVision AI — FastAPI Application
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.database.session import init_db
from backend.app.api.v1 import health, videos, matches
from backend.app.api import websocket

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ─────────────────────────────────────────
    logger.info("SportsVision AI backend starting up …")

    # Ensure storage dirs exist
    for d in [settings.STORAGE_DIR, settings.VIDEOS_DIR,
              settings.PROCESSED_DIR, settings.FRAMES_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    await init_db()
    logger.info("Database initialised ✓")
    yield

    # ── Shutdown ─────────────────────────────────────────
    logger.info("Backend shutting down")


app = FastAPI(
    title="SportsVision AI",
    description="Real-Time Football Intelligence & Performance Analytics",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────
app.include_router(health.router,   prefix="/api/v1")
app.include_router(videos.router,   prefix="/api/v1")
app.include_router(matches.router,  prefix="/api/v1")
app.include_router(websocket.router)


@app.get("/")
async def root():
    return {
        "name": "SportsVision AI",
        "version": "0.1.0",
        "docs": "/docs",
    }
