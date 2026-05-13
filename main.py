"""
Passport & Visa Photo Processing API

FastAPI application entry point.
Loads country configs at startup, registers routes,
and configures CORS for cross-origin access.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from config.countries.loader import load_all_configs
from config.settings import settings
from services.background import _get_modnet_session, _get_rembg_session

# ─── Logging setup ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: load configs on startup ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load country configs and initialize models on startup."""
    logger.info("=" * 60)
    logger.info("Starting Passport Photo Processing API")
    logger.info("=" * 60)

    # Load country configurations
    configs = load_all_configs()
    logger.info(f"Loaded {len(configs)} country config(s)")

    # Create storage directories
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload dir: {settings.upload_dir.resolve()}")
    logger.info(f"Result dir: {settings.result_dir.resolve()}")

    # Pre-load background removal models
    logger.info("Pre-loading MODNet (primary) background removal model...")
    _get_modnet_session()
    logger.info("Pre-loading rembg u2net (fallback) model...")
    _get_rembg_session()

    logger.info("API ready — accepting requests")
    logger.info("=" * 60)

    yield  # app is running

    logger.info("Shutting down...")


# ─── Create app ───

app = FastAPI(
    title="Passport Photo Processing API",
    description=(
        "Multi-country passport and visa photo processing system. "
        "Handles face detection, background validation/correction, "
        "smart cropping (hair-safe), and compliance metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── CORS ───

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Register routes ───

app.include_router(router)


# ─── Run with uvicorn ───

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
