"""
Optional standalone runner (same routes, no /us prefix):

    uvicorn us_passport.app:app --host 0.0.0.0 --port 8001

The normal way is the main app (`python main.py`), where these routes live under /us.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.models import HealthResponse
from config.settings import settings
from services.background import _get_modnet_session, _get_rembg_session

from .router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_modnet_session()
    _get_rembg_session()
    yield


app = FastAPI(title="US Passport Photo API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", version="1.0.0", countries_loaded=1)
