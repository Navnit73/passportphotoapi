"""
Application settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Global application configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    api_key: str = ""  # Change this in .env
    
    # Cloudinary
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""

    # Storage paths
    upload_dir: Path = Path("uploads")
    result_dir: Path = Path("results")

    # Processing
    max_upload_size_mb: int = 10
    default_country: str = "US"
    default_document_type: str = "passport"

    # Performance
    rembg_enabled: bool = True  # fallback only — not used on default path
    processing_timeout_seconds: int = 30

    # CORS
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        env_prefix = "PHOTO_"


settings = Settings()
