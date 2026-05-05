"""
Country configuration loader.

Loads JSON config files from the config/countries/ directory,
validates them against the unified CountryConfig schema,
and provides a registry for runtime lookup.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from .schema import CountryConfig

logger = logging.getLogger(__name__)

# ─── Global registry ───
_configs: dict[str, CountryConfig] = {}

# Directory containing country JSON files
_COUNTRIES_DIR = Path(__file__).parent


def load_all_configs() -> dict[str, CountryConfig]:
    """
    Load all country JSON configs from the countries/ directory.
    Each file is validated against CountryConfig schema.
    Returns a dict keyed by "{country_code}_{document_type}".
    """
    global _configs
    _configs.clear()

    json_files = sorted(_COUNTRIES_DIR.glob("*.json"))

    if not json_files:
        logger.warning(f"No country config files found in {_COUNTRIES_DIR}")
        return _configs

    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)

            config = CountryConfig(**raw)
            key = f"{config.country.code}_{config.document_type}"
            _configs[key] = config

            logger.info(
                f"Loaded config: {key} "
                f"({config.target_width_px}x{config.target_height_px}px, "
                f"head {config.face_constraints.head_height_pct.min}-"
                f"{config.face_constraints.head_height_pct.max}%)"
            )

        except Exception as e:
            logger.error(f"Failed to load {filepath.name}: {e}")

    logger.info(f"Loaded {len(_configs)} country config(s)")
    return _configs


def get_config(
    country_code: str,
    document_type: str = "passport",
) -> Optional[CountryConfig]:
    """
    Retrieve a loaded country config.

    Args:
        country_code: ISO country code (e.g. "US")
        document_type: "passport" or "visa"

    Returns:
        CountryConfig or None if not found.
    """
    key = f"{country_code}_{document_type}"
    config = _configs.get(key)

    if config is None:
        logger.warning(f"No config found for key '{key}'")

    return config


def get_all_configs() -> dict[str, CountryConfig]:
    """Return all loaded configs."""
    return dict(_configs)


def list_supported() -> list[dict]:
    """Return a summary list of all supported country/document combos."""
    return [
        {
            "country_code": cfg.country.code,
            "country_name": cfg.country.name,
            "document_type": cfg.document_type,
            "dimensions": f"{cfg.target_width_px}x{cfg.target_height_px}",
        }
        for cfg in _configs.values()
    ]
