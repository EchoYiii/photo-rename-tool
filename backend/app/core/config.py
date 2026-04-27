"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs"))
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"}

    MODEL_NAME = os.getenv("MODEL_NAME", "florence-community/Florence-2-base-ft")
    VALIDATION_MODEL_NAME = os.getenv(
        "VALIDATION_MODEL_NAME",
        "google/siglip2-base-patch16-224",
    )
    DEVICE_PREFERENCE = os.getenv("DEVICE_PREFERENCE", "auto").lower()
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.01"))
    MAX_LABELS = int(os.getenv("MAX_LABELS", "10"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
