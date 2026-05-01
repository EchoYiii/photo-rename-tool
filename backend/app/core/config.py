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

    MODEL_NAME = os.getenv("MODEL_NAME", "nlpconnect/vit-gpt2-image-captioning")
    VALIDATION_MODEL_NAME = os.getenv(
        "VALIDATION_MODEL_NAME",
        "google/siglip2-base-patch16-224",
    )
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    SUPPORTED_ELEMENT_RECOGNITION_MODELS = {
        "florence-community/Florence-2-base-ft": {
            "label": "Florence-2 base-ft（默认，准确）",
            "desc": "高准确度，多任务能力强；推荐 GPU（16GB+）或高性能 CPU。本地部署下载大。",
        },
        "Salesforce/blip-image-captioning-large": {
            "label": "BLIP 图像描述（快速、通用）",
            "desc": "通用图像描述能力，适合快速候选生成；推荐有 GPU（4-8GB）。",
        },
        "nlpconnect/vit-gpt2-image-captioning": {
            "label": "ViT-GPT2 图像描述（轻量、稳定）",
            "desc": "轻量级图像描述，资源需求低，可在 CPU 上运行但较慢。",
        },
        "yolov8": {
            "label": "YOLOv8 目标检测（物种识别专用）",
            "desc": "基于 YOLOv8 的目标检测模型，可识别 1000+ 物体类别，包括多种鸟类、动物。支持 GPU/CPU。",
        },
    }
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
