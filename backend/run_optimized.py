"""Optimized development entrypoint for GPUs with limited VRAM (like 4GB RTX 3050)."""

import os
import sys

# Optimize CUDA memory usage BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Use mirror for faster downloads
os.environ["HF_TIMEOUT"] = "600"  # 10 minutes timeout

# Optional: Set smaller default model if needed
# os.environ["MODEL_NAME"] = "nlpconnect/vit-gpt2-image-captioning"

from app.main import app
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print("=" * 70)
    print("🎯 Photo Rename Tool - Optimized for limited VRAM GPUs")
    print("=" * 70)
    print(f"✓ CUDA memory optimization enabled")
    print(f"✓ Mirror endpoint: {os.environ.get('HF_ENDPOINT')}")
    print(f"✓ Default timeout: {os.environ.get('HF_TIMEOUT')}s")
    print("=" * 70)
    
    # Check GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            print(f"✓ GPU: {gpu_info.name} ({gpu_info.total_memory / 1024**3:.1f}GB)")
            print(f"  Tip: If you get OOM, select CPU in frontend or use smaller model")
    except Exception as e:
        pass
    
    print()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
