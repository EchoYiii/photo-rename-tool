"""超激进的内存优化启动脚本 - 解决Windows页面文件太小错误(OS Error 1455)。

专为4GB显存或更低配置优化，重点解决虚拟内存不足问题。
"""

import os
import sys

# ====================== 1. Windows虚拟内存优化 ======================
# 限制PyTorch线程数，减少并发内存使用
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

# 禁用某些CUDA优化，减少内存开销
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

# Hugging Face 配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_TIMEOUT"] = "1200"  # 20分钟超时

# 默认使用轻量模型
os.environ["MODEL_NAME"] = "nlpconnect/vit-gpt2-image-captioning"
os.environ["DEVICE_PREFERENCE"] = "cpu"  # 默认用CPU避免GPU OOM

from app.main import app
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print("=" * 70)
    print("🎯 Photo Rename Tool - 低内存模式（解决页面文件太小）")
    print("=" * 70)
    print(f"✓ 线程限制: {os.environ.get('OMP_NUM_THREADS')} 核")
    print(f"✓ CUDA内存优化: max_split_size_mb:256")
    print(f"✓ 默认模型: {os.environ.get('MODEL_NAME')}")
    print(f"✓ 默认设备: CPU（推荐）")
    print(f"✓ 镜像源: {os.environ.get('HF_ENDPOINT')}")
    print("=" * 70)
    print()
    print("📝 提示: 如果仍遇到内存问题:")
    print("   1. 关闭其他占用内存的程序")
    print("   2. 增加Windows虚拟内存（页面文件）大小")
    print("   3. 始终选择CPU模式和轻量模型")
    print()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 禁用热重载节省内存
        workers=1,  # 单worker模式
    )
