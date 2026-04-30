import os
from pathlib import Path

user_home = Path.home()
cache_dir = user_home / ".cache" / "huggingface" / "hub"

print("=" * 60)
print("Hugging Face Model Cache Location")
print("=" * 60)
print(f"Default cache directory: {cache_dir}")
print(f"Directory exists: {cache_dir.exists()}")

if cache_dir.exists():
    print("\nContents:")
    for item in cache_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")
            # Check the size of each model dir
            try:
                size_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                print(f"    Size: {size_mb:.1f} MB")
            except:
                pass

print("\n" + "=" * 60)
print("Environment Variables:")
print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'Not set')}")
print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'Not set')}")
print("=" * 60)
