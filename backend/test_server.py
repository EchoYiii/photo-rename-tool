#!/usr/bin/env python
"""Test script to start the server with output."""

import sys
import uvicorn

print("Starting server...")
print(f"Python version: {sys.version}")

try:
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()