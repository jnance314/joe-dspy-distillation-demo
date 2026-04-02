#!/usr/bin/env python3
"""
Entry point for the DSPy Prompt Optimizer.

Local:     uv run run_web.py
Cloud Run: automatically uses PORT env var
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # Load .env BEFORE anything else so API keys are available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    # reload=True only when running locally (not in production)
    is_local = os.getenv("K_SERVICE") is None  # K_SERVICE is set by Cloud Run
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=is_local)
