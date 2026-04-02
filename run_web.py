#!/usr/bin/env python3
"""Entry point for the web UI. Run with: uv run run_web.py"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="127.0.0.1", port=8000, reload=True)
