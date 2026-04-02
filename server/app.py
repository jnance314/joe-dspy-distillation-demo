"""FastAPI application."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routes import router

app = FastAPI(title="DSPy Prompt Optimizer")

STATIC_DIR = Path(__file__).parent.parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
