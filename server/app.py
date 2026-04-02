"""FastAPI application."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()  # Ensure API keys are available regardless of entrypoint

from server.routes import router

app = FastAPI(title="DSPy Prompt Optimizer")

STATIC_DIR = Path(__file__).parent.parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/docs-page")
def docs_page():
    return FileResponse(str(STATIC_DIR / "docs.html"))
