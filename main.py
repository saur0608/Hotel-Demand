"""
Root-level entrypoint for FastAPI backend deployment (Render, Railway, Vercel, Docker).
Re-exports the FastAPI 'app' instance from app/api.py without relocating or modifying core logic.
"""

from app.api import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT

    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
