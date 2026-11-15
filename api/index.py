"""
Vercel Serverless Function entrypoint for FastAPI.
Routes all requests to the root FastAPI instance.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from main import app  # noqa: F401
