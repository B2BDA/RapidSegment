"""RapidSegment web backend — FastAPI app.

Serves the JSON API under ``/api`` and (in production) the built React frontend
from ``frontend/dist``.
"""
from __future__ import annotations

import os
import sys

# Ensure parent directory is on sys.path so relative imports work when run
# via `python -m uvicorn main:app` or `uvicorn main:app`.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from routers import m1, m2, m3, m4, m5, m6
from storage import SUITE_DIR

app = FastAPI(title="RapidSegment Web UI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (m1.router, m2.router, m3.router, m4.router, m5.router, m6.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    return {"status": "ok", "suite_dir": SUITE_DIR}


@app.post("/api/exit")
def exit_app():
    """Stop the server (single-user local app — mirrors Streamlit's Exit button)."""
    import threading
    threading.Thread(target=lambda: os._exit(0), daemon=True).start()
    return {"status": "stopping"}


# ── Static frontend (production) ──────────────────────────────────────────────
_FRONTEND_DIST = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist"))

if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = os.path.normpath(os.path.join(_FRONTEND_DIST, full_path))
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        index = os.path.join(_FRONTEND_DIST, "index.html")
        if os.path.exists(index):
            return FileResponse(index)
        return {"detail": "frontend not built — run `npm run build` in webui/frontend"}


def run_webui(host: str = "0.0.0.0", port: int = 8000):
    """Start the web backend with uvicorn."""
    import uvicorn
    uvicorn.run("rapidsegment.webui.backend.main:app", host=host, port=port, reload=True)
