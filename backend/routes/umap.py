"""
UMAP visualization route.
Uses the same file_id pattern as all other analysis endpoints.
"""

import os
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from umap_visualizer import is_ready, project_and_plot
from routes.upload import get_file_path

router = APIRouter(prefix="/api", tags=["umap"])


class UMAPRequest(BaseModel):
    file_id: str


@router.get("/umap/status")
async def umap_status():
    """Check whether the UMAP model is fitted and ready."""
    return {"ready": is_ready()}


@router.post("/umap/project")
async def project_audio(req: UMAPRequest):
    """
    Project an already-uploaded audio file into the UMAP 2D dialect space.

    Response JSON:
    {
        "umap_chart":        "<base64 PNG>",
        "predicted_dialect": "Egyptian Arabic",
        "umap_coords":       [x, y],
        "nearest_cluster":   2
    }
    """
    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "UMAP visualizer is not ready. "
                "Ensure umap-learn is installed and model artifacts exist."
            ),
        )

    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    try:
        result = project_and_plot(path)
        return result
    except Exception as exc:
        logging.error("UMAP projection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
