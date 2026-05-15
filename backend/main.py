"""
FastAPI main application for Arabic Dialect Detection.
"""
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"  # Must be set before numba is imported (Python 3.13 compat)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.analysis import router as analysis_router
from routes.transcription import router as transcription_router
from routes.conversion import router as conversion_router
from routes.mixer import router as mixer_router
from routes.umap import router as umap_router

app = FastAPI(
    title="Arabic Dialect Detection API",
    description="API for Arabic dialect fingerprinting and analysis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(transcription_router)
app.include_router(conversion_router)
app.include_router(mixer_router)
app.include_router(umap_router)


@app.get("/")
async def root():
    return {"message": "Arabic Dialect Detection API", "status": "running"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}
