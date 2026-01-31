"""FastAPI application for SSL attention visualization.

Run with:
    uvicorn app.backend.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add SSL attention source to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import API_PREFIX, CORS_ORIGINS
from app.backend.routers import (
    attention_router,
    comparison_router,
    images_router,
    metrics_router,
)

app = FastAPI(
    title="SSL Attention Visualization API",
    description="API for visualizing SSL model attention patterns on WikiChurches images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(images_router, prefix=API_PREFIX)
app.include_router(attention_router, prefix=API_PREFIX)
app.include_router(metrics_router, prefix=API_PREFIX)
app.include_router(comparison_router, prefix=API_PREFIX)


@app.get("/")
async def root() -> dict:
    """Root endpoint with API info."""
    return {
        "name": "SSL Attention Visualization API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "images": f"{API_PREFIX}/images",
            "attention": f"{API_PREFIX}/attention",
            "metrics": f"{API_PREFIX}/metrics",
            "comparison": f"{API_PREFIX}/compare",
        },
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    from app.backend.services.image_service import image_service
    from app.backend.services.metrics_service import metrics_service

    return {
        "status": "healthy",
        "checks": {
            "annotations_loaded": len(image_service.list_image_ids()) > 0,
            "metrics_db_available": metrics_service.db_exists,
        },
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
