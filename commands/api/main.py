"""
Analytics Engine API - Main entry point.
Loads config, initializes instances, and starts the FastAPI service.
"""

import warnings
from contextlib import asynccontextmanager

# Suppress expected warnings at startup
warnings.filterwarnings("ignore", message=".*protected namespace.*", category=UserWarning)

from fastapi import FastAPI, Request, HTTPException, status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from core.config import settings
from core.logger import logger
from internal.api.main import app as internal_app


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan - startup and shutdown.
    """
    try:
        logger.info(
            f"========== Starting {settings.service_name} v{settings.service_version} API service =========="
        )
        logger.info(f"API: {settings.api_host}:{settings.api_port}")

        logger.info(
            f"========== {settings.service_name} API service started successfully =========="
        )

        yield

        # Shutdown sequence
        logger.info("========== Shutting down API service ==========")
        logger.info("========== API service stopped successfully ==========")

    except Exception as e:
        logger.error(f"Fatal error in application lifespan: {e}")
        logger.exception("Lifespan error details:")
        raise


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    try:
        logger.info("Creating FastAPI application...")

        # Use the app from internal/api/main.py and add lifespan
        app = internal_app
        app.router.lifespan_context = lifespan

        logger.info("FastAPI application created successfully")
        return app

    except Exception as e:
        logger.error(f"Failed to create FastAPI application: {e}")
        logger.exception("Application creation error details:")
        raise


# Create application instance
try:
    logger.info("Initializing Analytics Engine API...")
    app = create_app()
    logger.info("Application instance created successfully")
except Exception as e:
    logger.error(f"Failed to create application instance: {e}")
    logger.exception("Startup error details:")
    raise


# Run with: uv run cmd/api/main.py
if __name__ == "__main__":
    import uvicorn
    import sys
    import os

    try:
        logger.info("========== Starting Uvicorn Server ==========")
        logger.info(f"Host: {settings.api_host}")
        logger.info(f"Port: {settings.api_port}")
        logger.info(f"Reload: {settings.api_reload}")

        # Ensure project root is in PYTHONPATH
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Set PYTHONPATH environment variable for subprocess (uvicorn reload)
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if project_root not in current_pythonpath:
            new_pythonpath = (
                f"{project_root}:{current_pythonpath}" if current_pythonpath else project_root
            )
            os.environ["PYTHONPATH"] = new_pythonpath

        # Use string path when reload=True
        if settings.api_reload:
            uvicorn.run(
                "cmd.api.main:app",
                host=settings.api_host,
                port=settings.api_port,
                reload=True,
                log_level="info",
            )
        else:
            uvicorn.run(
                app,
                host=settings.api_host,
                port=settings.api_port,
                reload=False,
                log_level="info",
            )

    except Exception as e:
        logger.error(f"Failed to start Uvicorn server: {e}")
        logger.exception("Uvicorn startup error details:")
        raise
