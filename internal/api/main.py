"""FastAPI application entry point for Analytics Engine."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Analytics Engine API",
    description="Social media analytics processing service",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "analytics-engine"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Analytics Engine API",
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "internal.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
