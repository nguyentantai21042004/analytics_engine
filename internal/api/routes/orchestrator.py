"""Dev/Test API endpoint that runs the full AnalyticsOrchestrator pipeline.

This route accepts a full Atomic JSON post and delegates processing to
`AnalyticsOrchestrator`, bypassing MinIO and RabbitMQ. It is intended for
development and debugging only.

WARNING: This endpoint should NOT be exposed in production environments.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status  # type: ignore
from pydantic import BaseModel, Field, field_validator  # type: ignore
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.config import settings
from core.logger import logger
from infrastructure.ai import PhoBERTONNX
from models.database import Base
from repositories.analytics_repository import AnalyticsRepository, AnalyticsRepositoryError
from services.analytics.orchestrator import AnalyticsOrchestrator
from services.analytics.sentiment import SentimentAnalyzer


router = APIRouter(prefix="/dev", tags=["dev-orchestrator"])


class OrchestratorRequest(BaseModel):
    """Request model matching Atomic JSON post structure."""

    meta: Dict[str, Any] = Field(..., description="Post metadata (id, platform, etc.)")
    content: Dict[str, Any] = Field(..., description="Post content (text, transcription, etc.)")
    interaction: Dict[str, Any] = Field(
        default_factory=dict, description="Engagement metrics"
    )
    author: Dict[str, Any] = Field(default_factory=dict, description="Author information")
    comments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Optional comments associated with the post"
    )

    model_config = {"protected_namespaces": ()}

    @field_validator("meta")
    @classmethod
    def validate_meta(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure meta contains required 'id' field."""
        if not v.get("id"):
            raise ValueError("meta.id is required")
        return v


class OrchestratorResponse(BaseModel):
    """Response model exposing orchestrator output for debugging."""

    status: str = Field(..., description="Processing status")
    data: Dict[str, Any] = Field(..., description="Final analytics payload")

    model_config = {"protected_namespaces": ()}


def _get_session_factory() -> sessionmaker:
    engine = create_engine(settings.database_url_sync)
    Base.metadata.bind = engine
    return sessionmaker(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a database session.

    Yields:
        SQLAlchemy session that is automatically closed after use.
    """
    session_factory = _get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def get_phobert(request: Request) -> PhoBERTONNX | None:
    """Get PhoBERT model from app.state (loaded once during lifespan).

    Returns None if model is not available, allowing graceful degradation.
    """
    if hasattr(request.app.state, "phobert") and request.app.state.phobert is not None:
        return request.app.state.phobert
    return None


@router.post(
    "/process-post-direct",
    response_model=OrchestratorResponse,
    summary="Process a single post via analytics pipeline",
    description="""
    Dev/Test endpoint that runs the full analytics pipeline on a single post.
    
    This endpoint bypasses MinIO and RabbitMQ, accepting the post data directly.
    It is intended for development, debugging, and testing purposes only.
    
    **WARNING**: Do not expose this endpoint in production environments.
    """,
)
async def dev_process_post_direct(
    request: OrchestratorRequest,
    db: Session = Depends(get_db),
    phobert: Optional[PhoBERTONNX] = Depends(get_phobert),
) -> OrchestratorResponse:
    """Dev/Test endpoint: process a single post JSON via AnalyticsOrchestrator.

    Args:
        request: Atomic JSON post data.
        db: Database session (injected).
        phobert: PhoBERT model instance (injected, may be None).

    Returns:
        OrchestratorResponse with status and analytics data.

    Raises:
        HTTPException: If processing fails.
    """
    post_id = request.meta.get("id", "unknown")
    logger.info("Dev endpoint processing post_id=%s", post_id)

    try:
        repo = AnalyticsRepository(db)

        sentiment_analyzer = None
        if phobert is not None:
            sentiment_analyzer = SentimentAnalyzer(phobert)

        orchestrator = AnalyticsOrchestrator(
            repository=repo,
            sentiment_analyzer=sentiment_analyzer,
        )

        post_data = request.model_dump()
        result = orchestrator.process_post(post_data)

        logger.info(
            "Dev endpoint completed: post_id=%s, impact_score=%.2f",
            post_id,
            result.get("impact_score", 0.0),
        )

        return OrchestratorResponse(status="SUCCESS", data=result)

    except ValueError as exc:
        logger.warning("Validation error for post_id=%s: %s", post_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except AnalyticsRepositoryError as exc:
        logger.error("Database error for post_id=%s: %s", post_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database error occurred",
        ) from exc

    except Exception as exc:
        logger.error("Unexpected error for post_id=%s: %s", post_id, exc)
        logger.exception("Dev endpoint error details:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing post: {exc}",
        ) from exc
