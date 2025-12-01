"""Message queue consumer entry point for Analytics Engine."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Optional, Callable, Any, TYPE_CHECKING, Iterator

try:
    from aio_pika import IncomingMessage  # type: ignore

    AIO_PIKA_AVAILABLE = True
except ImportError:
    AIO_PIKA_AVAILABLE = False
    if TYPE_CHECKING:
        from aio_pika import IncomingMessage  # type: ignore
    else:
        IncomingMessage = Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from core.logger import logger
from core.config import settings
from infrastructure.ai import PhoBERTONNX, SpacyYakeExtractor
from infrastructure.storage.minio_client import (
    MinioAdapter,
    MinioAdapterError,
    MinioObjectNotFoundError,
    MinioDecompressionError,
)
from models.database import Base
from repositories.analytics_repository import AnalyticsRepository, AnalyticsRepositoryError
from services.analytics.orchestrator import AnalyticsOrchestrator


def _create_session_factory() -> sessionmaker:
    """Create a synchronous SQLAlchemy session factory."""
    engine = create_engine(settings.database_url_sync)
    Base.metadata.bind = engine
    return sessionmaker(bind=engine)


@contextmanager
def _db_session(session_factory: sessionmaker) -> Iterator[Session]:
    """Context manager yielding a DB session and ensuring cleanup."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def create_message_handler(
    phobert: Optional[PhoBERTONNX], spacyyake: Optional[SpacyYakeExtractor]
) -> Callable[[IncomingMessage], None]:
    """Create message handler with AI model instances.

    This factory function creates a message handler that has access to
    the AI model instances passed in. The handler will process incoming
    messages from RabbitMQ.

    Args:
        phobert: PhoBERT model instance (may be None if initialization failed)
        spacyyake: SpaCy-YAKE extractor instance (may be None if initialization failed)

    Returns:
        Async callable that processes incoming messages

    Example:
        >>> handler = create_message_handler(phobert, spacyyake)
        >>> await rabbitmq_client.consume(handler)
    """

    minio_adapter = MinioAdapter()
    session_factory = _create_session_factory()

    async def message_handler(message: IncomingMessage) -> None:
        """Process incoming message from RabbitMQ.

        Args:
            message: Incoming message from RabbitMQ queue

        The handler processes messages in the following order:
        1. Decode and parse JSON envelope
        2. Fetch post data from MinIO if data_ref is present
        3. Run analytics pipeline via orchestrator
        4. Persist results to database

        Errors are logged and re-raised to trigger message nack.
        """
        async with message.process():
            post_id = "unknown"
            try:
                # Decode message body
                body = message.body.decode("utf-8")
                logger.info("Received message: %s...", body[:100])

                # Parse JSON envelope
                try:
                    envelope = json.loads(body)
                except json.JSONDecodeError as exc:
                    logger.error("Invalid JSON in message: %s", exc)
                    raise

                # Determine data source: direct post_data or MinIO reference
                post_data: dict[str, Any]
                if "data_ref" in envelope:
                    ref = envelope.get("data_ref") or {}
                    bucket = ref.get("bucket")
                    path = ref.get("path")

                    if not bucket or not path:
                        raise ValueError("Invalid data_ref: bucket and path are required")

                    logger.debug("Fetching post from MinIO: %s/%s", bucket, path)
                    try:
                        post_data = minio_adapter.download_json(bucket, path)
                    except MinioObjectNotFoundError:
                        logger.error("MinIO object not found: %s/%s", bucket, path)
                        raise
                    except MinioDecompressionError as exc:
                        logger.error(
                            "Decompression failed for MinIO object %s/%s: %s",
                            bucket,
                            path,
                            exc,
                        )
                        raise
                    except MinioAdapterError as exc:
                        logger.error("MinIO error fetching %s/%s: %s", bucket, path, exc)
                        raise
                else:
                    post_data = envelope

                # Extract identifiers for logging
                meta = post_data.get("meta") or {}
                post_id = meta.get("id", "unknown")
                platform = meta.get("platform", "UNKNOWN")
                logger.info("Processing post %s from %s via orchestrator", post_id, platform)

                # Create DB-backed repository and orchestrator per message
                with _db_session(session_factory) as db:
                    repo = AnalyticsRepository(db)

                    # SentimentAnalyzer is built from provided PhoBERT model if available
                    sentiment_analyzer = None
                    if phobert is not None:
                        sentiment_analyzer = SentimentAnalyzer(phobert)  # type: ignore[name-defined]

                    orchestrator = AnalyticsOrchestrator(
                        repository=repo,
                        sentiment_analyzer=sentiment_analyzer,
                    )
                    result = orchestrator.process_post(post_data)

                logger.info(
                    "Message processed successfully: post_id=%s, impact_score=%.2f",
                    post_id,
                    result.get("impact_score", 0.0),
                )

                # Message will be auto-acked when context exits without exception

            except (json.JSONDecodeError, ValueError) as exc:
                # Validation errors - message is malformed, don't retry
                logger.error("Validation error for post_id=%s: %s", post_id, exc)
                raise

            except (MinioAdapterError, AnalyticsRepositoryError) as exc:
                # Infrastructure errors - may be transient, could retry
                logger.error("Infrastructure error for post_id=%s: %s", post_id, exc)
                raise

            except Exception as exc:
                # Unexpected errors
                logger.error("Unexpected error processing post_id=%s: %s", post_id, exc)
                logger.exception("Message processing error details:")
                raise

    return message_handler
