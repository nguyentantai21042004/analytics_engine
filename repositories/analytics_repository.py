"""Repository for persisting analytics results to the database."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.logger import logger
from models.database import PostAnalytics


class AnalyticsRepositoryError(Exception):
    """Base exception for repository operations."""

    pass


class AnalyticsRepository:
    """Repository abstraction for `PostAnalytics` operations.

    This class provides a clean interface for persisting and retrieving
    analytics data, abstracting away SQLAlchemy details from the orchestrator.
    """

    def __init__(self, db: Session) -> None:
        """Initialize repository with database session.

        Args:
            db: SQLAlchemy session for database operations.
        """
        self.db = db

    def save(self, analytics_data: Dict[str, Any]) -> PostAnalytics:
        """Save analytics result into the `post_analytics` table.

        This method performs an insert-or-update (upsert-like) behavior based on
        the primary key `id`, so re-processing the same post overwrites the
        previous analytics record.

        Args:
            analytics_data: Dictionary containing analytics fields matching PostAnalytics model.

        Returns:
            The persisted PostAnalytics instance.

        Raises:
            AnalyticsRepositoryError: If database operation fails.
            ValueError: If analytics_data is missing required 'id' field.
        """
        post_id = analytics_data.get("id")
        if not post_id:
            raise ValueError("analytics_data must contain 'id' field")

        try:
            existing = self.get_by_id(post_id)

            if existing is None:
                post = PostAnalytics(**analytics_data)
                self.db.add(post)
                logger.debug("Creating new PostAnalytics record: id=%s", post_id)
            else:
                post = existing
                for key, value in analytics_data.items():
                    if hasattr(post, key):
                        setattr(post, key, value)
                logger.debug("Updating existing PostAnalytics record: id=%s", post_id)

            self.db.commit()
            self.db.refresh(post)
            return post

        except SQLAlchemyError as exc:
            self.db.rollback()
            logger.error("Database error saving analytics for post_id=%s: %s", post_id, exc)
            raise AnalyticsRepositoryError(f"Failed to save analytics: {exc}") from exc

    def get_by_id(self, post_id: str) -> Optional[PostAnalytics]:
        """Fetch a `PostAnalytics` record by its primary key.

        Args:
            post_id: The primary key of the post.

        Returns:
            PostAnalytics instance if found, None otherwise.
        """
        return self.db.query(PostAnalytics).filter(PostAnalytics.id == post_id).one_or_none()

    def get_by_project(
        self, project_id: str, *, limit: int = 100, offset: int = 0
    ) -> List[PostAnalytics]:
        """Fetch analytics records for a specific project.

        Args:
            project_id: The project UUID to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of PostAnalytics instances.
        """
        return (
            self.db.query(PostAnalytics)
            .filter(PostAnalytics.project_id == project_id)
            .order_by(PostAnalytics.analyzed_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    def delete_by_id(self, post_id: str) -> bool:
        """Delete a PostAnalytics record by its primary key.

        Args:
            post_id: The primary key of the post to delete.

        Returns:
            True if record was deleted, False if not found.
        """
        try:
            result = (
                self.db.query(PostAnalytics).filter(PostAnalytics.id == post_id).delete()
            )
            self.db.commit()
            return result > 0
        except SQLAlchemyError as exc:
            self.db.rollback()
            logger.error("Database error deleting post_id=%s: %s", post_id, exc)
            raise AnalyticsRepositoryError(f"Failed to delete analytics: {exc}") from exc
