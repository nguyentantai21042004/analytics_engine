"""Database models for Analytics Engine."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, Column, Float, Integer, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class PostAnalytics(Base):
    """Post analytics model."""

    __tablename__ = "post_analytics"

    id = Column(String(50), primary_key=True)
    project_id = Column(PG_UUID, nullable=False)
    platform = Column(String(20), nullable=False)

    # Timestamps
    published_at = Column(TIMESTAMP, nullable=False)
    analyzed_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Overall analysis
    overall_sentiment = Column(String(10), nullable=False)
    overall_sentiment_score = Column(Float)
    overall_confidence = Column(Float)

    # Intent
    primary_intent = Column(String(20), nullable=False)
    intent_confidence = Column(Float)

    # Impact
    impact_score = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False)
    is_viral = Column(Boolean, default=False)
    is_kol = Column(Boolean, default=False)

    # JSONB columns
    aspects_breakdown = Column(JSONB)
    keywords = Column(JSONB)
    sentiment_probabilities = Column(JSONB)
    impact_breakdown = Column(JSONB)

    # Raw metrics
    view_count = Column(Integer, default=0)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    save_count = Column(Integer, default=0)
    follower_count = Column(Integer, default=0)

    # Processing metadata
    processing_time_ms = Column(Integer)
    model_version = Column(String(50))
