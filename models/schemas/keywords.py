"""Keywords analysis schemas."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class KeywordSentimentBreakdown(BaseModel):
    """Sentiment breakdown for a keyword."""

    POSITIVE: int
    NEUTRAL: int
    NEGATIVE: int


class TopKeyword(BaseModel):
    """Top keyword with statistics."""

    keyword: str
    count: int
    avg_sentiment_score: float
    aspect: str
    sentiment_breakdown: KeywordSentimentBreakdown


class InputKeywordRank(BaseModel):
    """Ranking info for the input keyword."""

    keyword: str = Field(description="The input keyword searched for")
    rank: Optional[int] = Field(description="Rank position (1-based), null if not found")
    count: int = Field(description="Number of occurrences")
    avg_sentiment_score: float = Field(description="Average sentiment score")
    in_top: bool = Field(description="Whether keyword is in the top list")


class KeywordsData(BaseModel):
    """Top keywords response data."""

    keywords: List[TopKeyword]
    input_keyword_ranks: Optional[List[InputKeywordRank]] = Field(
        default=None,
        description="Ranking info for specified keywords (when include_rank_for is provided)",
    )


class KeywordsRequest(BaseModel):
    """Keywords query parameters."""

    limit: int = Field(default=20, ge=1, le=50, description="Number of keywords to return")
    include_rank_for: Optional[str] = Field(
        default=None,
        description="Comma-separated keywords to find ranking for (e.g. 'keyword1,keyword2,keyword3')",
    )
