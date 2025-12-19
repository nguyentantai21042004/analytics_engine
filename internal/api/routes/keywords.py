"""Top keywords API route."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.logger import logger
from models.schemas.base import ApiResponse, FilterParams, DateRangeParams
from models.schemas.keywords import (
    KeywordsData,
    TopKeyword,
    KeywordSentimentBreakdown,
    KeywordsRequest,
    InputKeywordRank,
)
from repository.analytics_api_repository import AnalyticsApiRepository

router = APIRouter()


class KeywordsFiltersRequest(FilterParams, DateRangeParams, KeywordsRequest):
    """Keywords request filters."""

    pass


@router.get("/top-keywords", response_model=ApiResponse[KeywordsData])
async def get_top_keywords(
    request: Request,
    filters: KeywordsFiltersRequest = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Get top keywords with sentiment analysis.

    Optionally include ranking info for a specific keyword using include_rank_for param.
    """
    try:
        request_id = getattr(request.state, "request_id", "unknown")

        logger.info(
            f"Request {request_id}: Getting top {filters.limit} keywords for project {filters.project_id}"
            + (f", include_rank_for={filters.include_rank_for}" if filters.include_rank_for else "")
        )

        # Create repository
        repo = AnalyticsApiRepository(db)

        # Get top keywords
        keywords_raw = await repo.get_top_keywords(
            project_id=filters.project_id,
            limit=filters.limit,
            brand_name=filters.brand_name,
            keyword=filters.keyword,
            from_date=filters.from_date,
            to_date=filters.to_date,
        )

        # Convert to response format
        keywords = []
        for kw in keywords_raw:
            sentiment_breakdown = KeywordSentimentBreakdown(
                POSITIVE=kw["sentiment_breakdown"]["POSITIVE"],
                NEUTRAL=kw["sentiment_breakdown"]["NEUTRAL"],
                NEGATIVE=kw["sentiment_breakdown"]["NEGATIVE"],
            )

            top_keyword = TopKeyword(
                keyword=kw["keyword"],
                count=kw["count"],
                avg_sentiment_score=kw["avg_sentiment_score"],
                aspect=kw["aspect"],
                sentiment_breakdown=sentiment_breakdown,
            )
            keywords.append(top_keyword)

        # Get ranking for specific keywords if requested (comma-separated)
        input_keyword_ranks = None
        if filters.include_rank_for:
            # Parse comma-separated keywords
            search_keywords = [
                kw.strip() for kw in filters.include_rank_for.split(",") if kw.strip()
            ]

            if search_keywords:
                input_keyword_ranks = []
                for search_kw in search_keywords:
                    rank_info = await repo.get_keyword_rank(
                        project_id=filters.project_id,
                        search_keyword=search_kw,
                        brand_name=filters.brand_name,
                        keyword=filters.keyword,
                        from_date=filters.from_date,
                        to_date=filters.to_date,
                        top_limit=filters.limit,
                    )

                    if rank_info:
                        input_keyword_ranks.append(
                            InputKeywordRank(
                                keyword=search_kw,
                                rank=rank_info["rank"],
                                count=rank_info["count"],
                                avg_sentiment_score=rank_info["avg_sentiment_score"],
                                in_top=rank_info["in_top"],
                            )
                        )
                    else:
                        # Keyword not found in DB
                        input_keyword_ranks.append(
                            InputKeywordRank(
                                keyword=search_kw,
                                rank=None,
                                count=0,
                                avg_sentiment_score=0.0,
                                in_top=False,
                            )
                        )

        keywords_data = KeywordsData(
            keywords=keywords,
            input_keyword_ranks=input_keyword_ranks,
        )

        return ApiResponse(
            success=True,
            data=keywords_data,
        )

    except Exception as e:
        logger.error(f"Error in get_top_keywords: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
