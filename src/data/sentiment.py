"""
src/data/sentiment.py — NewsAPI sentiment fetcher with exponential-backoff retry.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_MAX_RETRY_ATTEMPTS = 3
_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_SCHEMA_COLUMNS = ["published_at", "title", "description", "source", "url"]


def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical sentiment schema."""
    df = pd.DataFrame(columns=_SCHEMA_COLUMNS)
    df.index = pd.DatetimeIndex([], name="published_at", tz=None)
    # Drop the column since it becomes the index
    df = df.drop(columns=["published_at"], errors="ignore")
    return df


def _build_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of NewsAPI article dicts into the canonical DataFrame."""
    if not articles:
        return _empty_dataframe()

    rows = []
    for article in articles:
        published_raw = article.get("publishedAt") or article.get("published_at", "")
        source_obj = article.get("source", {})
        source_name = source_obj.get("name", "") if isinstance(source_obj, dict) else str(source_obj)
        rows.append(
            {
                "published_at": published_raw,
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": source_name,
                "url": article.get("url", ""),
            }
        )

    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    # Drop rows where date parsing failed
    df = df.dropna(subset=["published_at"])
    # Convert to tz-naive UTC
    df["published_at"] = df["published_at"].dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.set_index("published_at")
    df.index.name = "published_at"
    df = df.sort_index(ascending=True)
    return df


def fetch_sentiment_data(
    query: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
) -> pd.DataFrame:
    """
    Fetch news articles from NewsAPI and return them as a sentiment DataFrame.

    Parameters
    ----------
    query:
        Search query string, e.g. ``"Apple Inc"``.
    from_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    to_date:
        Inclusive end date in ``YYYY-MM-DD`` format.
    page_size:
        Number of articles per page (max 100 per NewsAPI limits).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["title", "description", "source", "url"]``
        and a UTC-normalized, tz-naive ``DatetimeIndex`` named ``"published_at"``,
        sorted ascending.  Returns an empty DataFrame (correct schema) when no
        articles are found — never raises on zero results.

    Raises
    ------
    EnvironmentError
        If ``NEWS_API_KEY`` is not set in the environment.
    requests.HTTPError
        If the API returns a non-retryable error status after all retry attempts.
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NEWS_API_KEY is not set in the environment. "
            "Please export NEWS_API_KEY=<your_key> before using fetch_sentiment_data."
        )

    params: dict[str, Any] = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "pageSize": page_size,
        "language": "en",
        "sortBy": "publishedAt",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    last_exception: Exception | None = None
    for attempt in range(_MAX_RETRY_ATTEMPTS):
        try:
            response = requests.get(_NEWSAPI_URL, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            last_exception = exc
            wait = 2 ** attempt
            logger.warning(
                "Request error on attempt %d/%d for query=%r: %s. Retrying in %ds.",
                attempt + 1,
                _MAX_RETRY_ATTEMPTS,
                query,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        if response.status_code in _RETRY_STATUS_CODES:
            wait = 2 ** attempt
            logger.warning(
                "HTTP %d on attempt %d/%d for query=%r. Retrying in %ds.",
                response.status_code,
                attempt + 1,
                _MAX_RETRY_ATTEMPTS,
                query,
                wait,
            )
            last_exception = requests.HTTPError(
                f"HTTP {response.status_code}", response=response
            )
            time.sleep(wait)
            continue

        # Non-retryable HTTP error
        if not response.ok:
            response.raise_for_status()

        payload: dict[str, Any] = response.json()
        articles: list[dict[str, Any]] = payload.get("articles", [])
        logger.debug(
            "NewsAPI returned %d article(s) for query=%r (%s → %s).",
            len(articles),
            query,
            from_date,
            to_date,
        )
        return _build_dataframe(articles)

    # All retry attempts exhausted
    if last_exception is not None:
        raise last_exception
    # Fallback — should not be reached, but return empty schema to be safe
    return _empty_dataframe()
