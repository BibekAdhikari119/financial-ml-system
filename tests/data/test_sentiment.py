"""
tests/data/test_sentiment.py — pytest tests for fetch_sentiment_data.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.sentiment import fetch_sentiment_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_article(published_at: str, title: str = "Test") -> dict:
    return {
        "publishedAt": published_at,
        "title": title,
        "description": "A test article.",
        "source": {"name": "Reuters"},
        "url": "https://example.com",
    }


def _mock_response(status_code: int, articles: list | None = None) -> MagicMock:
    """Return a mock requests.Response with the given status code and articles."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    payload = {"status": "ok", "totalResults": len(articles or []), "articles": articles or []}
    resp.json.return_value = payload
    resp.raise_for_status.side_effect = (
        None if resp.ok else __import__("requests").HTTPError(response=resp)
    )
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMissingApiKeyRaises:
    """fetch_sentiment_data must raise EnvironmentError when NEWS_API_KEY is absent."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="NEWS_API_KEY"):
            fetch_sentiment_data("Apple", "2024-01-01", "2024-01-07")


class TestRetryOn429:
    """fetch_sentiment_data must retry on HTTP 429 and succeed on the third attempt."""

    def test_retry_on_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEWS_API_KEY", "test-key-retry")

        articles = [_make_article("2024-01-03T10:00:00Z", "Retry success")]
        responses = [
            _mock_response(429),
            _mock_response(429),
            _mock_response(200, articles),
        ]

        call_count = {"n": 0}

        def _mock_get(*args, **kwargs):  # type: ignore[no-untyped-def]
            resp = responses[call_count["n"]]
            call_count["n"] += 1
            return resp

        with patch("src.data.sentiment.time.sleep"):  # skip real waits
            with patch("requests.get", side_effect=_mock_get):
                result = fetch_sentiment_data("Apple", "2024-01-01", "2024-01-07")

        assert call_count["n"] == 3, f"Expected 3 calls (2 retries + 1 success), got {call_count['n']}"
        assert not result.empty, "Result should not be empty after successful retry"
        assert "title" in result.columns


class TestEmptyResultReturnsCorrectSchema:
    """Zero articles must return an empty DataFrame with the correct column schema."""

    def test_empty_result_returns_correct_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEWS_API_KEY", "test-key-empty")
        mock_resp = _mock_response(200, articles=[])

        with patch("requests.get", return_value=mock_resp):
            result = fetch_sentiment_data("UnknownXYZCorp", "2024-01-01", "2024-01-07")

        assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
        assert result.empty, "DataFrame should be empty for zero articles"
        # Schema: index named 'published_at', columns title/description/source/url
        assert result.index.name == "published_at"
        for col in ("title", "description", "source", "url"):
            assert col in result.columns, f"Column '{col}' missing from empty result"


class TestReturnsSortedAscending:
    """Articles returned out of order must be sorted ascending by published_at."""

    def test_returns_sorted_ascending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEWS_API_KEY", "test-key-sort")

        # Intentionally out of order
        articles = [
            _make_article("2024-01-05T12:00:00Z", "Later article"),
            _make_article("2024-01-03T08:00:00Z", "Earlier article"),
            _make_article("2024-01-04T15:30:00Z", "Middle article"),
        ]
        mock_resp = _mock_response(200, articles)

        with patch("requests.get", return_value=mock_resp):
            result = fetch_sentiment_data("Apple", "2024-01-01", "2024-01-07")

        assert result.index.is_monotonic_increasing, (
            f"Index must be sorted ascending; got {list(result.index)}"
        )
        assert result.iloc[0]["title"] == "Earlier article"
        assert result.iloc[-1]["title"] == "Later article"


class TestIndexIsDatetimeUTC:
    """Index must be a tz-naive DatetimeIndex (UTC-normalized)."""

    def test_index_is_tz_naive_datetime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEWS_API_KEY", "test-key-tz")
        articles = [_make_article("2024-01-03T10:00:00Z", "TZ test")]
        mock_resp = _mock_response(200, articles)

        with patch("requests.get", return_value=mock_resp):
            result = fetch_sentiment_data("Apple", "2024-01-01", "2024-01-07")

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is None, "Index must be tz-naive (UTC-normalized)"


class TestRetryExhaustedRaises:
    """All retry attempts returning 429 must propagate an HTTPError."""

    def test_all_retries_exhausted_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NEWS_API_KEY", "test-key-exhaust")

        import requests as req

        responses = [_mock_response(429)] * 3

        call_count = {"n": 0}

        def _mock_get(*args, **kwargs):  # type: ignore[no-untyped-def]
            resp = responses[min(call_count["n"], 2)]
            call_count["n"] += 1
            return resp

        with patch("src.data.sentiment.time.sleep"):
            with patch("requests.get", side_effect=_mock_get):
                with pytest.raises(req.HTTPError):
                    fetch_sentiment_data("Apple", "2024-01-01", "2024-01-07")
