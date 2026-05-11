"""
src/models/sentiment_llm.py — Anthropic SDK-based financial sentiment scorer.
"""
from __future__ import annotations

import os

import anthropic


def score_sentiment(
    texts: list[str],
    model: str | None = None,
    max_tokens: int = 10,
) -> list[float]:
    """
    Score each text with an Anthropic LLM, returning floats in [-1.0, 1.0].

    When more than 10 texts are provided the system message uses prompt caching
    via the beta messages API to reduce latency and cost on repeated calls.

    Args:
        texts: List of financial news / document strings to score.
        model: Anthropic model ID.  Defaults to the SENTIMENT_MODEL env var,
               falling back to 'claude-haiku-4-5-20251001'.
        max_tokens: Maximum tokens to generate per request (a single number is
                    all that is needed).

    Returns:
        A list of sentiment scores, one per input text, clipped to [-1.0, 1.0].
        On parse failure for any text the corresponding score is 0.0 (neutral).

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set."
        )

    sentiment_model = model or os.environ.get(
        "SENTIMENT_MODEL", "claude-haiku-4-5-20251001"
    )

    client = anthropic.Anthropic(api_key=api_key)

    system_text = (
        "You are a financial sentiment analyzer. For the given text, output ONLY a single "
        "decimal number between -1.0 (very negative) and 1.0 (very positive). No explanation."
    )

    use_caching = len(texts) > 10

    scores: list[float] = []
    for text in texts:
        try:
            if use_caching:
                # Use beta prompt-caching API so the system message is cached
                # across the batch of calls.
                response = client.beta.messages.create(
                    model=sentiment_model,
                    max_tokens=max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system_text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": text}],
                    betas=["prompt-caching-2024-07-31"],
                )
            else:
                response = client.messages.create(
                    model=sentiment_model,
                    max_tokens=max_tokens,
                    system=system_text,
                    messages=[{"role": "user", "content": text}],
                )

            raw = response.content[0].text.strip()
            score = float(raw)
        except Exception:
            score = 0.0

        # Clip to valid range
        score = max(-1.0, min(1.0, score))
        scores.append(score)

    return scores
