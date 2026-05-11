"""
agents/orchestrator.py

Coordinates Planner → Coder → Reviewer → Summary pipeline via Anthropic SDK.
"""

import os
from pathlib import Path

import anthropic


def _load_system_prompt(agent_name: str) -> str:
    """
    Read the agent's .md file and return the body after the YAML frontmatter.

    The frontmatter is delimited by '---' lines at the top of the file.
    Everything after the second '---' line is returned as the system prompt.
    Returns an empty string if the file is not found.
    """
    md_path = Path(__file__).parent / f"{agent_name}.md"
    try:
        raw = md_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

    lines = raw.splitlines()
    separator_count = 0
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            separator_count += 1
            if separator_count == 2:
                body_start = i + 1
                break

    if separator_count < 2:
        # No frontmatter detected; return the whole file
        return raw

    return "\n".join(lines[body_start:]).lstrip("\n")


def _make_cached_system(text: str) -> list[dict]:
    """Return a cached system message list for the beta prompt-caching API."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def run_orchestrator(user_request: str) -> dict[str, str]:
    """
    Coordinates Planner → Coder → Verifier → Reviewer → Summary pipeline via Anthropic SDK.

    Args:
        user_request: The raw request string from the user.

    Returns:
        A dict with keys: "plan", "code", "verification", "review", "summary".

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it in your shell or add it to your .env file."
        )

    client = anthropic.Anthropic()

    plan: str = ""
    code: str = ""
    verification: str = ""
    review: str = ""
    summary: str = ""

    # ── Call 1: Planner ────────────────────────────────────────────────────────
    planner_system = _load_system_prompt("planner")
    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_make_cached_system(planner_system),
            messages=[{"role": "user", "content": user_request}],
            betas=["prompt-caching-2024-07-31"],
        )
        plan = response.content[0].text
    except anthropic.APIError as e:
        plan = f"[API Error: {e}]"

    # ── Call 2: Coder ──────────────────────────────────────────────────────────
    coder_system = _load_system_prompt("coder")
    coder_user = (
        f"## Implementation Plan\n\n{plan}\n\n## Original Request\n\n{user_request}"
    )
    try:
        response = client.beta.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=_make_cached_system(coder_system),
            messages=[{"role": "user", "content": coder_user}],
            betas=["prompt-caching-2024-07-31"],
        )
        code = response.content[0].text
    except anthropic.APIError as e:
        code = f"[API Error: {e}]"

    # ── Call 3: Verifier ───────────────────────────────────────────────────────
    verifier_system = _load_system_prompt("verifier")
    verifier_user = (
        f"## Code to Verify\n\n{code}\n\n## Original Request\n\n{user_request}"
    )
    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_make_cached_system(verifier_system),
            messages=[{"role": "user", "content": verifier_user}],
            betas=["prompt-caching-2024-07-31"],
        )
        verification = response.content[0].text
    except anthropic.APIError as e:
        verification = f"[API Error: {e}]"

    # ── Call 4: Reviewer ───────────────────────────────────────────────────────
    reviewer_system = _load_system_prompt("reviewer")
    reviewer_user = (
        f"## Code to Review\n\n{code}\n\n## Verification Results\n\n{verification}\n\n## Original Request\n\n{user_request}"
    )
    try:
        response = client.beta.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=_make_cached_system(reviewer_system),
            messages=[{"role": "user", "content": reviewer_user}],
            betas=["prompt-caching-2024-07-31"],
        )
        review = response.content[0].text
    except anthropic.APIError as e:
        review = f"[API Error: {e}]"

    # ── Call 5: Summary ────────────────────────────────────────────────────────
    summary_system = (
        "You are the Orchestrator. Provide a brief summary of the agent workflow "
        "outcome in 3-5 bullet points."
    )
    summary_user = f"Plan:\n{plan}\n\nCode:\n{code}\n\nVerification:\n{verification}\n\nReview:\n{review}"
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=summary_system,
            messages=[{"role": "user", "content": summary_user}],
        )
        summary = response.content[0].text
    except anthropic.APIError as e:
        summary = f"[API Error: {e}]"

    return {"plan": plan, "code": code, "verification": verification, "review": review, "summary": summary}
