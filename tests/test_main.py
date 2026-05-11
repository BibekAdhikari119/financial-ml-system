"""
tests/test_main.py

Smoke tests for the main.py entry point and orchestrator contract.
"""

import os
import subprocess
import sys


def test_main_missing_key() -> None:
    """Verify main.py exits with an error message when ANTHROPIC_API_KEY is unset.

    load_dotenv() only sets variables that are not already present in the environment,
    so setting ANTHROPIC_API_KEY to an empty string prevents the .env file from
    supplying a value and exercises the guard in main.py.
    """
    env = {k: v for k, v in os.environ.items()}
    env["ANTHROPIC_API_KEY"] = ""  # empty string → falsy; load_dotenv won't override
    result = subprocess.run(
        [sys.executable, "main.py", "test request"],
        capture_output=True,
        text=True,
        env=env,
        cwd="/Users/bibek/financial-ml-system",
    )
    assert result.returncode != 0
    assert "ANTHROPIC_API_KEY" in result.stdout or "ANTHROPIC_API_KEY" in result.stderr


def test_orchestrator_contract() -> None:
    """Verify run_orchestrator returns dict with all 4 expected keys (with mocked API)."""
    import anthropic
    from unittest.mock import MagicMock, patch

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="mocked response")]

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("anthropic.Anthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create.return_value = mock_message
            instance.beta.messages.create.return_value = mock_message

            from agents.orchestrator import run_orchestrator
            result = run_orchestrator("build a momentum strategy")

    assert set(result.keys()) == {"plan", "code", "verification", "review", "summary"}
    assert isinstance(result["plan"], str)
    assert isinstance(result["code"], str)
    assert isinstance(result["verification"], str)
    assert isinstance(result["review"], str)
    assert isinstance(result["summary"], str)
