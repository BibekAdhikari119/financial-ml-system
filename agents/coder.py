"""
Coder Agent — claude-opus-4-7
The ONLY agent that writes code. Called exclusively by the Orchestrator.
"""

import os
import anthropic

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are the Coder for a financial ML engineering project.

You are the ONLY agent in this system that writes code. Your output is always
complete, runnable Python code with minimal but precise comments.

Project context:
- Stack: Python, PyTorch, Stable-Baselines3, Hugging Face Transformers, yfinance,
  MLflow, FastAPI, Streamlit, Anthropic SDK
- Style: PEP 8, type hints on all function signatures, docstrings only when the
  purpose is non-obvious
- ML rules: no data leakage, reproducible seeds, proper train/val/test splits
- Finance rules: no lookahead bias, account for transaction costs, realistic slippage

When given reviewer feedback:
- Address every critical issue
- Address warnings unless you have a clear reason not to (state the reason in a comment)
- You may skip suggestions if they conflict with project constraints

Output format:
- Produce complete file contents, not fragments
- Begin each file with its path as a comment: # path/to/file.py
- Separate multiple files with --- FILE SEPARATOR ---"""


def run_coder(task: str, plan: str, reviewer_feedback: str = "") -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    parts = [f"Task: {task}"]
    if plan:
        parts.append(f"Plan:\n{plan}")
    if reviewer_feedback:
        parts.append(f"Reviewer Feedback to Address:\n{reviewer_feedback}")

    response = client.messages.create(
        model=MODEL,
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )

    return response.content[0].text
