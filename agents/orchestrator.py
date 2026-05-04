"""
Orchestrator Agent — claude-opus-4-7
Routes all tasks to sub-agents. Never writes code.
All sub-agent calls originate here.
"""

import os
import anthropic
from .planner import run_planner
from .coder import run_coder
from .reviewer import run_reviewer

MODEL = "claude-opus-4-7"
MAX_REVIEW_LOOPS = 2

SYSTEM_PROMPT = """You are the Orchestrator of a financial ML engineering team.
Your job is coordinate work between specialized agents: Planner, Coder, and Reviewer.

Rules you must follow at all times:
- You NEVER write code. Not even a single line or code block.
- You make ALL calls to sub-agents via the tools provided.
- You decompose the user's request and route it appropriately.
- After receiving a plan, you pass it to the Coder.
- After code is written, you always send it to the Reviewer.
- If the Reviewer flags critical issues, you route feedback back to the Coder.
- You return a clear summary to the user after the workflow completes.

Your responses are coordination text only: status updates, summaries, and tool calls."""

TOOLS = [
    {
        "name": "call_planner",
        "description": (
            "Call the Planner agent to break a task into a structured, numbered plan. "
            "Use this first for any new feature or task request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The high-level task or feature to plan.",
                },
                "context": {
                    "type": "string",
                    "description": "Any relevant context: project structure, constraints, prior decisions.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "call_coder",
        "description": (
            "Call the Coder agent to write or update code. "
            "Provide the full plan and any reviewer feedback so the Coder has complete context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific coding task to implement.",
                },
                "plan": {
                    "type": "string",
                    "description": "The structured plan from the Planner.",
                },
                "reviewer_feedback": {
                    "type": "string",
                    "description": "Feedback from the Reviewer to address. Empty string if first attempt.",
                },
            },
            "required": ["task", "plan"],
        },
    },
    {
        "name": "call_reviewer",
        "description": (
            "Call the Reviewer agent to review code for correctness, security, "
            "and ML best practices. Always call this after the Coder produces output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code produced by the Coder agent.",
                },
                "requirements": {
                    "type": "string",
                    "description": "The original task requirements the code should satisfy.",
                },
            },
            "required": ["code", "requirements"],
        },
    },
]


def _execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "call_planner":
        return run_planner(
            task=tool_input["task"],
            context=tool_input.get("context", ""),
        )
    elif tool_name == "call_coder":
        return run_coder(
            task=tool_input["task"],
            plan=tool_input.get("plan", ""),
            reviewer_feedback=tool_input.get("reviewer_feedback", ""),
        )
    elif tool_name == "call_reviewer":
        return run_reviewer(
            code=tool_input["code"],
            requirements=tool_input.get("requirements", ""),
        )
    return f"Unknown tool: {tool_name}"


def run_orchestrator(user_request: str) -> dict:
    """
    Run the full agentic workflow for a user request.
    Returns a dict with keys: plan, code, review, summary.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": user_request}]
    results = {"plan": "", "code": "", "review": "", "summary": ""}

    for _ in range(20):  # safety cap on total turns
        response = client.messages.create(
            model=MODEL,
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    results["summary"] = block.text
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_output = _execute_tool(block.name, block.input)

            # Track results by agent type
            if block.name == "call_planner":
                results["plan"] = tool_output
            elif block.name == "call_coder":
                results["code"] = tool_output
            elif block.name == "call_reviewer":
                results["review"] = tool_output

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return results
