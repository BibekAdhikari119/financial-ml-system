#!/usr/bin/env python3
"""Entry point for the financial ML multi-agent workflow."""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

from agents.orchestrator import run_orchestrator


def main() -> None:
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        print("Financial ML System — Multi-Agent Workflow")
        print("Enter your request (Ctrl+C to exit):")
        try:
            user_request = input("> ").strip()
        except KeyboardInterrupt:
            print()
            sys.exit(0)
        if not user_request:
            print("No request provided.")
            sys.exit(1)

    print(f"\nRunning agents for: {user_request}\n")
    print("=" * 60)

    results = run_orchestrator(user_request)

    sections = [
        ("PLAN",    results["plan"]),
        ("CODE",    results["code"]),
        ("REVIEW",  results["review"]),
        ("SUMMARY", results["summary"]),
    ]
    for title, content in sections:
        if content:
            print(f"\n{title}\n" + "-" * 40)
            print(content)


if __name__ == "__main__":
    main()
