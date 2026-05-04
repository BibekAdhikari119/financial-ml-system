"""
Financial ML System — Streamlit Dashboard
Entry point for the agentic workflow UI.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import run_orchestrator

st.set_page_config(
    page_title="Financial ML System",
    page_icon="📈",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Financial ML System")
    st.caption("Agentic AI/ML Engineering Workflow")
    st.divider()

    st.subheader("Agent Pipeline")
    st.markdown("""
    | Agent | Model |
    |-------|-------|
    | Orchestrator | Opus 4.7 |
    | Planner | Sonnet 4.6 |
    | Coder | Opus 4.7 |
    | Reviewer | Haiku 4.5 |
    """)
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Set ANTHROPIC_API_KEY in your environment or enter it here.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.caption("Built by Bibek Adhikari | CS7646 → Production ML")

# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("Financial ML Agent Workspace")
st.markdown(
    "Describe a feature or task. The Orchestrator will coordinate Planner → Coder → Reviewer."
)

EXAMPLE_PROMPTS = [
    "Build a technical indicator feature pipeline using SMA ratio, Bollinger Bands %, and CCI for JPM stock.",
    "Implement a PPO reinforcement learning trading agent using Stable-Baselines3.",
    "Create a sentiment feature extractor that pulls recent news headlines and scores them with a Hugging Face model.",
    "Write a backtesting engine that computes Sharpe ratio, max drawdown, and cumulative returns.",
]

with st.expander("Example prompts"):
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=prompt):
            st.session_state["user_request"] = prompt

user_request = st.text_area(
    "Your request",
    value=st.session_state.get("user_request", ""),
    height=120,
    placeholder="e.g. Build a momentum indicator feature pipeline for AAPL...",
)

run_button = st.button("Run Agent Workflow", type="primary", disabled=not api_key)

# ── Results ───────────────────────────────────────────────────────────────────
if run_button and user_request.strip():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Please provide your Anthropic API key in the sidebar.")
        st.stop()

    plan_col, code_col, review_col = st.columns([1, 1.4, 1])

    with plan_col:
        plan_placeholder = st.empty()
        plan_placeholder.info("Waiting for Planner...")

    with code_col:
        code_placeholder = st.empty()
        code_placeholder.info("Waiting for Coder...")

    with review_col:
        review_placeholder = st.empty()
        review_placeholder.info("Waiting for Reviewer...")

    summary_placeholder = st.empty()

    with st.spinner("Orchestrator is coordinating agents..."):
        try:
            results = run_orchestrator(user_request.strip())
        except Exception as e:
            st.error(f"Agent workflow failed: {e}")
            st.stop()

    # Plan panel
    with plan_col:
        plan_placeholder.empty()
        st.subheader("Planner Output")
        if results["plan"]:
            st.markdown(results["plan"])
        else:
            st.warning("No plan generated.")

    # Code panel
    with code_col:
        code_placeholder.empty()
        st.subheader("Coder Output")
        if results["code"]:
            st.code(results["code"], language="python")
        else:
            st.warning("No code generated.")

    # Review panel
    with review_col:
        review_placeholder.empty()
        st.subheader("Reviewer Output")
        if results["review"]:
            review_text = results["review"]
            if "FAIL" in review_text.upper():
                st.error(review_text)
            elif "WARNING" in review_text.upper():
                st.warning(review_text)
            else:
                st.success(review_text)
        else:
            st.warning("No review generated.")

    # Orchestrator summary
    summary_placeholder.empty()
    if results["summary"]:
        st.divider()
        st.subheader("Orchestrator Summary")
        st.markdown(results["summary"])

elif run_button and not user_request.strip():
    st.warning("Please enter a request before running.")
