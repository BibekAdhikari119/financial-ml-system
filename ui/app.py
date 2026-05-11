"""
Financial ML System — Streamlit Dashboard
Entry point for the agentic workflow UI.
"""

import os
import sys
import datetime

import pandas as pd
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
    | Verifier | Sonnet 4.6 |
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_agent, tab_market, tab_indicators, tab_backtest = st.tabs([
    "Agent Workspace",
    "Market Data",
    "Technical Indicators",
    "Backtest Results",
])

# ── Tab 1: Agent Workspace ────────────────────────────────────────────────────
with tab_agent:
    st.title("Financial ML Agent Workspace")
    st.markdown(
        "Describe a feature or task. The Orchestrator will coordinate Planner → Coder → Verifier → Reviewer."
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

    # ── Results ───────────────────────────────────────────────────────────────
    if run_button and user_request.strip():
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Please provide your Anthropic API key in the sidebar.")
            st.stop()

        plan_col, code_col, verify_col, review_col = st.columns([1, 1.4, 1, 1])

        with plan_col:
            plan_placeholder = st.empty()
            plan_placeholder.info("Waiting for Planner...")

        with code_col:
            code_placeholder = st.empty()
            code_placeholder.info("Waiting for Coder...")

        with verify_col:
            verify_placeholder = st.empty()
            verify_placeholder.info("Waiting for Verifier...")

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

        # Verification panel
        with verify_col:
            verify_placeholder.empty()
            st.subheader("Verifier Output")
            if results["verification"]:
                verify_text = results["verification"]
                if "FAIL" in verify_text.upper():
                    st.error(verify_text)
                elif "WARNING" in verify_text.upper():
                    st.warning(verify_text)
                else:
                    st.success(verify_text)
            else:
                st.warning("No verification generated.")

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

# ── Tab 2: Market Data ────────────────────────────────────────────────────────
with tab_market:
    st.header("Market Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        md_ticker = st.text_input("Ticker", value="AAPL", key="md_ticker")
    with col2:
        md_start = st.date_input("Start Date", value=datetime.date(2023, 1, 1), key="md_start")
    with col3:
        md_end = st.date_input("End Date", value=datetime.date(2023, 12, 31), key="md_end")

    if st.button("Fetch Market Data", key="fetch_market"):
        with st.spinner("Fetching data..."):
            try:
                from src.data.market import fetch_market_data
                df = fetch_market_data(md_ticker, str(md_start), str(md_end))
                st.session_state["market_df"] = df
                st.success(f"Fetched {len(df)} rows for {md_ticker}")
            except Exception as e:
                st.error(f"Error: {e}")

    if "market_df" in st.session_state:
        df = st.session_state["market_df"]
        st.subheader("OHLCV Data")
        st.dataframe(df.tail(20))

        # Simple close price chart
        st.subheader("Close Price")
        st.line_chart(df["Close"])

        # Volume bar chart
        st.subheader("Volume")
        st.bar_chart(df["Volume"])
    else:
        st.info("Enter a ticker and date range, then click 'Fetch Market Data'.")

# ── Tab 3: Technical Indicators ───────────────────────────────────────────────
with tab_indicators:
    st.header("Technical Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        ti_ticker = st.text_input("Ticker", value="AAPL", key="ti_ticker")
    with col2:
        ti_start = st.date_input("Start Date", value=datetime.date(2023, 1, 1), key="ti_start")
    with col3:
        ti_end = st.date_input("End Date", value=datetime.date(2023, 12, 31), key="ti_end")

    if st.button("Compute Indicators", key="compute_indicators"):
        with st.spinner("Computing indicators..."):
            try:
                from src.data.market import fetch_market_data
                from src.features.technical import build_features
                raw_df = fetch_market_data(ti_ticker, str(ti_start), str(ti_end))
                feat_df = build_features(raw_df)
                st.session_state["features_df"] = feat_df
                st.success(f"Computed {len(feat_df.columns)} features for {ti_ticker}")
            except Exception as e:
                st.error(f"Error: {e}")

    if "features_df" in st.session_state:
        feat_df = st.session_state["features_df"]

        st.subheader("Price + Moving Averages")
        price_cols = ["Close"] + [c for c in feat_df.columns if c.startswith("sma_") or c.startswith("ema_")]
        st.line_chart(feat_df[[c for c in price_cols if c in feat_df.columns]])

        if "rsi_14" in feat_df.columns:
            st.subheader("RSI (14)")
            st.line_chart(feat_df["rsi_14"])

        if "macd_line" in feat_df.columns:
            st.subheader("MACD")
            st.line_chart(feat_df[["macd_line", "macd_signal"]])
    else:
        st.info("Enter a ticker and date range, then click 'Compute Indicators'.")

# ── Tab 4: Backtest Results ───────────────────────────────────────────────────
with tab_backtest:
    st.header("Backtest Results")
    col1, col2 = st.columns(2)
    with col1:
        bt_ticker = st.text_input("Ticker", value="AAPL", key="bt_ticker")
        bt_start = st.date_input("Start Date", value=datetime.date(2022, 1, 1), key="bt_start")
        bt_end = st.date_input("End Date", value=datetime.date(2023, 12, 31), key="bt_end")
    with col2:
        bt_capital = st.number_input("Initial Capital ($)", value=100_000.0, step=10_000.0, key="bt_capital")
        bt_tc = st.number_input("Transaction Cost (bps)", value=10, step=1, key="bt_tc")
        bt_slip = st.number_input("Slippage (bps)", value=5, step=1, key="bt_slip")

    if st.button("Run Backtest", key="run_backtest"):
        with st.spinner("Running backtest..."):
            try:
                from src.data.market import fetch_market_data
                from src.features.technical import build_features
                from src.backtesting.engine import BacktestEngine, BacktestResult
                from src.backtesting.strategy import EnsembleStrategy
                from src.models.ensemble import EnsembleSignal

                raw_df = fetch_market_data(bt_ticker, str(bt_start), str(bt_end))
                feat_df = build_features(raw_df)
                n = len(feat_df)
                signals = [EnsembleSignal(0.0, 0.0, 0.0, 0.0) for _ in range(n)]
                strategy = EnsembleStrategy(signals)
                engine = BacktestEngine(
                    feat_df, strategy,
                    initial_capital=bt_capital,
                    transaction_cost_bps=int(bt_tc),
                    slippage_bps=int(bt_slip),
                )
                result: BacktestResult = engine.run()
                st.session_state["backtest_result"] = result
                st.success("Backtest complete!")
            except Exception as e:
                st.error(f"Error: {e}")

    if "backtest_result" in st.session_state:
        result = st.session_state["backtest_result"]

        st.subheader("Equity Curve")
        st.line_chart(result.equity_curve)

        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame(
            list(result.metrics.items()),
            columns=["Metric", "Value"]
        ).set_index("Metric")
        st.dataframe(metrics_df)
    else:
        st.info("Configure parameters and click 'Run Backtest'.")
