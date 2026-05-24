"""
Financial ML System — Streamlit Dashboard
Ready-to-use financial analysis and AI signal app.
"""

from __future__ import annotations

import os
import sys
import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="Financial ML System",
    page_icon="📈",
    layout="wide",
)

_TODAY = datetime.date.today()
_ONE_YEAR_AGO = _TODAY - datetime.timedelta(days=365)
_TWO_YEARS_AGO = _TODAY - datetime.timedelta(days=730)

POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "SPY", "QQQ"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Financial ML")
    st.caption("AI-powered market analysis")
    st.divider()

    st.markdown("""
    **Tabs**
    - **AI Signals** — transformer buy/sell signals
    - **Market Data** — OHLCV charts & tables
    - **Technical Indicators** — SMA, RSI, MACD overlays
    - **Backtest** — strategy vs. buy-and-hold
    """)
    st.divider()

    # Model status indicator
    model_path = os.environ.get("MLFLOW_MODEL_PATH", "")
    if model_path:
        st.success(f"Model loaded\n`{model_path[:40]}...`" if len(model_path) > 40 else f"Model loaded\n`{model_path}`")
    else:
        st.warning("No model loaded\n\nSet `MLFLOW_MODEL_PATH` env var to enable AI predictions.")

    st.divider()
    st.caption("Built by Bibek Adhikari")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_signals, tab_market, tab_indicators, tab_backtest = st.tabs([
    "AI Signals",
    "Market Data",
    "Technical Indicators",
    "Backtest Results",
])

# ── Tab 1: AI Signals ─────────────────────────────────────────────────────────
with tab_signals:
    st.header("AI Market Signals")
    st.markdown(
        "Select a ticker and date range to get an AI-powered buy/sell signal "
        "from the transformer model alongside key price statistics."
    )

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        sig_ticker_choice = st.selectbox("Ticker", POPULAR_TICKERS + ["Other..."], key="sig_ticker_choice")
        if sig_ticker_choice == "Other...":
            sig_ticker = st.text_input("Enter ticker symbol", value="BRK-B", key="sig_ticker_custom").upper().strip()
        else:
            sig_ticker = sig_ticker_choice
    with col2:
        sig_start = st.date_input("From", value=_ONE_YEAR_AGO, key="sig_start")
    with col3:
        sig_end = st.date_input("To", value=_TODAY, key="sig_end")
    with col4:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyze", type="primary", key="analyze_signal")

    if analyze_btn and sig_ticker:
        with st.spinner(f"Fetching data for {sig_ticker}..."):
            try:
                from src.data.market import fetch_market_data
                from src.features.technical import build_features

                df = fetch_market_data(sig_ticker, str(sig_start), str(sig_end))
                feat_df = build_features(df)

                last_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
                pct_change = (last_close - prev_close) / prev_close * 100
                period_high = float(df["High"].max())
                period_low = float(df["Low"].min())
                avg_volume = int(df["Volume"].mean())

                st.session_state.update({
                    "sig_df": df,
                    "sig_feat_df": feat_df,
                    "sig_label": sig_ticker,
                    "sig_stats": {
                        "last_close": last_close,
                        "pct_change": pct_change,
                        "period_high": period_high,
                        "period_low": period_low,
                        "avg_volume": avg_volume,
                        "data_points": len(df),
                    },
                    "sig_signal": None,
                    "sig_model_error": "",
                })

                # Attempt transformer signal if model is configured
                env_model_path = os.environ.get("MLFLOW_MODEL_PATH", "")
                if env_model_path:
                    try:
                        import pickle
                        import torch
                        from src.api._model_loader import load_model_and_scaler
                        from src.features.dataset import TimeSeriesDataset
                        from src.models.ensemble import generate_ensemble_signal

                        window_size = int(os.environ.get("MLFLOW_WINDOW_SIZE", "60"))
                        ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
                        feature_cols = [c for c in feat_df.columns if c not in ohlcv_cols]

                        model, scaler = load_model_and_scaler(env_model_path)
                        ds = TimeSeriesDataset(
                            feat_df, feature_cols, "Close",
                            window_size=window_size, horizon=1,
                            scaler=scaler, fit_scaler=False,
                        )
                        if len(ds) > 0:
                            x, _ = ds[len(ds) - 1]
                            x = x.unsqueeze(0)
                            with torch.no_grad():
                                pred = float(model(x).item())
                            signal = generate_ensemble_signal(pred, 0.0)
                            st.session_state["sig_signal"] = signal
                    except Exception as exc:
                        st.session_state["sig_model_error"] = str(exc)

            except Exception as e:
                st.error(f"Failed to fetch data: {e}")

    if "sig_df" in st.session_state:
        df = st.session_state["sig_df"]
        feat_df = st.session_state["sig_feat_df"]
        stats = st.session_state["sig_stats"]
        label = st.session_state.get("sig_label", "")
        signal = st.session_state.get("sig_signal")
        model_err = st.session_state.get("sig_model_error", "")

        # ── Price snapshot ────────────────────────────────────────────────────
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        delta_str = f"{stats['pct_change']:+.2f}%"
        c1.metric("Last Close", f"${stats['last_close']:.2f}", delta_str)
        c2.metric("Period High", f"${stats['period_high']:.2f}")
        c3.metric("Period Low", f"${stats['period_low']:.2f}")
        c4.metric("Avg Daily Volume", f"{stats['avg_volume']:,}")
        c5.metric("Trading Days", stats["data_points"])

        # ── Signal panel ─────────────────────────────────────────────────────
        st.divider()
        sig_col, breakdown_col, chart_col = st.columns([1, 1, 3])

        with sig_col:
            st.subheader("Signal")
            if signal is not None:
                score = signal.ensemble_score
                if score > 0.1:
                    st.success(f"### BUY\nScore: `{score:+.3f}`")
                elif score < -0.1:
                    st.error(f"### SELL\nScore: `{score:+.3f}`")
                else:
                    st.info(f"### HOLD\nScore: `{score:+.3f}`")
            else:
                st.info("### —\nNo model loaded")
                if model_err:
                    with st.expander("Model error"):
                        st.code(model_err)
                else:
                    st.caption("Set `MLFLOW_MODEL_PATH` to enable predictions.")

        with breakdown_col:
            st.subheader("Breakdown")
            if signal is not None:
                st.metric("Confidence", f"{signal.confidence:.1%}")
                st.metric("Transformer Score", f"{signal.transformer_score:+.4f}")
                st.metric("Sentiment Score", f"{signal.sentiment_score:+.4f}")
            else:
                st.markdown("""
                | Component | Status |
                |-----------|--------|
                | Transformer | Not loaded |
                | Sentiment | No API key |
                """)

        with chart_col:
            st.subheader(f"{label} — Close Price")
            st.line_chart(df["Close"])

        # ── Technical charts ─────────────────────────────────────────────────
        with st.expander("Technical Indicators", expanded=False):
            ind_col1, ind_col2 = st.columns(2)

            with ind_col1:
                if all(c in feat_df.columns for c in ["sma_20", "sma_50"]):
                    ma_df = feat_df[["Close", "sma_20", "sma_50"]].rename(
                        columns={"Close": "Price", "sma_20": "SMA 20", "sma_50": "SMA 50"}
                    )
                    st.subheader("Price vs Moving Averages")
                    st.line_chart(ma_df)

                if "rsi_14" in feat_df.columns:
                    st.subheader("RSI (14)")
                    rsi_series = feat_df["rsi_14"]
                    st.line_chart(rsi_series)
                    last_rsi = float(rsi_series.iloc[-1])
                    rsi_label = "Overbought (>70)" if last_rsi > 70 else ("Oversold (<30)" if last_rsi < 30 else "Neutral")
                    st.caption(f"Current RSI: **{last_rsi:.1f}** — {rsi_label}")

            with ind_col2:
                if all(c in feat_df.columns for c in ["macd_line", "macd_signal"]):
                    st.subheader("MACD")
                    st.line_chart(feat_df[["macd_line", "macd_signal"]].rename(
                        columns={"macd_line": "MACD", "macd_signal": "Signal"}
                    ))

                if all(c in feat_df.columns for c in ["bb_upper", "bb_lower"]):
                    bb_df = feat_df[["Close", "bb_upper", "bb_lower"]].rename(
                        columns={"Close": "Price", "bb_upper": "BB Upper", "bb_lower": "BB Lower"}
                    )
                    st.subheader("Bollinger Bands")
                    st.line_chart(bb_df)
    else:
        st.info("Select a ticker above and click **Analyze** to get started.")

# ── Tab 2: Market Data ────────────────────────────────────────────────────────
with tab_market:
    st.header("Market Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        md_ticker = st.text_input("Ticker", value="AAPL", key="md_ticker")
    with col2:
        md_start = st.date_input("Start Date", value=_ONE_YEAR_AGO, key="md_start")
    with col3:
        md_end = st.date_input("End Date", value=_TODAY, key="md_end")

    if st.button("Fetch Market Data", key="fetch_market"):
        with st.spinner("Fetching data..."):
            try:
                from src.data.market import fetch_market_data
                df = fetch_market_data(md_ticker, str(md_start), str(md_end))
                st.session_state["market_df"] = df
                st.session_state["market_ticker"] = md_ticker
                st.success(f"Fetched {len(df)} trading days for {md_ticker}")
            except Exception as e:
                st.error(f"Error: {e}")

    if "market_df" in st.session_state:
        df = st.session_state["market_df"]
        ticker_lbl = st.session_state.get("market_ticker", "")

        # Summary metrics
        last = df["Close"].iloc[-1]
        first = df["Close"].iloc[0]
        total_return = (last - first) / first * 100
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Close", f"${last:.2f}")
        m2.metric("Period Return", f"{total_return:+.2f}%")
        m3.metric("52-wk High", f"${df['High'].max():.2f}")
        m4.metric("52-wk Low", f"${df['Low'].min():.2f}")

        st.subheader(f"{ticker_lbl} — Close Price")
        st.line_chart(df["Close"])

        st.subheader("Volume")
        st.bar_chart(df["Volume"])

        with st.expander("Raw OHLCV Table (last 30 days)"):
            st.dataframe(df.tail(30))
    else:
        st.info("Enter a ticker and date range, then click **Fetch Market Data**.")

# ── Tab 3: Technical Indicators ───────────────────────────────────────────────
with tab_indicators:
    st.header("Technical Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        ti_ticker = st.text_input("Ticker", value="AAPL", key="ti_ticker")
    with col2:
        ti_start = st.date_input("Start Date", value=_ONE_YEAR_AGO, key="ti_start")
    with col3:
        ti_end = st.date_input("End Date", value=_TODAY, key="ti_end")

    if st.button("Compute Indicators", key="compute_indicators"):
        with st.spinner("Computing indicators..."):
            try:
                from src.data.market import fetch_market_data
                from src.features.technical import build_features
                raw_df = fetch_market_data(ti_ticker, str(ti_start), str(ti_end))
                feat_df = build_features(raw_df)
                st.session_state["features_df"] = feat_df
                st.session_state["features_ticker"] = ti_ticker
                st.success(f"Computed {len(feat_df.columns)} features for {ti_ticker}")
            except Exception as e:
                st.error(f"Error: {e}")

    if "features_df" in st.session_state:
        feat_df = st.session_state["features_df"]
        ticker_lbl = st.session_state.get("features_ticker", "")

        st.subheader(f"{ticker_lbl} — Price + Moving Averages")
        ma_cols = ["Close"] + [c for c in feat_df.columns if c.startswith("sma_") or c.startswith("ema_")]
        rename_map = {
            "sma_10": "SMA 10", "sma_20": "SMA 20", "sma_50": "SMA 50",
            "ema_12": "EMA 12", "ema_26": "EMA 26",
        }
        chart_df = feat_df[[c for c in ma_cols if c in feat_df.columns]].rename(columns=rename_map)
        st.line_chart(chart_df)

        left_col, right_col = st.columns(2)

        with left_col:
            if "rsi_14" in feat_df.columns:
                st.subheader("RSI (14)")
                st.line_chart(feat_df["rsi_14"].rename("RSI"))
                last_rsi = float(feat_df["rsi_14"].iloc[-1])
                if last_rsi > 70:
                    st.warning(f"RSI {last_rsi:.1f} — Overbought territory (>70)")
                elif last_rsi < 30:
                    st.warning(f"RSI {last_rsi:.1f} — Oversold territory (<30)")
                else:
                    st.success(f"RSI {last_rsi:.1f} — Neutral zone")

            if all(c in feat_df.columns for c in ["bb_upper", "bb_lower", "bb_pct_b"]):
                st.subheader("Bollinger Bands %B")
                st.line_chart(feat_df["bb_pct_b"].rename("%B"))
                st.caption("%B > 1: price above upper band (overbought) | %B < 0: below lower band (oversold)")

        with right_col:
            if all(c in feat_df.columns for c in ["macd_line", "macd_signal", "macd_histogram"]):
                st.subheader("MACD")
                st.line_chart(feat_df[["macd_line", "macd_signal"]].rename(
                    columns={"macd_line": "MACD", "macd_signal": "Signal"}
                ))
                st.subheader("MACD Histogram")
                st.bar_chart(feat_df["macd_histogram"].rename("Histogram"))

            if "atr_14" in feat_df.columns:
                st.subheader("ATR (14) — Volatility")
                st.line_chart(feat_df["atr_14"].rename("ATR"))
    else:
        st.info("Enter a ticker and date range, then click **Compute Indicators**.")

# ── Tab 4: Backtest Results ───────────────────────────────────────────────────
with tab_backtest:
    st.header("Backtest Results")
    col1, col2 = st.columns(2)
    with col1:
        bt_ticker = st.text_input("Ticker", value="AAPL", key="bt_ticker")
        bt_start = st.date_input("Start Date", value=_TWO_YEARS_AGO, key="bt_start")
        bt_end = st.date_input("End Date", value=_TODAY, key="bt_end")
        bt_strategy = st.selectbox(
            "Strategy",
            ["SMA Momentum", "RSI Mean-Reversion", "MACD Trend"],
            key="bt_strategy",
            help=(
                "SMA Momentum: long when Close > SMA-20, short otherwise. "
                "RSI Mean-Reversion: long when RSI < 35, short when RSI > 65. "
                "MACD Trend: long when MACD line > signal, short otherwise."
            ),
        )
    with col2:
        bt_capital = st.number_input("Initial Capital ($)", value=100_000.0, step=10_000.0, key="bt_capital")
        bt_tc = st.number_input("Transaction Cost (bps)", value=10, step=1, key="bt_tc")
        bt_slip = st.number_input("Slippage (bps)", value=5, step=1, key="bt_slip")
        st.caption("1 bps = 0.01%. Typical equity: 5–15 bps per side.")

    if st.button("Run Backtest", key="run_backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                from src.data.market import fetch_market_data
                from src.features.technical import build_features
                from src.backtesting.engine import BacktestEngine, BacktestResult
                from src.backtesting.strategy import EnsembleStrategy
                from src.models.ensemble import EnsembleSignal

                raw_df = fetch_market_data(bt_ticker, str(bt_start), str(bt_end))
                feat_df = build_features(raw_df)

                def _make_signals(df: pd.DataFrame, strategy_name: str) -> list[EnsembleSignal]:
                    sigs = []
                    for i in range(len(df)):
                        row = df.iloc[i]
                        if strategy_name == "SMA Momentum":
                            score = 0.5 if row["Close"] > row["sma_20"] else -0.5
                        elif strategy_name == "RSI Mean-Reversion":
                            rsi = row.get("rsi_14", 50.0)
                            score = 0.5 if rsi < 35 else (-0.5 if rsi > 65 else 0.0)
                        else:  # MACD Trend
                            score = 0.5 if row.get("macd_line", 0) > row.get("macd_signal", 0) else -0.5
                        sigs.append(EnsembleSignal(score, 0.0, score, abs(score)))
                    return sigs

                signals = _make_signals(feat_df, bt_strategy)
                strategy = EnsembleStrategy(signals)
                engine = BacktestEngine(
                    feat_df, strategy,
                    initial_capital=bt_capital,
                    transaction_cost_bps=int(bt_tc),
                    slippage_bps=int(bt_slip),
                )
                result: BacktestResult = engine.run()
                st.session_state["backtest_result"] = result
                st.session_state["backtest_strategy"] = bt_strategy
                st.session_state["backtest_ticker"] = bt_ticker

                bh_sigs = [EnsembleSignal(1.0, 0.0, 1.0, 1.0)] * len(feat_df)
                bh_engine = BacktestEngine(
                    feat_df, EnsembleStrategy(bh_sigs),
                    initial_capital=bt_capital,
                    transaction_cost_bps=int(bt_tc),
                    slippage_bps=int(bt_slip),
                )
                bh_result: BacktestResult = bh_engine.run()
                st.session_state["bh_result"] = bh_result

                st.success(f"Backtest complete — {len(result.trades)} trades executed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if "backtest_result" in st.session_state:
        result = st.session_state["backtest_result"]
        bh_result = st.session_state["bh_result"]
        strat_name = st.session_state.get("backtest_strategy", "")
        bt_lbl = st.session_state.get("backtest_ticker", "")

        st.subheader(f"{bt_lbl} — {strat_name} vs. Buy & Hold")
        chart_df = pd.DataFrame({
            strat_name: result.equity_curve.values,
            "Buy & Hold": bh_result.equity_curve.values,
        }, index=result.equity_curve.index)
        st.line_chart(chart_df)

        equity = result.equity_curve
        drawdown = (equity / equity.cummax() - 1.0) * 100
        st.subheader("Strategy Drawdown (%)")
        st.area_chart(drawdown)

        metric_labels = {
            "sharpe_ratio": "Sharpe Ratio",
            "sortino_ratio": "Sortino Ratio",
            "max_drawdown": "Max Drawdown",
            "cagr": "CAGR",
            "calmar_ratio": "Calmar Ratio",
        }

        st.subheader("Performance Metrics")
        left, right = st.columns(2)
        with left:
            st.subheader(f"Strategy: {strat_name}")
            for k, v in result.metrics.items():
                label = metric_labels.get(k, k)
                fmt = f"{v:.2%}" if k in ("max_drawdown", "cagr") else f"{v:.3f}"
                st.metric(label, fmt)
        with right:
            st.subheader("Benchmark: Buy & Hold")
            for k, v in bh_result.metrics.items():
                label = metric_labels.get(k, k)
                fmt = f"{v:.2%}" if k in ("max_drawdown", "cagr") else f"{v:.3f}"
                st.metric(label, fmt)

        st.subheader("Head-to-Head Comparison")

        def _beats(metric: str, strat_val: float, bh_val: float) -> str:
            if metric == "max_drawdown":
                return "✓" if strat_val > bh_val else "✗"
            return "✓" if strat_val > bh_val else "✗"

        cmp_data = {
            "Metric": list(metric_labels.values()),
            "Strategy": [f"{v:.2%}" if k in ("max_drawdown", "cagr") else f"{v:.3f}"
                         for k, v in result.metrics.items()],
            "Buy & Hold": [f"{v:.2%}" if k in ("max_drawdown", "cagr") else f"{v:.3f}"
                           for k, v in bh_result.metrics.items()],
            "Strategy Wins": [_beats(k, sv, bv)
                               for (k, sv), (_, bv) in zip(result.metrics.items(), bh_result.metrics.items())],
        }
        st.dataframe(pd.DataFrame(cmp_data).set_index("Metric"))

        if not result.trades.empty:
            with st.expander(f"Trade Log ({len(result.trades)} trades)"):
                st.dataframe(result.trades)
    else:
        st.info("Select a ticker and strategy, then click **Run Backtest**.")
