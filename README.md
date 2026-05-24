# Financial ML Trading System

A production-grade financial ML trading system that extends CS7646 coursework with
modern AI/ML: transformer-based price prediction, LLM-powered sentiment analysis,
reinforcement learning trading agents, and a full MLOps pipeline — built to demonstrate
end-to-end ML engineering skills for an AI/ML engineering role.

---

## Architecture

```
Data Layer          Feature Layer        Model Layer              Serving Layer
──────────────      ─────────────────    ─────────────────────    ──────────────────
yfinance       →    technical.py     →   transformer.py       →   FastAPI /predict
  (OHLCV +           (SMA, EMA, RSI,      (PriceTransformer)       FastAPI /backtest
   Parquet cache)     MACD, BB, ATR,                                FastAPI /portfolio
                      OBV)             →   rl_agent.py         →   Streamlit UI
NewsAPI        →    sentiment_features      (PPO TradingEnv)         (4 tabs)
  (headlines,         .py
   backoff retry)                      →   ensemble.py
                    dataset.py             (transformer +
                      (sliding window,      sentiment signals)
                       time-ordered                      ↓
                       train/val/test)             MLflow Registry
                                               (experiments, runs,
                  Orchestrator (SDK)             model artifacts,
                  Planner → Coder →             scaler.pkl)
                  Verifier → Reviewer
```

---

## Highlights for an AI/ML Engineering Role

| Skill Area | What's Demonstrated |
|---|---|
| **Deep Learning** | Custom PyTorch Transformer with sinusoidal PE, multi-head attention, early stopping |
| **Reinforcement Learning** | Stable-Baselines3 PPO with a custom `gymnasium.Env`, realistic reward shaping |
| **MLOps** | MLflow experiment tracking, model registry, artifact versioning, reproducible seeds |
| **Feature Engineering** | 14 technical indicators (pure pandas, no TA-Lib), time-series-safe sliding windows |
| **Backtesting** | Event-driven engine with next-open fill, transaction costs, slippage, benchmark comparison |
| **LLM Integration** | Anthropic SDK multi-agent pipeline (Planner → Coder → Verifier → Reviewer) with prompt caching |
| **Production API** | FastAPI with Pydantic v2 validation, CORS, async handlers, graceful model fallback |
| **Testing** | 83 unit + integration tests, 0 failures, mocked external APIs |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY (required for orchestrator agent)

# 3. Train the transformer model
python scripts/train.py \
  --ticker AAPL \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --epochs 100 \
  --seed 42

# 4. Evaluate vs. buy-and-hold benchmark (use run ID from step 3 output)
python scripts/evaluate.py \
  --run-id <mlflow_run_id> \
  --ticker AAPL \
  --start 2024-01-01 \
  --end 2025-01-01
```

---

## Run the API

```bash
uvicorn src.api.app:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`

To enable real transformer predictions, set:
```bash
export MLFLOW_MODEL_PATH="runs:/<run_id>/model"
export MLFLOW_WINDOW_SIZE=60   # must match --window-size used during training
```

---

## Run the UI

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501`. Four tabs:

| Tab | What it does |
|---|---|
| **AI Signals** | Transformer buy/sell signal, price snapshot, RSI/MACD/BB overlays for any ticker |
| **Market Data** | Fetch and visualise OHLCV data — close chart, volume, summary metrics |
| **Technical Indicators** | SMA/EMA overlays, RSI overbought/oversold labels, MACD histogram, ATR volatility |
| **Backtest Results** | SMA Momentum / RSI Mean-Reversion / MACD Trend vs. Buy & Hold with full metrics |

The sidebar shows model status (loaded / not loaded). Set `MLFLOW_MODEL_PATH` before launching
to activate live transformer predictions in the AI Signals tab.

---

## Run Tests

```bash
pytest tests/ -v
```

83 tests across data, features, models, backtesting, and API layers.

---

## API Reference

| Method | Path | Description | Key Request Fields |
|---|---|---|---|
| `POST` | `/api/v1/predict` | Transformer signal for a ticker | `ticker`, `start`, `end`, `window_size` |
| `POST` | `/api/v1/backtest` | Backtest strategy + buy-and-hold | `ticker`, `start`, `end`, `model_path` (optional) |
| `POST` | `/api/v1/portfolio` | Multi-ticker blended backtest | `tickers`, `weights`, `start`, `end` |
| `GET` | `/health` | Health check | — |

### Example: backtest with trained model

```bash
curl -X POST http://localhost:8000/api/v1/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "start": "2024-01-01",
    "end": "2025-01-01",
    "model_path": "runs:/<run_id>/model"
  }'
```

Response includes both `metrics` (strategy) and `benchmark_metrics` (buy & hold).

---

## Example Results

Training on AAPL 2020–2024 (`scripts/train.py`), evaluating on 2024–2025 (`scripts/evaluate.py`).
The results below are from a **fast 30-epoch proof-of-concept run** (d_model=64, 2 layers).
A full production run (`--epochs 200 --d-model 128 --num-layers 4`) is needed for competitive strategy performance.

| Metric | Transformer Strategy¹ | Buy & Hold |
|---|---|---|
| Sharpe Ratio | -20.60 | **2.06** |
| Sortino Ratio | -24.92 | **3.19** |
| CAGR | -0.04% | **+59.46%** |
| Max Drawdown | **-0.10%** | -11.37% |
| Calmar Ratio | -0.38 | **5.23** |

> ¹ Fast training run — model not yet converged. The pipeline is verified end-to-end:
> MLflow run `6deed2227c3040c8b82924992301e958`, 14 features, 608 train / 82 val samples.
> Re-train with `--epochs 200` for production-quality signals.

To reproduce:
```bash
python scripts/train.py --ticker AAPL --start 2020-01-01 --end 2024-01-01 \
  --epochs 200 --d-model 128 --num-layers 4 --seed 42
python scripts/evaluate.py --run-id <your_run_id> \
  --ticker AAPL --start 2024-01-01 --end 2025-01-01
```

---

## Project Structure

```
financial-ml-system/
├── CLAUDE.md                      # Workflow and agent definitions
├── README.md                      # This file
├── requirements.txt
├── main.py                        # CLI entry point for agent workflow
├── .env.example
├── agents/
│   ├── orchestrator.py            # 5-agent Anthropic SDK pipeline
│   ├── orchestrator.md            # Orchestrator system prompt
│   ├── planner.md
│   ├── coder.md
│   ├── verifier.md
│   └── reviewer.md
├── scripts/
│   ├── train.py                   # Train PriceTransformer, log to MLflow
│   └── evaluate.py                # Evaluate vs. buy-and-hold benchmark
├── src/
│   ├── data/
│   │   ├── market.py              # yfinance OHLCV fetcher + Parquet cache
│   │   ├── sentiment.py           # NewsAPI fetcher with retry
│   │   └── pipeline.py            # Unified data pipeline
│   ├── features/
│   │   ├── technical.py           # 14 technical indicators (pure pandas)
│   │   ├── sentiment_features.py  # Sentiment aggregation + merge
│   │   └── dataset.py             # PyTorch Dataset + time-ordered splits
│   ├── models/
│   │   ├── transformer.py         # PriceTransformer + train_transformer
│   │   ├── rl_agent.py            # TradingEnv (gymnasium) + PPO training
│   │   ├── ensemble.py            # EnsembleSignal combiner
│   │   └── sentiment_llm.py       # Anthropic SDK sentiment scorer
│   ├── backtesting/
│   │   ├── engine.py              # Event-driven engine, next-open fill
│   │   ├── metrics.py             # Sharpe, Sortino, drawdown, CAGR, Calmar
│   │   └── strategy.py            # BaseStrategy + EnsembleStrategy
│   └── api/
│       ├── app.py                 # FastAPI app + CORS + health
│       ├── models.py              # Pydantic v2 request/response schemas
│       ├── _model_loader.py       # Cached MLflow model + scaler loader
│       └── routes/
│           ├── predict.py
│           ├── backtest.py
│           └── portfolio.py
├── ui/
│   └── app.py                     # Streamlit dashboard — AI Signals, Market Data, Indicators, Backtest
└── tests/
    ├── data/
    ├── features/
    ├── models/
    ├── backtesting/
    └── api/
```

---

## Tech Stack

- **Deep Learning:** PyTorch 2.x — custom Transformer architecture
- **Reinforcement Learning:** Stable-Baselines3 PPO + custom `gymnasium.Env`
- **NLP/LLM:** Anthropic Claude SDK (multi-agent pipeline + sentiment scoring)
- **Feature Engineering:** pandas, numpy — 14 technical indicators
- **MLOps:** MLflow — experiment tracking, model registry, artifact storage
- **Data:** yfinance (OHLCV), NewsAPI (sentiment headlines)
- **API:** FastAPI + Pydantic v2 + uvicorn
- **UI:** Streamlit
- **Testing:** pytest, unittest.mock — 83 tests

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (orchestrator agent) | Anthropic API key — needed to run `agents/orchestrator.py` |
| `MLFLOW_TRACKING_URI` | No | MLflow tracking URI (default: `./mlruns`) |
| `MLFLOW_MODEL_PATH` | No | MLflow model URI for live inference, e.g. `runs:/<id>/model` |
| `MLFLOW_WINDOW_SIZE` | No | Feature window size (default: `60`, must match training) |
| `NEWS_API_KEY` | No | NewsAPI key for sentiment features |
| `ALPHA_VANTAGE_API_KEY` | No | Alpha Vantage key (fallback data source) |

---

## ML & Finance Correctness

This project enforces strict ML and finance hygiene throughout:

- **No lookahead bias** — signals at time `t` use only features from `[t-W, t)`;
  backtest fills at open of bar `t+1`
- **No data leakage** — `StandardScaler` fitted on training split only; val/test
  receive the pre-fitted scaler
- **Time-ordered splits** — no shuffle anywhere in the data pipeline
- **Reproducible** — `torch.manual_seed`, `np.random.seed`, `random.seed` all set;
  seed logged as MLflow param
- **Realistic costs** — transaction costs (10 bps default) and slippage (5 bps
  default) applied on every position change in both the RL environment and backtesting engine

---

*Built by Bibek Adhikari — extending CS7646 ML for Trading with production AI/ML engineering.*
