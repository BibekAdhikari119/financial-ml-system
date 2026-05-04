# Financial ML System — Claude Agentic Workflow

## Project Purpose
A production-grade financial ML trading system built by Bibek Adhikari.
Extends CS7646 coursework with modern AI/ML: transformer-based signals,
LLM-powered sentiment analysis, RL trading agents, and MLOps tooling.

---

## Agent Roles & Models

| Agent | Model | Responsibility |
|-------|-------|---------------|
| **Orchestrator** | `claude-opus-4-7` | Routes all tasks to sub-agents. Makes ALL calls. Never writes code. |
| **Planner** | `claude-sonnet-4-6` | Breaks tasks into structured, actionable plans. Never writes code. |
| **Coder** | `claude-opus-4-7` | The ONLY agent that writes or modifies code. |
| **Reviewer** | `claude-haiku-4-5-20251001` | Reviews code for correctness, security, and ML best practices. Never writes code. |

---

## Workflow Rules

1. **All sub-agent calls go through the Orchestrator.** No agent calls another agent directly.
2. **Only the Coder writes code.** Orchestrator, Planner, and Reviewer output text only.
3. **Canonical flow for every feature:**
   ```
   User Request
       → Orchestrator (decomposes task)
           → Planner (structured plan)
           → Coder (implements plan)
           → Reviewer (reviews output)
           → Coder (addresses review feedback, if any)
       → Orchestrator (returns final result to user)
   ```
4. **Review loops:** If Reviewer flags critical issues, Orchestrator sends feedback back to Coder. Max 2 revision loops before escalating to user.
5. **Orchestrator never produces code blocks** in its responses — only coordination text and tool calls.
6. **Planner output format:** Always a numbered list of discrete, implementable tasks with clear acceptance criteria.
7. **Reviewer output format:** Structured sections — `[PASS/FAIL]`, Issues (critical / warning / suggestion), and a brief recommendation.

---

## Project Structure

```
financial-ml-system/
├── CLAUDE.md                  # This file — workflow definition
├── agents/
│   ├── orchestrator.py        # Orchestrator: routes tasks, calls sub-agents
│   ├── planner.py             # Planner: produces structured plans
│   ├── coder.py               # Coder: writes all code
│   └── reviewer.py            # Reviewer: reviews code output
├── src/
│   ├── data/                  # Data ingestion, Yahoo Finance, news APIs
│   ├── features/              # Technical indicators, sentiment features
│   ├── models/                # RL agent, transformer model, ensemble
│   ├── backtesting/           # Backtesting engine, performance metrics
│   └── api/                   # FastAPI service layer
├── ui/
│   └── app.py                 # Streamlit dashboard
├── tests/                     # Unit and integration tests
├── .env.example               # Environment variable template
└── requirements.txt
```

---

## Tech Stack

- **ML/AI:** PyTorch, Stable-Baselines3 (RL), Hugging Face Transformers
- **Data:** yfinance, Alpha Vantage / NewsAPI for sentiment
- **MLOps:** MLflow for experiment tracking
- **Backend:** FastAPI
- **UI:** Streamlit
- **Agents:** Anthropic Python SDK (`anthropic`)

---

## Model Notes

Orchestrator and Coder both use `claude-opus-4-7` — the most capable model —
because coordination decisions and code generation are the highest-stakes tasks.
Planner uses `claude-sonnet-4-6` for structured reasoning. Reviewer uses
`claude-haiku-4-5-20251001` for fast, cost-efficient code review passes.

---

## Environment Variables

```
ANTHROPIC_API_KEY=        # Required for all agents
ALPHA_VANTAGE_API_KEY=    # Market data (optional, yfinance is fallback)
NEWS_API_KEY=             # Sentiment data
MLFLOW_TRACKING_URI=      # Default: ./mlruns
```
