# Financial ML System — Claude Agentic Workflow

## Project Purpose
A production-grade financial ML trading system built by Bibek Adhikari.
Extends CS7646 coursework with modern AI/ML: transformer-based signals,
LLM-powered sentiment analysis, RL trading agents, and MLOps tooling.

---

## Agent Roles & Models

| Agent | Model | Color | Responsibility |
|-------|-------|-------|---------------|
| **Orchestrator** | `claude-opus-4-7` | purple | Routes all tasks to sub-agents. Makes ALL calls. Never writes code. |
| **Planner** | `claude-sonnet-4-6` | cyan | Breaks tasks into structured, actionable plans. Never writes code. |
| **Coder** | `claude-opus-4-7` | green | The ONLY agent that writes or modifies code. |
| **Verifier** | `claude-sonnet-4-6` | orange | Runs tests, lint, type checks, and runtime validation. Never writes code. |
| **Reviewer** | `claude-haiku-4-5-20251001` | blue | Conceptual and architectural review. Never writes code. |

---

## Workflow Rules

1. **All sub-agent calls go through the Orchestrator.** No agent calls another agent directly.
2. **Only the Coder writes code.** All other agents output text only.
3. **Canonical flow for every feature:**
   ```
   User Request
       → Orchestrator (decomposes task)
           → Planner (structured plan with acceptance criteria)
           → Coder (implements plan)
           → Verifier (tests, lint, type checks, runtime validation)
           → Reviewer (correctness, ML hygiene, finance rules)
           → Coder (addresses verifier + reviewer feedback, if any)
       → Orchestrator (returns final result to user)
   ```
4. **Revision loops:** If Verifier or Reviewer flags critical issues, Orchestrator routes feedback back to Coder. Max 2 revision loops before escalating to user.
5. **Orchestrator never produces code blocks** — only coordination text and agent handoffs.
6. **Planner output format:** Numbered steps with explicit acceptance criteria, dependency list, files to create/modify, and ML & finance checklist.
7. **Verifier output format:** Commands executed, structured `[PASS/PASS WITH WARNINGS/FAIL]`, critical issues with exact file:line, runtime notes.
8. **Reviewer output format:** Structured `[PASS/PASS WITH WARNINGS/FAIL]`, critical issues, warnings, suggestions, and a one-sentence recommendation.

---

## Project Structure

```
financial-ml-system/
├── CLAUDE.md                  # This file — workflow and agent definitions
├── agents/
│   ├── orchestrator.md        # Orchestrator agent definition
│   ├── planner.md             # Planner agent definition
│   ├── coder.md               # Coder agent definition
│   ├── verifier.md            # Verifier agent definition
│   └── reviewer.md            # Reviewer agent definition
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
- **Agents:** Claude Code subagent system (`agents/*.md`)

---

## Model Notes

**Orchestrator** and **Coder** use `claude-opus-4-7` — the most capable model — because coordination decisions and code generation are the highest-stakes tasks.

**Planner** and **Verifier** both use `claude-sonnet-4-6`: structured reasoning and test execution require solid capability but not the full power of Opus.

**Reviewer** uses `claude-haiku-4-5-20251001` for fast, cost-efficient conceptual review passes.

---

## Environment Variables

```
ANTHROPIC_API_KEY=        # Required for all agents
ALPHA_VANTAGE_API_KEY=    # Market data (optional, yfinance is fallback)
NEWS_API_KEY=             # Sentiment data
MLFLOW_TRACKING_URI=      # Default: ./mlruns
```
