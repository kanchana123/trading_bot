# TradingBot

Python trading platform for Indian equities through **Angel One SmartAPI**. You design a strategy, backtest it on historical candles, paper-trade the same logic on live ticks, then (optionally) send real orders.

The running product is a FastAPI server, a SQLite store, and a static dashboard. Broker login is optional at startup: the API comes up without credentials, but historical download, websocket ticks, and live orders need a valid Angel session.

## What is wired vs experimental

| Path | Role |
| --- | --- |
| `BollingerBand` | Registered for **backtest only** |
| `KernelMomentum` | Registered for **backtest and realtime** (virtual or real) |
| `strategies/low_risk_llm_equity/` | Standalone LLM backtest script (sandbox + news + kernel). Not exposed on the API |
| `dqn_trader`, `legacy_dqn_trader`, `RL_google_colab`, `RL_RNN_Intraday`, `LLM_Risk_Management`, `options_combinations`, `GPT_RL.py` | Research / notebook experiments. Not imported by `main.py` |

## Features

- Named strategy instances with JSON params stored in SQLite
- Historical backtest via Angel candle API; orders and `end_value` persisted on a portfolio
- Virtual (paper) and real deployments on websocket ticks
- Static dashboard at `/` to create strategies, run backtests, and inspect orders
- React investment desk at `/ui/` (Vite `web/` on port 3000 in dev) with Desk and **How it works** menus
- Optional `X-API-Key` on mutating routes; real trading requires that key plus an active Angel session

## Architecture

Clients talk only to FastAPI. The v2 graph can retrieve and propose; `TradeExecutor` is the only path that writes a fill. Models never call Angel.

```mermaid
flowchart TB
  subgraph clients [Clients]
    Desk["React investment desk /ui/"]
    Dash["Static dashboard /"]
    Cli[HTTP client]
  end

  subgraph fastapi [FastAPI main.py]
    Auth[X-API-Key]
    V1["v1 REST<br/>strategies backtest deploy realtime"]
    V2["v2 REST + SSE<br/>runs pending approve reject"]
  end

  subgraph v2stack [V2 investment graph]
    direction TB
    Research[research]
    Quant[quant]
    Risk[risk]
    Adj[adjudicator]
    PolicyN[policy]
    HITL[HITL interrupt]
    ExecN[execute_order]
    Research --> Quant --> Risk --> Adj --> PolicyN --> HITL
    HITL -->|approve| ExecN
    HITL -->|reject| Stop[run ended]
  end

  subgraph v2support [V2 layers]
    RAG["RAG corpus BM25 + dense reranker"]
    GOV["PolicyEngine + guardrails"]
    Trace[Tracing]
    Audit[AuditLedger]
  end

  subgraph classic [Classic trading]
    SM[StrategyManager]
    BT[Backtest]
    RT[RealtimeTrader]
    ST["BollingerBand / KernelMomentum"]
  end

  subgraph shared [Shared execution and store]
    EX[TradeExecutor]
    DB[(SQLite)]
  end

  subgraph angel [Angel One]
    REST[SmartConnect REST]
    WS[SmartWebSocket]
  end

  Desk --> V2
  Dash --> V1
  Dash --> V2
  Cli --> Auth
  Auth --> V1
  Auth --> V2
  V2 --> Research
  Research --> RAG
  Quant --> ST
  PolicyN --> GOV
  Research --> Trace
  ExecN --> Audit
  ExecN --> EX
  V1 --> SM
  V1 --> BT
  V1 --> RT
  BT --> ST
  RT --> ST
  RT --> EX
  SM --> DB
  BT --> DB
  RT --> DB
  EX --> DB
  Audit --> DB
  BT --> REST
  EX --> REST
  RT --> WS
```

`execute_order` and the live trader share `TradeExecutor`. Virtual fills stay in SQLite; real fills call Angel `placeOrder` immediately.

## System design

Classic backtest and realtime paths in more detail:

### Backtest flow

```mermaid
sequenceDiagram
  participant U as Dashboard
  participant API as FastAPI
  participant S as Strategy instance
  participant BT as Backtest
  participant A as AngelAPI
  participant DB as SQLite

  U->>API: POST /api/strategies/create
  API->>DB: insert strategy name, class, params
  U->>API: POST /api/backtest
  API->>DB: insert portfolio
  API->>S: build_backtest_strategy(class, params)
  API->>BT: run(stock, dates, interval)
  alt no DataFrame passed
    BT->>A: download_historical_data
    A-->>BT: OHLCV bars
  end
  loop each bar after warmup
    BT->>S: should_buy / should_sell / select_quantity
    S-->>BT: signal
  end
  BT-->>API: orders + final value
  API->>DB: insert orders, update portfolio.end_value
  API-->>U: portfolio_id, orders_count, final_value
```

### Realtime flow

```mermaid
sequenceDiagram
  participant U as Dashboard
  participant API as FastAPI
  participant RT as RealtimeTrader thread
  participant WS as AngelWebSocketClient
  participant S as KernelStrategy
  participant EX as TradeExecutor
  participant DB as SQLite
  participant B as Angel One

  U->>API: POST /api/deployments trading_mode=virtual|real
  API->>DB: insert deployment is_active=false
  U->>API: PUT /api/deployments/{id}/activate
  U->>API: POST /api/realtime/start
  API->>RT: start_trading on background thread
  RT->>B: login, jwt + feed token
  RT->>WS: connect
  WS->>B: subscribe instrument tokens
  B-->>WS: tick LTP
  WS->>RT: _handle_tick_data
  RT->>S: on_new_tick
  S-->>RT: buy/sell or none
  alt signal
    RT->>EX: execute_order
    alt trading_mode=real
      EX->>B: placeOrder
      B-->>EX: broker order id
    end
    EX->>DB: insert order virtual|real
    EX-->>RT: success
    RT->>S: confirm_fill
  end
```

### Data model

```mermaid
erDiagram
  strategies ||--o{ portfolios : "strategy_id"
  strategies ||--o{ deployments : "strategy_id"
  portfolios ||--o{ orders : "portfolio_id"
  portfolios ||--o{ deployments : "portfolio_id"

  strategies {
    int id PK
    text name UK
    text desc
    text strategy_class_name
    text params
  }
  portfolios {
    int id PK
    text name UK
    int strategy_id FK
    real starting_value
    real end_value
    text stock
  }
  orders {
    int id PK
    int portfolio_id FK
    text order_type
    text transaction_type
    real price
    int quantity
    datetime timestamp
    text metadata
    real portfolio_value
  }
  deployments {
    int id PK
    int portfolio_id FK
    int strategy_id FK
    text token_subscriptions
    text trading_mode
    bool is_active
  }
```

`order_type` is `backtest`, `virtual`, or `real`. `trading_mode` on a deployment is `virtual` or `real`.

## Repository layout

```
main.py                 FastAPI app, CORS, API key, uvicorn entrypoint
dashboard/index.html    Static UI served at GET /
db/                     SQLite schema + managers (foreign keys on)
base_models/            Strategy, Backtest, Order, AngelAPI, OHLC helpers
realtime/               RealtimeTrader, websocket client, TradeExecutor
strategies/             Strategy implementations (API + experiments)
utils/logger_setup.py   File + console logging helper
tests/                  DB, executor, sandbox, backtest tests
```

## Getting started

1. Clone the repo and create a virtualenv.
2. Install dependencies:

```sh
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in:

| Variable | Used for |
| --- | --- |
| `ANGEL_API_KEY` | SmartAPI key |
| `ANGEL_CLIENT_CODE` | Client code |
| `ANGEL_PASSWORD` | Login PIN/password |
| `ANGEL_TOTP` | TOTP secret |
| `OPENAI_API_KEY` | LLM equity script only |
| `TRADING_BOT_API_KEY` | Protects create / backtest / deploy / realtime routes |

Real trading is rejected if `TRADING_BOT_API_KEY` is unset or Angel is offline.

4. Tables are created on API startup. To create them alone:

```sh
python db/create_tables.py
```

5. Run the API:

```sh
python main.py
```

Server: `http://127.0.0.1:9000`. Open that URL for the dashboard. When `TRADING_BOT_API_KEY` is set, send it as `X-API-Key`.

6. Tests:

```sh
pytest
```

## API

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| GET | `/` | no | Dashboard |
| GET | `/api/health` | no | Angel session, realtime, API key flag, `v2`, `llm_can_execute=false` |
| GET | `/api/v2/meta` | no | Graph edges, LangGraph availability, interrupt_before execute_order |
| POST | `/api/v2/runs` | yes | Start research→quant→risk→adjudicator→policy; always pauses before execute |
| GET | `/api/v2/runs` | no | Audit ledger summaries |
| GET | `/api/v2/runs/pending` | no | HITL approval queue |
| GET | `/api/v2/runs/{id}` | no | Full state, proposal, policy, traces |
| GET | `/api/v2/runs/{id}/events` | no | SSE of node thoughts |
| POST | `/api/v2/runs/{id}/approve` | yes | `{operator_id}` then TradeExecutor paper/real |
| POST | `/api/v2/runs/{id}/reject` | yes | `{operator_id}` |
| GET | `/api/strategies/objects` | no | `BollingerBand`, `KernelMomentum` |
| POST | `/api/strategies/create` | yes | `{name, strategy_class, params}` |
| GET | `/api/strategies/db` | no | Saved instances |
| GET | `/api/portfolios` | no | Optional `?strategy_id=` |
| GET | `/api/orders` | no | Optional `?portfolio_id=` |
| POST | `/api/backtest` | yes | Creates portfolio, runs backtest, stores orders and `end_value` |
| GET | `/api/deployments` | no | All deployments |
| GET | `/api/deployments/{id}` | no | One deployment |
| POST | `/api/deployments` | yes | `{portfolio_id, strategy_id, token_subscriptions, trading_mode}` |
| PUT | `/api/deployments/{id}/activate` | yes | Reloads subscriptions if the trader is running |
| PUT | `/api/deployments/{id}/deactivate` | yes | Same reload behavior |
| PUT | `/api/deployments/{id}/tokens` | yes | Replace websocket token list |
| POST | `/api/realtime/start` | yes | Websocket loop in a background thread |
| POST | `/api/realtime/stop` | yes | Disconnect and clear instances |

Auth = `X-API-Key` when `TRADING_BOT_API_KEY` is set. If that env var is empty, mutating routes stay open and a warning is logged.

## How a strategy plugs in

**Backtest** (`base_models/strategy.py`): implement `process_data`, `should_buy`, `should_sell`, `select_quantity`, then register the class in `BACKTEST_STRATEGY_CLASSES` in `main.py`.

**Realtime** (`strategies/base_strategy_rt.py`): implement `on_new_tick` (return a trade dict or `None`). Register in `STRATEGY_CLASS_MAP` in `realtime/realtime_trader.py`. Position size should change only in `confirm_fill` after `TradeExecutor` succeeds.

OHLC columns are normalized to `Open` / `High` / `Low` / `Close` / `Volume` before indicators run.

## How it works

Open **How it works** on the React desk (`/ui/` or `http://127.0.0.1:3000`) or on the static dashboard (`/`). Models retrieve and propose; they never call the broker. A deterministic policy engine and a human sit in front of every fill.

```mermaid
flowchart LR
  R[Research] --> Q[Quant]
  Q --> K[Risk]
  K --> A[Adjudicator]
  A --> P[Policy]
  P --> H[HITL pause]
  H -->|approve| E[Execute via TradeExecutor]
  H -->|reject| X[Run ended]
```

1. **Research** retrieves fixture 10-K / 10-Q / news chunks for the symbol (hybrid BM25 + dense search, then a reranker). The summary must be grounded in those excerpts. Prompt-injection style text is screened out.
2. **Quant** runs Kernel momentum and Bollinger band logic on completed bars and emits structured signals, not broker calls.
3. **Risk** scores concentration, historical VaR, drawdown, circuit-breaker returns, and whether research was grounded.
4. **Adjudicator** merges those inputs into one order proposal (side, quantity, limit, rationale, citations) or rejects with a typed reason.
5. **Policy** is rule-based: about 5% of NAV in one name, stop-loss bound, VaR cap, drawdown cap, cash check. A model cannot override this.
6. **HITL** always pauses before `execute_order`. Real mode requires an operator. This build also pauses paper mode so you can inspect the proposal.
7. **Execute** is the only node allowed to call `TradeExecutor`, and only after policy is replayed.

**Approve runs immediately.** It does not wait for the next trading day and does not place an AMO / GTD / overnight delivery order.

- **Virtual (paper)** writes a SQLite row with `order_type=virtual`. Angel is not contacted.
- **Real** calls Angel `placeOrder` now as LIMIT / INTRADAY if the session is live. A closed market or bad token is a broker reject, not a next-session queue.

Fixture research exists for `RELIANCE-EQ` and `ICICIBANK-EQ`. Start a run from the desk, inspect citations and policy in the inspector, then approve or reject from the queue. Audit rows replay the graph snapshot.

## V2 multi-agent graph

`agents/graph.py` implements the flow above. Policy lives in `governance/policy_engine.py`. Execution is `interrupt_before=["execute_order"]`. Only `agents/nodes/execution_node.py` may call `TradeExecutor`; it re-runs policy and requires an operator id for real mode.

```mermaid
flowchart LR
  subgraph research [Research]
    Tools[rag/tools]
    Store[vector_store]
    Rank[reranker]
    Corpus[fixture 10-K / 10-Q / news]
    Tools --> Store
    Store --> Rank
    Corpus --> Store
  end

  subgraph quant [Quant]
    Kernel[KernelMomentum]
    BB[BollingerBand]
  end

  subgraph decision [Decision]
    RiskN[risk_agent]
    AdjN[adjudicator]
    PolN[policy_node]
    Engine[PolicyEngine]
    PolN --> Engine
  end

  subgraph exec [Execution]
    HITL2[operator approve / reject]
    Node[execution_node]
    TE[TradeExecutor]
    Node --> TE
  end

  research --> quant --> RiskN --> AdjN --> PolN --> HITL2 --> Node
```

Dashboard `/` is the original static console (Desk + How it works). The React investment desk is `web/` — `npm install && npm run dev` on port 3000 (proxies `/api`), or `npm run build` and open `/ui/` on the FastAPI server.

## LLM equity script

`strategies/low_risk_llm_equity/main.py` is a separate backtest: Angel candles → kernel signal → optional LLM-generated Python → news sentiment → orders CSV + HTML chart. Generated code is AST-checked and executed in a restricted sandbox (`os`, `eval`, and similar calls are rejected). It is not served by FastAPI.
