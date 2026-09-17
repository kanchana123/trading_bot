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
- Optional `X-API-Key` on mutating routes; real trading requires that key plus an active Angel session

## System design

High-level components and how they talk to Angel One and SQLite:

```mermaid
flowchart TB
  subgraph clients [Clients]
    Dash[Dashboard /]
    HttpClient[HTTP client]
  end

  subgraph api [FastAPI - main.py]
    Routes[REST /api/*]
    Auth[X-API-Key guard]
  end

  subgraph domain [Domain]
    SM[StrategyManager]
    PM[PortfolioManager]
    OM[OrdersManager]
    DM[DeploymentManager]
    BT[Backtest]
    RT[RealtimeTrader]
    EX[TradeExecutor]
  end

  subgraph strategies [Strategies]
    BB[BollingerBand - backtest]
    KM[KernelMomentum - backtest + RT]
  end

  subgraph persistence [Local store]
    DB[(SQLite trading_bot.db)]
  end

  subgraph broker [Angel One]
    HTTP[SmartConnect REST]
    WS[SmartWebSocket ticks]
  end

  Dash --> Routes
  HttpClient --> Routes
  Routes --> Auth
  Auth --> SM
  Auth --> PM
  Auth --> OM
  Auth --> DM
  Auth --> BT
  Auth --> RT
  SM --> DB
  PM --> DB
  OM --> DB
  DM --> DB
  BT --> BB
  BT --> KM
  BT --> HTTP
  BT --> OM
  RT --> KM
  RT --> WS
  RT --> EX
  EX --> OM
  EX --> HTTP
  HTTP --> broker
  WS --> broker
```

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
| GET | `/api/health` | no | Angel session + realtime running + whether API key is required |
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

## LLM equity script

`strategies/low_risk_llm_equity/main.py` is a separate backtest: Angel candles → kernel signal → optional LLM-generated Python → news sentiment → orders CSV + HTML chart. Generated code is AST-checked and executed in a restricted sandbox (`os`, `eval`, and similar calls are rejected). It is not served by FastAPI.
