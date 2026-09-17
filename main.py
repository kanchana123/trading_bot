from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import hmac
import logging
import os

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from strategies.bollingerBandStrategy import BollingerBandStrategy
from strategies.action_price.KernelTrader import KernelBacktestAdapter
from db.strategy_manager import StrategyManager
from db.portfolio_manager import PortfolioManager
from db.orders_manager import OrdersManager
from db.deployment_manager import DeploymentManager
from db.create_tables import create_database
from base_models.backtest import Backtest
from base_models.angel_api import AngelAPI
from realtime.realtime_trader import RealtimeTrader, STRATEGY_CLASS_MAP as RT_STRATEGY_CLASS_MAP
from realtime.trade_executor import TradeExecutor

BACKTEST_STRATEGY_CLASSES = {
    "BollingerBand": BollingerBandStrategy,
    "KernelMomentum": KernelBacktestAdapter,
}


def build_backtest_strategy(class_name: str, params: Optional[Dict] = None):
    strategy_cls = BACKTEST_STRATEGY_CLASSES.get(class_name)
    if strategy_cls is None:
        raise KeyError(class_name)
    return strategy_cls(params=params or {})


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    expected = os.getenv("TRADING_BOT_API_KEY")
    if not expected:
        logger.warning("TRADING_BOT_API_KEY is not set; mutating endpoints are unprotected.")
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


strategy_manager = StrategyManager()
portfolio_manager = PortfolioManager()
orders_manager = OrdersManager()
deployment_manager = DeploymentManager()

angel_api_http_client = AngelAPI(connect=True)
trade_executor = TradeExecutor(
    orders_manager=orders_manager, angel_api_client=angel_api_http_client
)
realtime_trader_instance: Optional[RealtimeTrader] = RealtimeTrader(
    deployment_manager=deployment_manager,
    strategy_manager=strategy_manager,
    orders_manager=orders_manager,
    trade_executor=trade_executor,
    angel_api_http_client=angel_api_http_client,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_database()
    yield
    if realtime_trader_instance and realtime_trader_instance._is_running:
        realtime_trader_instance.stop_trading()


app = FastAPI(title="TradingBot API", lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:9000",
    "http://127.0.0.1:9000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StrategyCreate(BaseModel):
    name: Optional[str] = None
    strategy_name: Optional[str] = None
    strategy_class: Optional[str] = None
    params: Optional[Dict] = None

    def class_key(self) -> str:
        return self.strategy_class or self.strategy_name or ""

    def instance_name(self) -> str:
        return self.name or self.class_key()


class BacktestCreate(BaseModel):
    strategy_id: int
    stock: str
    start_date: str
    end_date: str
    portfolio_value: float
    portfolio_name: str
    interval: Optional[str] = "ONE_DAY"


class TokenSubscription(BaseModel):
    instrumentToken: str
    symbol: str
    exchange: str


class DeploymentCreate(BaseModel):
    portfolio_id: int
    strategy_id: int
    token_subscriptions: List[TokenSubscription]
    trading_mode: str


class DeploymentUpdateTokens(BaseModel):
    token_subscriptions: List[TokenSubscription]


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "angel_session": bool(
            angel_api_http_client and angel_api_http_client.is_session_active()
        ),
        "realtime_running": bool(
            realtime_trader_instance and realtime_trader_instance._is_running
        ),
        "api_key_required": bool(os.getenv("TRADING_BOT_API_KEY")),
    }


@app.get("/api/strategies/objects")
async def get_strategy_objects():
    return list(BACKTEST_STRATEGY_CLASSES.keys())


@app.post("/api/strategies/create", dependencies=[Depends(require_api_key)])
async def create_strategy_db(strategy_data: StrategyCreate):
    class_key = strategy_data.class_key()
    if class_key not in BACKTEST_STRATEGY_CLASSES:
        raise HTTPException(status_code=400, detail="Invalid strategy name")
    try:
        strategy_object = build_backtest_strategy(class_key, strategy_data.params)
        strategy_id = strategy_manager.create_strategy(
            name=strategy_data.instance_name(),
            desc=strategy_object.generate_desc()
            if hasattr(strategy_object, "generate_desc")
            else getattr(strategy_object, "desc", ""),
            strategy_class_name=class_key,
            params=strategy_data.params or {},
        )
        return {"strategy_id": strategy_id}
    except Exception as e:
        logger.error("Failed to create strategy: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies/db")
async def get_strategies_from_db():
    return strategy_manager.get_all_strategies()


@app.get("/api/portfolios")
async def get_portfolios_by_strategy(strategy_id: int = None):
    portfolios = portfolio_manager.get_all_portfolios()
    payload = [
        {
            "id": p[0],
            "name": p[1],
            "strategy_id": p[2],
            "starting_value": p[3],
            "end_value": p[4],
            "stock": p[5],
        }
        for p in portfolios
        if strategy_id is None or p[2] == strategy_id
    ]
    return payload


@app.get("/api/orders")
async def get_orders_by_portfolio(portfolio_id: int = None):
    orders = orders_manager.get_all_orders()
    payload = [
        {
            "id": o[0],
            "portfolio_id": o[1],
            "order_type": o[2],
            "transaction_type": o[3],
            "price": o[4],
            "quantity": o[5],
            "timestamp": o[6],
            "metadata": o[7],
            "portfolio_value": o[8],
        }
        for o in orders
        if portfolio_id is None or o[1] == portfolio_id
    ]
    return payload


@app.post("/api/backtest", dependencies=[Depends(require_api_key)])
async def backtest_strategy(backtest_data: BacktestCreate):
    strategy_from_db = strategy_manager.get_strategy(backtest_data.strategy_id)
    if not strategy_from_db:
        raise HTTPException(status_code=404, detail="Strategy not found")

    class_key = strategy_from_db.get("strategy_class_name") or strategy_from_db.get("name")
    if class_key not in BACKTEST_STRATEGY_CLASSES:
        raise HTTPException(status_code=404, detail="Strategy object not found")

    strategy_obj = build_backtest_strategy(class_key, strategy_from_db.get("params") or {})
    portfolio_id = portfolio_manager.create_portfolio(
        name=backtest_data.portfolio_name,
        strategy_id=backtest_data.strategy_id,
        starting_value=backtest_data.portfolio_value,
        stock=backtest_data.stock,
    )

    backtest = Backtest(
        strategy_obj,
        backtest_data.stock,
        backtest_data.start_date,
        backtest_data.end_date,
        initial_portfolio_value=backtest_data.portfolio_value,
        interval=backtest_data.interval or "ONE_DAY",
        angel_api=angel_api_http_client,
    )
    orders = backtest.run()
    logger.info("Backtest produced %s orders", len(orders))

    for order in orders:
        orders_manager.create_order(
            portfolio_id,
            order.order_type,
            order.transaction_type,
            order.price,
            order.quantity,
            order.metadata,
            order.portfolio_value,
            order.timestamp,
        )

    final_value = (
        backtest.portfolio_value_history[-1]
        if backtest.portfolio_value_history
        else backtest_data.portfolio_value
    )
    portfolio_manager.update_portfolio(portfolio_id, end_value=final_value)
    portfolio = portfolio_manager.get_portfolio(portfolio_id)
    return {
        "portfolio_id": portfolio_id,
        "portfolio": {
            "id": portfolio[0],
            "name": portfolio[1],
            "strategy_id": portfolio[2],
            "starting_value": portfolio[3],
            "end_value": portfolio[4],
            "stock": portfolio[5],
        },
        "orders_count": len(orders),
        "final_value": final_value,
    }


@app.post("/api/deployments", status_code=201, dependencies=[Depends(require_api_key)])
async def create_new_deployment(deployment_data: DeploymentCreate):
    if not portfolio_manager.get_portfolio(deployment_data.portfolio_id):
        raise HTTPException(
            status_code=404,
            detail=f"Portfolio with ID {deployment_data.portfolio_id} not found.",
        )
    db_strategy = strategy_manager.get_strategy(deployment_data.strategy_id)
    if not db_strategy:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy with ID {deployment_data.strategy_id} not found.",
        )

    strategy_name_from_db = db_strategy.get("strategy_class_name") or db_strategy.get("name")
    if strategy_name_from_db not in RT_STRATEGY_CLASS_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Strategy '{strategy_name_from_db}' is not configured for real-time trading.",
        )

    if deployment_data.trading_mode not in ("virtual", "real"):
        raise HTTPException(
            status_code=400, detail="Invalid trading_mode. Must be 'virtual' or 'real'."
        )
    if deployment_data.trading_mode == "real":
        if not os.getenv("TRADING_BOT_API_KEY"):
            raise HTTPException(
                status_code=503,
                detail="Real trading requires TRADING_BOT_API_KEY to be set.",
            )
        if not angel_api_http_client or not angel_api_http_client.is_session_active():
            raise HTTPException(
                status_code=503,
                detail="Real trading mode selected, but Angel API client is not available.",
            )

    token_subs_dict_list = [model_to_dict(ts) for ts in deployment_data.token_subscriptions]
    deployment_id = deployment_manager.create_deployment(
        portfolio_id=deployment_data.portfolio_id,
        strategy_id=deployment_data.strategy_id,
        token_subscriptions=token_subs_dict_list,
        trading_mode=deployment_data.trading_mode,
        is_active=False,
    )
    if deployment_id:
        return {
            "deployment_id": deployment_id,
            "message": "Deployment created successfully. Activate to start trading.",
        }
    raise HTTPException(status_code=500, detail="Failed to create deployment in database.")


@app.get("/api/deployments")
async def list_all_deployments():
    return deployment_manager.get_all_deployments()


@app.get("/api/deployments/{deployment_id}")
async def get_specific_deployment(deployment_id: int):
    deployment = deployment_manager.get_deployment_by_id(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment


@app.put("/api/deployments/{deployment_id}/activate", dependencies=[Depends(require_api_key)])
async def activate_deployment_endpoint(deployment_id: int):
    if not deployment_manager.update_deployment_status(deployment_id, is_active=True):
        raise HTTPException(status_code=404, detail="Deployment not found or failed to activate.")
    if realtime_trader_instance and realtime_trader_instance._is_running:
        realtime_trader_instance.reload_deployments()
        return {
            "message": f"Deployment {deployment_id} activated. RealtimeTrader reloading deployments."
        }
    return {
        "message": f"Deployment {deployment_id} activated. Start RealtimeTrader if not running."
    }


@app.put(
    "/api/deployments/{deployment_id}/deactivate",
    dependencies=[Depends(require_api_key)],
)
async def deactivate_deployment_endpoint(deployment_id: int):
    if not deployment_manager.update_deployment_status(deployment_id, is_active=False):
        raise HTTPException(
            status_code=404, detail="Deployment not found or failed to deactivate."
        )
    if realtime_trader_instance and realtime_trader_instance._is_running:
        realtime_trader_instance.reload_deployments()
        return {
            "message": f"Deployment {deployment_id} deactivated. RealtimeTrader reloading deployments."
        }
    return {"message": f"Deployment {deployment_id} deactivated."}


@app.put("/api/deployments/{deployment_id}/tokens", dependencies=[Depends(require_api_key)])
async def update_deployment_tokens_endpoint(
    deployment_id: int, token_data: DeploymentUpdateTokens
):
    token_subs_dict_list = [model_to_dict(ts) for ts in token_data.token_subscriptions]
    if not deployment_manager.update_deployment_tokens(deployment_id, token_subs_dict_list):
        raise HTTPException(
            status_code=404, detail="Deployment not found or failed to update tokens."
        )
    if realtime_trader_instance and realtime_trader_instance._is_running:
        realtime_trader_instance.reload_deployments()
        return {
            "message": f"Deployment {deployment_id} tokens updated. RealtimeTrader reloading deployments."
        }
    return {"message": f"Deployment {deployment_id} tokens updated."}


@app.post("/api/realtime/start", dependencies=[Depends(require_api_key)])
async def start_realtime_trader():
    if not realtime_trader_instance:
        raise HTTPException(status_code=503, detail="RealtimeTrader not initialized.")
    if not angel_api_http_client or not angel_api_http_client.is_session_active():
        raise HTTPException(
            status_code=503,
            detail="Angel API session is not active. Check broker credentials.",
        )
    if realtime_trader_instance._is_running:
        return {"message": "RealtimeTrader is already running."}
    realtime_trader_instance.start_trading()
    return {"message": "RealtimeTrader start initiated in a background thread."}


@app.post("/api/realtime/stop", dependencies=[Depends(require_api_key)])
async def stop_realtime_trader():
    if not realtime_trader_instance or not realtime_trader_instance._is_running:
        return {"message": "RealtimeTrader is not running or not initialized."}
    realtime_trader_instance.stop_trading()
    return {"message": "RealtimeTrader stop initiated."}


DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
DASHBOARD_INDEX = os.path.join(DASHBOARD_DIR, "index.html")
if os.path.isfile(DASHBOARD_INDEX):
    @app.get("/")
    async def dashboard_index():
        return FileResponse(DASHBOARD_INDEX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
