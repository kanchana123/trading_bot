import pytest
from fastapi.testclient import TestClient

from db.create_tables import create_database
from db.deployment_manager import DeploymentManager
from db.orders_manager import OrdersManager
from db.portfolio_manager import PortfolioManager
from db.strategy_manager import StrategyManager
from realtime.trade_executor import TradeExecutor


API_KEY = "test-secret"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_BOT_API_KEY", API_KEY)
    for key in ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PASSWORD", "ANGEL_TOTP"):
        monkeypatch.delenv(key, raising=False)

    db = str(tmp_path / "api.db")
    create_database(db)

    import main

    main.strategy_manager = StrategyManager(db)
    main.portfolio_manager = PortfolioManager(db)
    main.orders_manager = OrdersManager(db)
    main.deployment_manager = DeploymentManager(db)
    main.trade_executor = TradeExecutor(main.orders_manager, main.angel_api_http_client)

    with TestClient(main.app) as test_client:
        yield test_client


def _auth():
    return {"X-API-Key": API_KEY}


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["api_key_required"] is True
    assert body["angel_session"] is False


def test_strategy_objects(client):
    res = client.get("/api/strategies/objects")
    assert res.status_code == 200
    assert "BollingerBand" in res.json()
    assert "KernelMomentum" in res.json()


def test_create_strategy_requires_api_key(client):
    res = client.post(
        "/api/strategies/create",
        json={"name": "bb", "strategy_class": "BollingerBand", "params": {"window": 10}},
    )
    assert res.status_code == 401


def test_create_strategy_rejects_unknown_class(client):
    res = client.post(
        "/api/strategies/create",
        json={"name": "x", "strategy_class": "NotAStrategy"},
        headers=_auth(),
    )
    assert res.status_code == 400


def test_create_strategy_and_list(client):
    created = client.post(
        "/api/strategies/create",
        json={
            "name": "my-bb",
            "strategy_class": "BollingerBand",
            "params": {"window": 10, "std_multiplier": 2},
        },
        headers=_auth(),
    )
    assert created.status_code == 200
    strategy_id = created.json()["strategy_id"]
    listed = client.get("/api/strategies/db")
    assert listed.status_code == 200
    row = next(item for item in listed.json() if item["id"] == strategy_id)
    assert row["name"] == "my-bb"
    assert row["strategy_class_name"] == "BollingerBand"
    assert row["params"]["window"] == 10


def test_backtest_persists_portfolio_end_value(client, monkeypatch):
    import main
    from base_models.orders import Order

    class _FakeBacktest:
        def __init__(self, *args, **kwargs):
            self.portfolio_value_history = [100000.0, 101250.0]
            self.orders = [
                Order("backtest", "buy", 100.0, 1, "buy", 100000.0, None),
                Order("backtest", "sell", 110.0, 1, "sell", 101250.0, None),
            ]

        def run(self):
            return self.orders

    monkeypatch.setattr(main, "Backtest", _FakeBacktest)
    created = client.post(
        "/api/strategies/create",
        json={"name": "km", "strategy_class": "KernelMomentum", "params": {}},
        headers=_auth(),
    )
    strategy_id = created.json()["strategy_id"]
    res = client.post(
        "/api/backtest",
        json={
            "strategy_id": strategy_id,
            "stock": "ICICIBANK-EQ",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "portfolio_value": 100000,
            "portfolio_name": "bt-1",
            "interval": "ONE_DAY",
        },
        headers=_auth(),
    )
    assert res.status_code == 200
    body = res.json()
    assert body["orders_count"] == 2
    assert body["final_value"] == 101250.0
    assert body["portfolio"]["end_value"] == 101250.0
    orders = client.get("/api/orders", params={"portfolio_id": body["portfolio_id"]})
    assert len(orders.json()) == 2


def test_backtest_unknown_strategy(client):
    res = client.post(
        "/api/backtest",
        json={
            "strategy_id": 999,
            "stock": "ICICIBANK-EQ",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "portfolio_value": 100000,
            "portfolio_name": "missing",
        },
        headers=_auth(),
    )
    assert res.status_code == 404


def test_deployment_and_realtime_guards(client, monkeypatch):
    import main

    class _FakeBacktest:
        def __init__(self, *args, **kwargs):
            self.portfolio_value_history = [100000.0]
            self.orders = []

        def run(self):
            return []

    monkeypatch.setattr(main, "Backtest", _FakeBacktest)

    created = client.post(
        "/api/strategies/create",
        json={"name": "live-km", "strategy_class": "KernelMomentum", "params": {}},
        headers=_auth(),
    )
    strategy_id = created.json()["strategy_id"]
    bt = client.post(
        "/api/backtest",
        json={
            "strategy_id": strategy_id,
            "stock": "RELIANCE-EQ",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "portfolio_value": 100000,
            "portfolio_name": "rt-port",
        },
        headers=_auth(),
    )
    portfolio_id = bt.json()["portfolio_id"]

    bollie = client.post(
        "/api/strategies/create",
        json={"name": "bb-live", "strategy_class": "BollingerBand", "params": {}},
        headers=_auth(),
    )
    bb_id = bollie.json()["strategy_id"]
    rejected = client.post(
        "/api/deployments",
        json={
            "portfolio_id": portfolio_id,
            "strategy_id": bb_id,
            "token_subscriptions": [
                {"instrumentToken": "2885", "symbol": "RELIANCE-EQ", "exchange": "NSE_EQ"}
            ],
            "trading_mode": "virtual",
        },
        headers=_auth(),
    )
    assert rejected.status_code == 400

    real = client.post(
        "/api/deployments",
        json={
            "portfolio_id": portfolio_id,
            "strategy_id": strategy_id,
            "token_subscriptions": [
                {"instrumentToken": "2885", "symbol": "RELIANCE-EQ", "exchange": "NSE_EQ"}
            ],
            "trading_mode": "real",
        },
        headers=_auth(),
    )
    assert real.status_code == 503

    virtual = client.post(
        "/api/deployments",
        json={
            "portfolio_id": portfolio_id,
            "strategy_id": strategy_id,
            "token_subscriptions": [
                {"instrumentToken": "2885", "symbol": "RELIANCE-EQ", "exchange": "NSE_EQ"}
            ],
            "trading_mode": "virtual",
        },
        headers=_auth(),
    )
    assert virtual.status_code == 201
    dep_id = virtual.json()["deployment_id"]
    activated = client.put(f"/api/deployments/{dep_id}/activate", headers=_auth())
    assert activated.status_code == 200
    listed = client.get("/api/deployments")
    assert any(row["id"] == dep_id for row in listed.json())

    start = client.post("/api/realtime/start", headers=_auth())
    assert start.status_code == 503
