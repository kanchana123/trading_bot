import json
import sqlite3

import pytest

from db.create_tables import create_database
from db.deployment_manager import DeploymentManager
from db.orders_manager import OrdersManager
from db.portfolio_manager import PortfolioManager
from db.strategy_manager import StrategyManager
from realtime.trade_executor import TradeExecutor


def _seeded_portfolio(tmp_path):
    db = str(tmp_path / "test.db")
    create_database(db)
    sm = StrategyManager(db)
    pm = PortfolioManager(db)
    sid = sm.create_strategy("s1", "d", "KernelMomentum", {"threshold": 1.5})
    pid = pm.create_portfolio("p1", sid, 10000, stock="RELIANCE-EQ")
    return db, sid, pid


def test_strategy_params_round_trip(tmp_path):
    db = str(tmp_path / "test.db")
    create_database(db)
    mgr = StrategyManager(db)
    sid = mgr.create_strategy(
        name="bb-1",
        desc="test",
        strategy_class_name="BollingerBand",
        params={"window": 10},
    )
    row = mgr.get_strategy(sid)
    assert row["strategy_class_name"] == "BollingerBand"
    assert row["params"]["window"] == 10
    assert mgr.get_all_strategies()[0]["name"] == "bb-1"


def test_duplicate_strategy_name_is_rejected(tmp_path):
    db = str(tmp_path / "test.db")
    create_database(db)
    mgr = StrategyManager(db)
    mgr.create_strategy("dup", "a", "BollingerBand", {})
    with pytest.raises(sqlite3.IntegrityError):
        mgr.create_strategy("dup", "b", "KernelMomentum", {})


def test_missing_strategy_returns_none(tmp_path):
    db = str(tmp_path / "test.db")
    create_database(db)
    assert StrategyManager(db).get_strategy(999) is None


def test_foreign_keys_block_orphan_orders(tmp_path):
    db = str(tmp_path / "test.db")
    create_database(db)
    om = OrdersManager(db)
    with pytest.raises(sqlite3.IntegrityError):
        om.create_order(
            portfolio_id=999,
            order_type="virtual",
            transaction_type="buy",
            price=10,
            quantity=1,
        )


def test_orders_manager_rejects_invalid_types(tmp_path):
    db, _, pid = _seeded_portfolio(tmp_path)
    om = OrdersManager(db)
    with pytest.raises(ValueError, match="order_type"):
        om.create_order(pid, "paper", "buy", 10, 1)
    with pytest.raises(ValueError, match="transaction_type"):
        om.create_order(pid, "virtual", "hold", 10, 1)


def test_virtual_order_is_persisted(tmp_path):
    db, sid, pid = _seeded_portfolio(tmp_path)
    om = OrdersManager(db)
    executor = TradeExecutor(om, angel_api_client=None)
    result = executor.execute_order(
        portfolio_id=pid,
        strategy_id=sid,
        token_details={"symbol": "RELIANCE-EQ", "instrumentToken": "2885", "exchange": "NSE_EQ"},
        transaction_type="buy",
        quantity=1,
        price=100.5,
        trading_mode="virtual",
    )
    assert result["success"] is True
    saved = om.get_order(result["order_id"])
    assert saved[3] == "buy"
    assert json.loads(saved[7])["symbol"] == "RELIANCE-EQ"


def test_executor_rejects_bad_quantity_and_side():
    executor = TradeExecutor(orders_manager=None)
    assert executor.execute_order(1, 1, {}, "hold", 1, 10)["success"] is False
    assert executor.execute_order(1, 1, {}, "buy", 0, 10)["success"] is False
    assert executor.execute_order(1, 1, {}, "buy", 1, 10, trading_mode="live")["success"] is False


def test_real_order_requires_active_session(tmp_path):
    db, sid, pid = _seeded_portfolio(tmp_path)
    om = OrdersManager(db)
    executor = TradeExecutor(om, angel_api_client=None)
    result = executor.execute_order(
        portfolio_id=pid,
        strategy_id=sid,
        token_details={"symbol": "RELIANCE-EQ", "instrumentToken": "2885", "exchange": "NSE_EQ"},
        transaction_type="buy",
        quantity=1,
        price=100.5,
        trading_mode="real",
    )
    assert result["success"] is False
    assert om.get_all_orders() == []


class _FakeAngel:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def is_session_active(self):
        return True

    def place_order(self, **kwargs):
        self.calls.append(kwargs)
        if self.ok:
            return {"success": True, "broker_order_id": "BRK-1"}
        return {"success": False, "error": "rejected"}


def test_real_order_persists_broker_id_on_success(tmp_path):
    db, sid, pid = _seeded_portfolio(tmp_path)
    om = OrdersManager(db)
    angel = _FakeAngel(ok=True)
    executor = TradeExecutor(om, angel_api_client=angel)
    result = executor.execute_order(
        portfolio_id=pid,
        strategy_id=sid,
        token_details={"symbol": "RELIANCE-EQ", "instrumentToken": "2885", "exchange": "NSE_EQ"},
        transaction_type="BUY",
        quantity=2,
        price=101,
        trading_mode="real",
    )
    assert result["success"] is True
    assert result["broker_order_id"] == "BRK-1"
    assert angel.calls[0]["quantity"] == 2
    meta = json.loads(om.get_order(result["order_id"])[7])
    assert meta["broker_order_id"] == "BRK-1"


def test_rejected_real_order_is_not_saved(tmp_path):
    db, sid, pid = _seeded_portfolio(tmp_path)
    om = OrdersManager(db)
    executor = TradeExecutor(om, angel_api_client=_FakeAngel(ok=False))
    result = executor.execute_order(
        portfolio_id=pid,
        strategy_id=sid,
        token_details={"symbol": "RELIANCE-EQ", "instrumentToken": "2885", "exchange": "NSE_EQ"},
        transaction_type="buy",
        quantity=1,
        price=100,
        trading_mode="real",
    )
    assert result["success"] is False
    assert om.get_all_orders() == []


def test_deployment_lifecycle(tmp_path):
    db, sid, pid = _seeded_portfolio(tmp_path)
    dm = DeploymentManager(db)
    tokens = [{"instrumentToken": "2885", "symbol": "RELIANCE-EQ", "exchange": "NSE_EQ"}]
    dep_id = dm.create_deployment(pid, sid, tokens, "virtual", is_active=False)
    assert dep_id
    row = dm.get_deployment_by_id(dep_id)
    assert row["trading_mode"] == "virtual"
    assert dm.get_active_deployments() == []
    assert dm.update_deployment_status(dep_id, True)
    assert dm.get_active_deployments()[0]["id"] == dep_id
    assert dm.update_deployment_tokens(dep_id, tokens)
    assert dm.delete_deployment(dep_id)
    assert dm.get_deployment_by_id(dep_id) is None


def test_portfolio_end_value_update(tmp_path):
    db, sid, pid = _seeded_portfolio(tmp_path)
    pm = PortfolioManager(db)
    assert pm.update_portfolio(pid, end_value=12345.0)
    row = pm.get_portfolio(pid)
    assert row[4] == 12345.0
