import pandas as pd
import pytest

from base_models.backtest import Backtest
from base_models.ohlc import normalize_ohlc_columns
from tests.helpers import ScriptedStrategy, ohlc_frame


def test_normalize_renames_lowercase_columns():
    df = pd.DataFrame({"close": [1, 2], "open": [1, 2], "high": [2, 3], "low": [0, 1]})
    out = normalize_ohlc_columns(df)
    assert list(out.columns) == ["Close", "Open", "High", "Low"]


def test_normalize_leaves_none_and_empty():
    assert normalize_ohlc_columns(None) is None
    empty = pd.DataFrame()
    assert normalize_ohlc_columns(empty) is empty


def test_normalize_does_not_overwrite_existing_close():
    df = pd.DataFrame({"Close": [1], "close": [2]})
    out = normalize_ohlc_columns(df)
    assert out["Close"].iloc[0] == 1


def test_backtest_returns_no_orders_when_data_empty():
    strategy = ScriptedStrategy()
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-01-10",
        data=pd.DataFrame({"Close": []}),
        angel_api=_InactiveAngel(),
    )
    assert bt.run() == []


def test_backtest_rejects_inverted_dates():
    strategy = ScriptedStrategy()
    with pytest.raises(ValueError, match="Start date"):
        Backtest(
            strategy,
            stock="TEST-EQ",
            start_date="2024-02-01",
            end_date="2024-01-01",
            data=pd.DataFrame(),
            angel_api=_InactiveAngel(),
        )


def test_backtest_requires_close_column():
    strategy = ScriptedStrategy()
    data = pd.DataFrame({"Price": [1, 2, 3, 4]}, index=pd.date_range("2024-01-01", periods=4))
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-01-04",
        data=data,
    )
    with pytest.raises(ValueError, match="Close"):
        bt.run()


def test_backtest_buy_and_sell_updates_cash_and_orders():
    data = ohlc_frame(n=8, start=100.0)
    strategy = ScriptedStrategy(buy_at=2, sell_at=5, quantity=2)
    initial = 100000.0
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-01-08",
        data=data,
        initial_portfolio_value=initial,
    )
    orders = bt.run()
    assert [o.transaction_type for o in orders] == ["buy", "sell"]
    assert orders[0].quantity == 2
    assert orders[0].order_type == "backtest"
    assert orders[1].quantity == 2
    buy_cost = orders[0].price * 2
    sell_proceeds = orders[1].price * 2
    expected_final = initial - buy_cost + sell_proceeds
    assert bt.portfolio_value_history[-1] == pytest.approx(expected_final)
    assert bt.current_position == 0


def test_backtest_skips_buy_when_cash_is_insufficient():
    data = ohlc_frame(n=6, start=100.0)
    strategy = ScriptedStrategy(buy_at=2, sell_at=5, quantity=10_000)
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-01-06",
        data=data,
        initial_portfolio_value=100.0,
    )
    assert bt.run() == []
    assert bt.current_position == 0


def test_backtest_normalizes_lowercase_ohlc_for_strategy():
    data = ohlc_frame(n=6).rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low"})
    strategy = ScriptedStrategy(buy_at=2, sell_at=4, quantity=1)
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-01-06",
        data=data,
        initial_portfolio_value=100000,
    )
    orders = bt.run()
    assert len(orders) == 2
    assert "Close" in strategy.data.columns


def test_backtest_metadata_omits_momentum_when_missing():
    data = ohlc_frame(n=6)
    strategy = ScriptedStrategy(buy_at=2, sell_at=4, quantity=1)
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-01-06",
        data=data,
        initial_portfolio_value=100000,
    )
    orders = bt.run()
    assert "momentum" not in orders[0].metadata


class _InactiveAngel:
    def is_session_active(self):
        return False
