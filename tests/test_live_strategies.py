from datetime import datetime, timedelta

import pandas as pd
import pytest

from base_models.backtest import Backtest
from strategies.dual_mode import DualModeStrategy
from strategies.indicators import rsi, sma
from strategies.low_risk_combo import LowRiskComboStrategy
from strategies.registry import STRATEGY_CATALOG, STRATEGY_CLASS_MAP, supports_realtime
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.sma_crossover import SMACrossoverStrategy


TOKEN = {"instrumentToken": "11536", "symbol": "TCS-EQ", "exchange": "NSE_EQ"}


def _v_shape(n_down=25, n_up=30, start=120.0):
    closes = [start - i for i in range(n_down)] + [start - n_down + i for i in range(1, n_up + 1)]
    rows = [
        {"Open": c - 0.3, "High": c + 0.5, "Low": c - 0.5, "Close": float(c), "Volume": 1000}
        for c in closes
    ]
    return pd.DataFrame(rows, index=pd.date_range("2024-01-01", periods=len(rows), freq="D"))


def test_registry_exposes_every_strategy_for_all_modes():
    assert set(STRATEGY_CATALOG) == set(STRATEGY_CLASS_MAP)
    for name, spec in STRATEGY_CATALOG.items():
        assert supports_realtime(name)
        assert spec["backtest"] is not None
        assert spec["realtime"] is not None
        backtest = spec["backtest"](params=spec["default_params"])
        live = spec["realtime"](params=spec["default_params"])
        assert hasattr(backtest, "should_buy")
        assert hasattr(live, "on_new_tick")


def test_sma_buy_and_sell_on_crossover():
    strategy = SMACrossoverStrategy(params={"fast": 2, "slow": 3})
    strategy.data = pd.DataFrame(
        {
            "Close": [10.0, 10.0, 10.0, 12.0, 8.0],
            "sma_fast": [10.0, 10.0, 10.0, 11.0, 10.0],
            "sma_slow": [10.0, 10.0, 10.0, 10.5, 10.7],
        }
    )
    assert not strategy.should_buy(2)
    assert strategy.should_buy(3)
    assert not strategy.should_sell(3)
    assert strategy.should_sell(4)


def test_rsi_buy_and_sell_on_threshold_cross():
    strategy = RSIReversionStrategy(params={"oversold": 30, "overbought": 70, "trend_sma": 0})
    strategy.data = pd.DataFrame(
        {
            "Close": [10.0, 11.0, 12.0, 13.0],
            "rsi": [28.0, 31.0, 72.0, 69.0],
        }
    )
    assert strategy.should_buy(1)
    assert not strategy.should_sell(1)
    assert strategy.should_sell(3)
    assert not strategy.should_buy(3)


def test_low_risk_combo_requires_rsi_filter():
    strategy = LowRiskComboStrategy(params={"rsi_max_buy": 55, "rsi_sell": 70})
    strategy.data = pd.DataFrame(
        {
            "Close": [10.0, 8.0, 12.0, 20.0, 15.0],
            "lower_band": [9.0, 9.0, 9.0, 9.0, 9.0],
            "upper_band": [16.0, 16.0, 16.0, 16.0, 16.0],
            "rsi": [40.0, 40.0, 80.0, 80.0, 80.0],
        }
    )
    assert not strategy.should_buy(2)
    strategy.data.loc[2, "rsi"] = 40.0
    assert strategy.should_buy(2)
    assert strategy.should_sell(4)


def test_sma_backtest_produces_a_round_trip():
    data = _v_shape()
    strategy = SMACrossoverStrategy(params={"fast": 5, "slow": 15, "quantity": 1})
    bt = Backtest(
        strategy,
        stock="TEST-EQ",
        start_date="2024-01-01",
        end_date="2024-04-01",
        data=data,
        initial_portfolio_value=100000,
    )
    sides = [o.transaction_type for o in bt.run()]
    assert "buy" in sides
    assert all(o.order_type == "backtest" for o in bt.orders)


def test_dual_mode_stop_loss_sells_without_changing_position():
    strategy = SMACrossoverStrategy(
        params={"fast": 2, "slow": 4, "quantity": 1, "stop_loss_pct": 2.0}
    )
    strategy.confirm_fill({"action": "buy", "quantity": 1, "price": 100.0})
    assert strategy.current_position == 1
    signal = strategy.on_new_tick(
        {"ltp": "97.5", "exchange_timestamp": "2024-01-02T09:15:00"},
        TOKEN,
    )
    assert signal is not None
    assert signal["action"] == "sell"
    assert signal["reason"] == "stop_loss"
    assert strategy.current_position == 1
    strategy.confirm_fill(signal)
    assert strategy.current_position == 0


def test_sma_live_emits_buy_on_completed_bar_without_pre_fill_position():
    strategy = SMACrossoverStrategy(
        params={"fast": 2, "slow": 4, "quantity": 1, "bar_interval_minutes": 1}
    )
    closes = [10, 10, 10, 10, 10, 11, 12, 13]
    start = datetime(2024, 1, 2, 9, 15)
    signals = []
    for i, px in enumerate(closes):
        tick_time = start + timedelta(minutes=i)
        signal = strategy.on_new_tick(
            {"ltp": str(px), "exchange_timestamp": tick_time.isoformat()},
            TOKEN,
        )
        if signal:
            signals.append(signal)
    assert any(s["action"] == "buy" for s in signals)
    assert strategy.current_position == 0


def test_indicators_sma_and_rsi():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    means = sma(close, 3)
    assert pd.isna(means.iloc[1])
    assert means.iloc[2] == pytest.approx(2.0)
    rsi_vals = rsi(pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.5]), 3)
    assert rsi_vals.notna().sum() > 0


def test_dual_mode_bar_signal_uses_capital_for_fractional_size():
    class _AlwaysBuy(DualModeStrategy):
        def process_data(self):
            return None

        def should_buy(self, current_index, portfolio=None):
            return current_index >= 1

        def should_sell(self, current_index, portfolio=None):
            return False

    strategy = _AlwaysBuy(
        "AlwaysBuy",
        {"quantity": 1, "position_fraction": 0.1, "capital": 10000, "window": 1},
    )
    strategy.data = pd.DataFrame({"Close": [100.0, 100.0], "timestamp": [0, 1]})
    signal = strategy._bar_signal()
    assert signal["action"] == "buy"
    assert signal["quantity"] == 10
