from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from strategies.action_price.KernelTrader import KernelBacktestAdapter, KernelStrategy
from tests.helpers import ohlc_frame


TOKEN = {"instrumentToken": "2885", "symbol": "RELIANCE-EQ", "exchange": "NSE_EQ"}


def test_kernel_adapter_momentum_is_causal():
    data = ohlc_frame(n=30)
    adapter = KernelBacktestAdapter(params={"kernel": [-2, -1, 0, 1, 2], "threshold": 0.1})
    adapter.set_data(data.copy())
    adapter.process_data()
    original = adapter.data["momentum"].copy()

    mutated = data.copy()
    mutated.iloc[-1, mutated.columns.get_loc("Close")] = 999
    adapter.set_data(mutated)
    adapter.process_data()
    pd.testing.assert_series_equal(
        original.iloc[:-1].reset_index(drop=True),
        adapter.data["momentum"].iloc[:-1].reset_index(drop=True),
    )


def test_confirm_fill_updates_position_only_on_success_path():
    strategy = KernelStrategy(params={"quantity": 3})
    assert strategy.current_position == 0
    strategy.confirm_fill({"action": "buy", "quantity": 3})
    assert strategy.current_position == 3
    strategy.confirm_fill({"action": "sell", "quantity": 3})
    assert strategy.current_position == 0


def test_on_new_tick_does_not_change_position_before_fill():
    strategy = KernelStrategy(params={"threshold": 1.5, "quantity": 1})
    strategy.historical_bars = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 15), datetime(2024, 1, 1, 9, 16)],
            "Open": [10.0, 10.0],
            "High": [12.0, 12.0],
            "Low": [8.0, 8.0],
            "Close": [11.0, 11.0],
            "Volume": [1, 1],
            "momentum": [0.0, -2.0],
        }
    )
    strategy.current_bar_start_time = datetime(2024, 1, 1, 9, 17)
    strategy.current_bar_ticks = []
    signal = strategy.on_new_tick(
        {"ltp": "11.2", "exchange_timestamp": "2024-01-01T09:17:10"},
        TOKEN,
    )
    assert signal is not None
    assert signal["action"] == "buy"
    assert strategy.current_position == 0


def test_on_new_tick_emits_at_most_one_signal_per_bar():
    strategy = KernelStrategy(params={"threshold": 1.5, "quantity": 1})
    strategy.historical_bars = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 15), datetime(2024, 1, 1, 9, 16)],
            "Open": [10.0, 10.0],
            "High": [12.0, 12.0],
            "Low": [8.0, 8.0],
            "Close": [11.0, 11.0],
            "Volume": [1, 1],
            "momentum": [0.0, -2.0],
        }
    )
    strategy.current_bar_start_time = datetime(2024, 1, 1, 9, 17)
    strategy.current_bar_ticks = []
    first = strategy.on_new_tick(
        {"ltp": "11.2", "exchange_timestamp": "2024-01-01T09:17:10"},
        TOKEN,
    )
    second = strategy.on_new_tick(
        {"ltp": "11.3", "exchange_timestamp": "2024-01-01T09:17:20"},
        TOKEN,
    )
    assert first is not None
    assert second is None
