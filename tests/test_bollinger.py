import sqlite3

import pandas as pd
import pytest

from strategies.bollingerBandStrategy import BollingerBandStrategy
from tests.helpers import ohlc_frame


def test_bollinger_process_data_adds_bands():
    strategy = BollingerBandStrategy(params={"window": 5, "std_multiplier": 2})
    strategy.set_data(ohlc_frame(n=20))
    strategy.process_data()
    assert {"middle_band", "upper_band", "lower_band"} <= set(strategy.data.columns)
    assert strategy.data["middle_band"].notna().sum() > 0


def test_bollinger_accepts_lowercase_close():
    strategy = BollingerBandStrategy(params={"window": 5})
    df = ohlc_frame(n=15).rename(columns=str.lower)
    strategy.set_data(df)
    strategy.process_data()
    assert "Close" in strategy.data.columns
    assert "lower_band" in strategy.data.columns


def test_bollinger_buy_on_cross_up_from_lower_band():
    strategy = BollingerBandStrategy(params={"window": 2})
    strategy.data = pd.DataFrame(
        {
            "Close": [10.0, 10.0, 5.0, 12.0],
            "lower_band": [9.0, 9.0, 9.0, 9.0],
            "upper_band": [15.0, 15.0, 15.0, 15.0],
        }
    )
    assert not strategy.should_buy(2)
    assert strategy.should_buy(3)
    assert not strategy.should_sell(3)


def test_bollinger_sell_on_cross_down_from_upper_band():
    strategy = BollingerBandStrategy(params={"window": 2})
    strategy.data = pd.DataFrame(
        {
            "Close": [10.0, 10.0, 20.0, 12.0],
            "lower_band": [5.0, 5.0, 5.0, 5.0],
            "upper_band": [15.0, 15.0, 15.0, 15.0],
        }
    )
    assert strategy.should_sell(3)
    assert not strategy.should_buy(3)


def test_bollinger_select_quantity_uses_twenty_percent():
    strategy = BollingerBandStrategy()
    assert strategy.select_quantity(price=100, portfolio_value=10000) == 20
    assert strategy.select_quantity(price=1_000_000, portfolio_value=100) == 1
