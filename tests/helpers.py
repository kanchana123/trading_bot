from datetime import datetime

import pandas as pd


def ohlc_frame(n=40, start=100.0, freq="D", start_date="2024-01-01"):
    rows = []
    price = start
    for i in range(n):
        price += 1 if i % 7 != 0 else -2
        rows.append(
            {
                "Open": price - 0.5,
                "High": price + 1,
                "Low": price - 1,
                "Close": price,
                "Volume": 1000 + i,
            }
        )
    idx = pd.date_range(start_date, periods=n, freq=freq)
    return pd.DataFrame(rows, index=idx)


class ScriptedStrategy:
    """Deterministic strategy for Backtest accounting tests."""

    def __init__(self, buy_at=2, sell_at=5, quantity=1, window=1):
        self.name = "Scripted"
        self.window = window
        self.data = None
        self.buy_at = buy_at
        self.sell_at = sell_at
        self.quantity = quantity

    def set_data(self, data):
        self.data = data

    def process_data(self):
        return None

    def should_buy(self, current_index: int) -> bool:
        return current_index == self.buy_at

    def should_sell(self, current_index: int) -> bool:
        return current_index == self.sell_at

    def select_quantity(self, price: float, portfolio_value: float) -> int:
        return self.quantity
