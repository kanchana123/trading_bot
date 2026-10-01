from typing import Optional

import pandas as pd

from strategies.dual_mode import DualModeStrategy
from strategies.indicators import rsi, sma


class RSIReversionStrategy(DualModeStrategy):
    """Buy when RSI crosses up through oversold; sell when it crosses down through overbought."""

    def __init__(self, name: str = "RSIReversion", params: dict = None):
        super().__init__(name, params or {})
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.oversold = float(self.params.get("oversold", 30))
        self.overbought = float(self.params.get("overbought", 70))
        self.trend_sma = int(self.params.get("trend_sma", 0) or 0)
        self.window = max(self.rsi_period, self.trend_sma, 2)
        self.desc = self.generate_desc()

    def generate_desc(self):
        return (
            f"{self.name}: RSI({self.rsi_period}) buy < {self.oversold} reclaim, "
            f"sell > {self.overbought} fade"
            f"{f', trend SMA {self.trend_sma}' if self.trend_sma else ''}."
        )

    def process_data(self):
        if self.data is None or len(self.data) < self.rsi_period:
            return
        self.data["rsi"] = rsi(self.data["Close"], self.rsi_period)
        if self.trend_sma:
            self.data["trend_sma"] = sma(self.data["Close"], self.trend_sma)

    def should_buy(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if self.data is None or current_index < 1 or "rsi" not in self.data.columns:
            return False
        prev = self.data.iloc[current_index - 1]
        curr = self.data.iloc[current_index]
        if pd.isna(prev.get("rsi")) or pd.isna(curr.get("rsi")):
            return False
        crossed = prev["rsi"] < self.oversold <= curr["rsi"]
        if not crossed:
            return False
        if self.trend_sma and "trend_sma" in self.data.columns:
            if pd.isna(curr["trend_sma"]) or curr["Close"] < curr["trend_sma"]:
                return False
        return True

    def should_sell(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if self.data is None or current_index < 1 or "rsi" not in self.data.columns:
            return False
        prev = self.data.iloc[current_index - 1]
        curr = self.data.iloc[current_index]
        if pd.isna(prev.get("rsi")) or pd.isna(curr.get("rsi")):
            return False
        return bool(prev["rsi"] > self.overbought >= curr["rsi"])
