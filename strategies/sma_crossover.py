from typing import Optional

import pandas as pd

from strategies.dual_mode import DualModeStrategy
from strategies.indicators import sma


class SMACrossoverStrategy(DualModeStrategy):
    """Buy when fast SMA crosses above slow SMA; sell on the opposite cross."""

    def __init__(self, name: str = "SMACrossover", params: dict = None):
        super().__init__(name, params or {})
        self.fast = int(self.params.get("fast", 10))
        self.slow = int(self.params.get("slow", 30))
        if self.fast >= self.slow:
            self.slow = self.fast + 10
        self.window = self.slow
        self.desc = self.generate_desc()

    def generate_desc(self):
        return (
            f"{self.name}: SMA({self.fast}) vs SMA({self.slow}). "
            "Long when the fast average crosses above the slow average."
        )

    def process_data(self):
        if self.data is None or len(self.data) < self.slow:
            return
        close = self.data["Close"]
        self.data["sma_fast"] = sma(close, self.fast)
        self.data["sma_slow"] = sma(close, self.slow)

    def should_buy(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if self.data is None or current_index < 1:
            return False
        prev = self.data.iloc[current_index - 1]
        curr = self.data.iloc[current_index]
        if pd.isna(prev.get("sma_fast")) or pd.isna(curr.get("sma_slow")):
            return False
        return bool(prev["sma_fast"] <= prev["sma_slow"] and curr["sma_fast"] > curr["sma_slow"])

    def should_sell(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if self.data is None or current_index < 1:
            return False
        prev = self.data.iloc[current_index - 1]
        curr = self.data.iloc[current_index]
        if pd.isna(prev.get("sma_fast")) or pd.isna(curr.get("sma_slow")):
            return False
        return bool(prev["sma_fast"] >= prev["sma_slow"] and curr["sma_fast"] < curr["sma_slow"])
