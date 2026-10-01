from typing import Optional

import pandas as pd

from strategies.dual_mode import DualModeStrategy
from strategies.indicators import bollinger_bands, kernel_momentum, rsi


class LowRiskComboStrategy(DualModeStrategy):
    """
    Conservative long-only: enter only when a kernel or Bollinger buy agrees
    with a non-overbought RSI. Exit on the opposite signal, RSI overbought, or
    live stop-loss (stop_loss_pct).
    """

    def __init__(self, name: str = "LowRiskCombo", params: dict = None):
        params = dict(params or {})
        params.setdefault("stop_loss_pct", 0.5)
        params.setdefault("position_fraction", 0.15)
        super().__init__(name, params)
        self.bb_window = int(self.params.get("window", 20))
        self.std_multiplier = float(self.params.get("std_multiplier", 2))
        self.kernel = self.params.get("kernel", [-2, -1, 0, 1, 2])
        self.threshold = float(self.params.get("threshold", 1.5))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_max_buy = float(self.params.get("rsi_max_buy", 55))
        self.rsi_sell = float(self.params.get("rsi_sell", 70))
        self.window = max(self.bb_window, len(self.kernel), self.rsi_period, 2)
        self.desc = self.generate_desc()

    def generate_desc(self):
        return (
            f"{self.name}: kernel OR Bollinger buy with RSI < {self.rsi_max_buy}; "
            f"exit on opposite signal, RSI > {self.rsi_sell}, or stop {self.stop_loss_pct}%."
        )

    def process_data(self):
        if self.data is None or self.data.empty:
            return
        close = self.data["Close"]
        middle, upper, lower = bollinger_bands(close, self.bb_window, self.std_multiplier)
        self.data["middle_band"] = middle
        self.data["upper_band"] = upper
        self.data["lower_band"] = lower
        self.data["rsi"] = rsi(close, self.rsi_period)
        if {"High", "Low"} <= set(self.data.columns):
            self.data["momentum"] = kernel_momentum(
                self.data["High"], self.data["Low"], close, self.kernel
            )

    def _rsi_allows_buy(self, row) -> bool:
        if "rsi" not in row.index or pd.isna(row.get("rsi")):
            return True
        return row["rsi"] < self.rsi_max_buy

    def should_buy(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if self.data is None or current_index < 1:
            return False
        prev = self.data.iloc[current_index - 1]
        curr = self.data.iloc[current_index]
        if not self._rsi_allows_buy(curr):
            return False
        bb_buy = (
            pd.notna(prev.get("lower_band"))
            and pd.notna(curr.get("lower_band"))
            and prev["Close"] <= prev["lower_band"]
            and curr["Close"] > curr["lower_band"]
        )
        kernel_buy = False
        if "momentum" in self.data.columns:
            if pd.notna(prev.get("momentum")) and pd.notna(curr.get("momentum")):
                kernel_buy = prev["momentum"] > -self.threshold and curr["momentum"] <= -self.threshold
        return bool(bb_buy or kernel_buy)

    def should_sell(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if self.data is None or current_index < 1:
            return False
        prev = self.data.iloc[current_index - 1]
        curr = self.data.iloc[current_index]
        rsi_exit = pd.notna(curr.get("rsi")) and curr["rsi"] >= self.rsi_sell
        bb_sell = (
            pd.notna(prev.get("upper_band"))
            and pd.notna(curr.get("upper_band"))
            and prev["Close"] >= prev["upper_band"]
            and curr["Close"] < curr["upper_band"]
        )
        kernel_sell = False
        if "momentum" in self.data.columns:
            if pd.notna(prev.get("momentum")) and pd.notna(curr.get("momentum")):
                kernel_sell = prev["momentum"] < self.threshold and curr["momentum"] >= self.threshold
        return bool(rsi_exit or bb_sell or kernel_sell)
