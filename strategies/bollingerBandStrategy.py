from typing import Optional

import pandas as pd

from strategies.dual_mode import DualModeStrategy
from strategies.indicators import atr, bollinger_bands, rsi, sma


class BollingerBandStrategy(DualModeStrategy):
    """
    Mean-reversion: buy when price reclaims the lower band, sell when it fails
    the upper band. Optional SMA trend filter, RSI confirmation, and ATR-based
    quantity sizing for live/paper trading.
    """

    def __init__(self, name: str = "BollingerBand", params: dict = None):
        params = dict(params or {})
        if "position_fraction" not in params and "quantity" not in params:
            params["position_fraction"] = 0.2
        super().__init__(name, params)
        self.bb_window = int(self.params.get("window", 20))
        self.std_multiplier = float(self.params.get("std_multiplier", 2))
        self.trend_sma = int(self.params.get("trend_sma", 0) or 0)
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_max_buy = self.params.get("rsi_max_buy")
        self.use_atr_size = bool(self.params.get("use_atr_size", False))
        self.atr_period = int(self.params.get("atr_period", 14))
        self.atr_risk_fraction = float(self.params.get("atr_risk_fraction", 0.01))
        self.window = max(self.bb_window, self.trend_sma, self.rsi_period, 2)
        self.desc = self.generate_desc()

    def generate_desc(self):
        return (
            f"{self.name}: BB({self.bb_window},{self.std_multiplier})"
            f"{f', trend SMA {self.trend_sma}' if self.trend_sma else ''}"
            f"{', RSI filter' if self.rsi_max_buy is not None else ''}. "
            "Buy on reclaim of lower band; sell on rejection of upper band."
        )

    def process_data(self):
        if self.data is None or len(self.data) < self.bb_window:
            return
        close = self.data["Close"]
        middle, upper, lower = bollinger_bands(close, self.bb_window, self.std_multiplier)
        self.data["middle_band"] = middle
        self.data["upper_band"] = upper
        self.data["lower_band"] = lower
        if self.trend_sma:
            self.data["trend_sma"] = sma(close, self.trend_sma)
        if self.rsi_max_buy is not None:
            self.data["rsi"] = rsi(close, self.rsi_period)
        if self.use_atr_size and {"High", "Low"} <= set(self.data.columns):
            self.data["atr"] = atr(self.data["High"], self.data["Low"], close, self.atr_period)

    def _row_ok(self, current_index: int) -> bool:
        return self.data is not None and current_index >= self.bb_window

    def should_buy(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if not self._row_ok(current_index):
            return False
        current = self.data.iloc[current_index]
        previous = self.data.iloc[current_index - 1]
        if pd.isna(previous.get("lower_band")) or pd.isna(current.get("lower_band")):
            return False
        crossed = previous["Close"] <= previous["lower_band"] and current["Close"] > current["lower_band"]
        if not crossed:
            return False
        if self.trend_sma and "trend_sma" in self.data.columns:
            if pd.isna(current["trend_sma"]) or current["Close"] < current["trend_sma"]:
                return False
        if self.rsi_max_buy is not None and "rsi" in self.data.columns:
            if pd.isna(current["rsi"]) or current["rsi"] > float(self.rsi_max_buy):
                return False
        return True

    def should_sell(self, current_index: int, portfolio: Optional[dict] = None) -> bool:
        if not self._row_ok(current_index):
            return False
        current = self.data.iloc[current_index]
        previous = self.data.iloc[current_index - 1]
        if pd.isna(previous.get("upper_band")) or pd.isna(current.get("upper_band")):
            return False
        return bool(
            previous["Close"] >= previous["upper_band"]
            and current["Close"] < current["upper_band"]
        )

    def select_quantity(self, price: float, portfolio_value: float) -> int:
        if (
            self.use_atr_size
            and self.data is not None
            and "atr" in self.data.columns
            and len(self.data)
            and price > 0
        ):
            last_atr = self.data["atr"].iloc[-1]
            if pd.notna(last_atr) and last_atr > 0:
                risk_cash = portfolio_value * self.atr_risk_fraction
                qty = int(risk_cash / last_atr)
                return max(qty, 1)
        return super().select_quantity(price, portfolio_value)
