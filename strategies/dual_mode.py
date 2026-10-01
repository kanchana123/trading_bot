import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from base_models.ohlc import normalize_ohlc_columns
from base_models.strategy import Strategy
from strategies.base_strategy_rt import RealTimeStrategy

logger = logging.getLogger(__name__)


class DualModeStrategy(Strategy, RealTimeStrategy):
    """
    One strategy class for backtest, paper (virtual), and real trading.

    Subclasses implement process_data / should_buy / should_sell / select_quantity.
    Live ticks are aggregated into bars, then those same methods run on completed bars.
    """

    def __init__(self, name: str, params: Optional[Dict] = None):
        params = params or {}
        Strategy.__init__(self, name, params)
        RealTimeStrategy.__init__(self, name, params)
        self.fixed_quantity = int(self.params.get("quantity", 1) or 1)
        self.position_fraction = float(self.params.get("position_fraction", 0) or 0)
        self.bar_interval_minutes = int(self.params.get("bar_interval_minutes", 1) or 1)
        self.max_historical_bars = int(self.params.get("max_historical_bars", 300) or 300)
        self.stop_loss_pct = float(self.params.get("stop_loss_pct", 0) or 0)
        self.window = int(self.params.get("window", 1) or 1)

        self.current_bar_ticks: List[Dict] = []
        self.current_bar_start_time: Optional[datetime] = None
        self.historical_bars = pd.DataFrame(
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        self.current_position = 0
        self.avg_entry = 0.0
        self._last_signal_bar_time = None
        self.desc = f"{self.name} params={self.params}"

    def generate_desc(self) -> str:
        return f"{self.name} params={self.params}"

    def select_quantity(self, price: float, portfolio_value: float) -> int:
        if price <= 0:
            return 1
        if self.position_fraction > 0 and portfolio_value:
            qty = int(portfolio_value * self.position_fraction / price)
            return max(qty, 1)
        return max(self.fixed_quantity, 1)

    def confirm_fill(self, trade_signal: Dict[str, Any]):
        quantity = int(trade_signal.get("quantity") or 0)
        action = (trade_signal.get("action") or "").lower()
        price = float(trade_signal.get("price") or 0)
        if action == "buy":
            new_qty = self.current_position + quantity
            if new_qty > 0 and price:
                self.avg_entry = (
                    (self.avg_entry * self.current_position) + (price * quantity)
                ) / new_qty
            self.current_position = new_qty
        elif action == "sell":
            self.current_position = max(self.current_position - quantity, 0)
            if self.current_position == 0:
                self.avg_entry = 0.0

    def _parse_tick(self, tick_data: Dict, token_details: Dict):
        price_str = tick_data.get("ltp") or tick_data.get("last_traded_price")
        if price_str is None:
            return None, None
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            logger.warning(
                "Could not parse price from tick for %s: %s",
                token_details.get("symbol"),
                price_str,
            )
            return None, None

        ts_value = (
            tick_data.get("exchange_timestamp")
            or tick_data.get("feed_timestamp")
            or tick_data.get("ft")
        )
        tick_time = datetime.now()
        if ts_value is None and "tt" in tick_data:
            try:
                ts_value = int(tick_data["tt"]) / 1000
            except (TypeError, ValueError):
                ts_value = None
        if isinstance(ts_value, str):
            try:
                tick_time = pd.to_datetime(ts_value).to_pydatetime()
            except Exception:
                tick_time = datetime.now()
        elif isinstance(ts_value, (int, float)):
            tick_time = datetime.fromtimestamp(ts_value)
        return price, tick_time

    def _align_bar_start(self, tick_time: datetime) -> datetime:
        start = tick_time.replace(second=0, microsecond=0)
        offset = start.minute % self.bar_interval_minutes
        return start - timedelta(minutes=offset)

    def _finalize_bar(self):
        if not self.current_bar_ticks or not self.current_bar_start_time:
            return
        prices = [t["price"] for t in self.current_bar_ticks]
        new_bar = pd.DataFrame(
            [
                {
                    "timestamp": self.current_bar_start_time,
                    "Open": prices[0],
                    "High": max(prices),
                    "Low": min(prices),
                    "Close": prices[-1],
                    "Volume": 0,
                }
            ]
        )
        self.historical_bars = pd.concat([self.historical_bars, new_bar], ignore_index=True)
        if len(self.historical_bars) > self.max_historical_bars:
            self.historical_bars = self.historical_bars.iloc[-self.max_historical_bars :]
        self.set_data(normalize_ohlc_columns(self.historical_bars.copy()))
        self.process_data()

    def _stop_loss_signal(self, price: float) -> Optional[Dict[str, Any]]:
        if (
            self.current_position > 0
            and self.stop_loss_pct > 0
            and self.avg_entry > 0
            and price <= self.avg_entry * (1 - self.stop_loss_pct / 100.0)
        ):
            return {
                "action": "sell",
                "quantity": self.current_position,
                "price": price,
                "order_type": "LIMIT",
                "reason": "stop_loss",
            }
        return None

    def _bar_signal(self) -> Optional[Dict[str, Any]]:
        if self.data is None or self.data.empty:
            return None
        idx = len(self.data) - 1
        if idx < max(int(self.window or 1), 1):
            return None
        bar_time = None
        if "timestamp" in self.data.columns:
            bar_time = self.data["timestamp"].iloc[idx]
        if bar_time is not None and self._last_signal_bar_time == bar_time:
            return None
        close = float(self.data["Close"].iloc[idx])
        capital = float(self.params.get("capital") or 0)
        if self.position_fraction > 0:
            fallback = close / max(self.position_fraction, 0.01)
            qty = self.select_quantity(close, capital or fallback)
        else:
            qty = self.fixed_quantity
        signal = None
        if self.current_position <= 0 and self.should_buy(idx):
            signal = {"action": "buy", "quantity": qty, "price": close, "order_type": "LIMIT"}
        elif self.current_position > 0 and self.should_sell(idx):
            signal = {
                "action": "sell",
                "quantity": self.current_position,
                "price": close,
                "order_type": "LIMIT",
            }
        if signal is not None:
            self._last_signal_bar_time = bar_time
        return signal

    def on_new_tick(self, tick_data: Dict, token_details: Dict) -> Optional[Dict[str, Any]]:
        price, tick_time = self._parse_tick(tick_data, token_details)
        if price is None:
            return None

        stop = self._stop_loss_signal(price)
        if stop:
            return stop

        if (
            not self.current_bar_start_time
            or tick_time >= self.current_bar_start_time + timedelta(minutes=self.bar_interval_minutes)
        ):
            self._finalize_bar()
            signal = self._bar_signal()
            self.current_bar_start_time = self._align_bar_start(tick_time)
            self.current_bar_ticks = []
            self.current_bar_ticks.append({"price": price, "time": tick_time})
            return signal

        self.current_bar_ticks.append({"price": price, "time": tick_time})
        return None
