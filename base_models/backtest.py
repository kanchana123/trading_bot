from datetime import datetime
from typing import List, Optional
import logging
import pandas as pd
from base_models.orders import Order
from base_models.ohlc import normalize_ohlc_columns

logger = logging.getLogger(__name__)


class Backtest:
    """
    Runs a backtest for a given trading strategy.
    """

    def __init__(
        self,
        strategy,
        stock: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame = None,
        initial_portfolio_value: float = 100000.0,
        interval: str = "ONE_DAY",
        angel_api=None,
    ):
        self.strategy = strategy
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.angel_api = angel_api
        self.data = normalize_ohlc_columns(data) if data is not None else None
        if self.data is None or (hasattr(self.data, "empty") and self.data.empty):
            self.data = self._get_historical_data()
        self.strategy.set_data(self.data)
        if self.data is not None and not self.data.empty:
            self.strategy.process_data()
        self.orders: List[Order] = []
        self.portfolio_value_history: List[float] = []
        self.initial_portfolio_value = initial_portfolio_value
        self.current_position = 0

    def _get_historical_data(self) -> pd.DataFrame:
        start_date_obj = datetime.strptime(self.start_date[:10], "%Y-%m-%d")
        end_date_obj = datetime.strptime(self.end_date[:10], "%Y-%m-%d")
        if start_date_obj > end_date_obj:
            raise ValueError("Start date cannot be greater than end date.")

        api = self.angel_api
        if api is None:
            try:
                from base_models.angel_api import AngelAPI

                api = AngelAPI()
            except Exception as e:
                logger.error("Could not initialize AngelAPI for historical data: %s", e)
                return pd.DataFrame()

        if not getattr(api, "is_session_active", lambda: False)():
            logger.error("Angel API session is not active; cannot fetch historical data.")
            return pd.DataFrame()

        from_date = f"{self.start_date[:10]} 09:15"
        to_date = f"{self.end_date[:10]} 15:30"
        df = api.download_historical_data(
            exchange="NSE",
            instrument_symbol=self.stock,
            interval=self.interval,
            from_date=from_date,
            to_date=to_date,
        )
        if df is None:
            return pd.DataFrame()
        return normalize_ohlc_columns(df)

    def run(self) -> List[Order]:
        logger.info(
            "Running backtest for %s from %s to %s with %s",
            self.stock,
            self.start_date,
            self.end_date,
            getattr(self.strategy, "name", type(self.strategy).__name__),
        )
        if self.data is None or self.data.empty:
            logger.warning("No data available for backtesting.")
            return []

        if "Close" not in self.data.columns:
            raise ValueError("Backtest data must include a Close column.")

        cash_balance = self.initial_portfolio_value
        self.current_position = 0
        self.portfolio_value_history = []

        initial_total_value = cash_balance
        self.portfolio_value_history.append(initial_total_value)
        self.strategy.data["portfolio_value"] = 0.0
        if not self.strategy.data.empty:
            first_idx = self.strategy.data.index[0]
            self.strategy.data.loc[first_idx, "portfolio_value"] = initial_total_value

        if "timestamp" not in self.strategy.data.columns and isinstance(
            self.strategy.data.index, pd.DatetimeIndex
        ):
            self.strategy.data["timestamp"] = self.strategy.data.index

        hist = max(int(getattr(self.strategy, "window", 1) or 1), 1)

        for i in range(hist, len(self.data)):
            current_price = self.data.iloc[i]["Close"]
            if "timestamp" in self.data.columns:
                current_timestamp = self.data.iloc[i]["timestamp"]
            else:
                current_timestamp = self.data.index[i]
            data_idx = self.strategy.data.index[i]

            current_total_value = cash_balance + (self.current_position * current_price)
            portfolio_value_before_trade = current_total_value

            should_buy_signal = self.strategy.should_buy(i)
            should_sell_signal = self.strategy.should_sell(i)

            if self.current_position == 0 and should_buy_signal:
                quantity_to_buy = self.strategy.select_quantity(
                    current_price, portfolio_value_before_trade
                )
                cost = current_price * quantity_to_buy

                if cash_balance >= cost and quantity_to_buy > 0:
                    metadata = self._build_metadata("Buy", i)
                    order = Order(
                        "backtest",
                        "buy",
                        current_price,
                        quantity_to_buy,
                        metadata,
                        portfolio_value_before_trade,
                        current_timestamp,
                    )
                    self.orders.append(order)
                    cash_balance -= cost
                    self.current_position += quantity_to_buy
                    current_total_value = cash_balance + (
                        self.current_position * current_price
                    )
                else:
                    logger.info(
                        "Skipping buy at index %s: Insufficient cash (%.2f < %.2f)",
                        i,
                        cash_balance,
                        cost,
                    )
                    current_total_value = portfolio_value_before_trade

            elif self.current_position > 0 and should_sell_signal:
                quantity_to_sell = self.current_position
                proceeds = current_price * quantity_to_sell
                metadata = self._build_metadata("Sell", i)
                order = Order(
                    "backtest",
                    "sell",
                    current_price,
                    quantity_to_sell,
                    metadata,
                    portfolio_value_before_trade,
                    current_timestamp,
                )
                self.orders.append(order)
                cash_balance += proceeds
                self.current_position -= quantity_to_sell
                current_total_value = cash_balance
            else:
                current_total_value = cash_balance + (
                    self.current_position * current_price
                )

            self.portfolio_value_history.append(current_total_value)
            self.strategy.data.loc[data_idx, "portfolio_value"] = current_total_value

        return self.orders

    def _build_metadata(self, action: str, index: int) -> str:
        metadata = f"Strategy: {self.strategy.name} - {action} signal at index: {index}"
        if "momentum" in self.data.columns:
            value = self.data.iloc[index]["momentum"]
            if pd.notna(value):
                metadata += f", momentum: {value}"
        return metadata
