# in basic_strategy.py

from typing import Dict, Optional
import pandas as pd
import numpy as np

from base_models.ohlc import normalize_ohlc_columns


class Strategy:
    """
    Base class for trading strategies.
    """

    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initializes the strategy.

        Args:
            name (str): The name of the strategy.
            params (Optional[Dict], optional): Parameters for the strategy. Defaults to None.
        """
        self.name = name
        self.params: Dict = params or {}
        self.data = None  # DataFrame to hold the processed data
        self.desc = ""

    def set_data(self, data: pd.DataFrame):
        """
        Sets the data for the strategy.

        Args:
            data (pd.DataFrame): The input data, typically OHLCV data.
        """
        self.data = normalize_ohlc_columns(data)

    def process_data(self):
        """
        Processes the entire historical dataset to calculate indicators and other
        necessary values before the backtest run starts.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_data method.")

    def should_buy(self, current_index: int, portfolio: Optional[Dict] = None) -> bool:
        """
        Determines if a buy signal is generated.

        Args:
            current_index(int): index of the current data.
            portfolio (Optional[Dict], optional): Dictionary with portfolio state
                                        (e.g., cash, position, value). Defaults to None.
        Returns:
            bool: True if a buy signal is generated, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement should_buy method.")

    def should_sell(self, current_index: int, portfolio: Optional[Dict] = None) -> bool:
        """
        Determines if a sell signal is generated.

        Args:
            current_index(int): index of the current data.
            portfolio (Optional[Dict], optional): Dictionary with portfolio state
                                        (e.g., cash, position, value). Defaults to None.
        Returns:
            bool: True if a sell signal is generated, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement should_sell method.")

    def should_hold(self, current_index: int, portfolio: Optional[Dict] = None) -> bool:
        """
        Determines if the action should be hold.

        Args:
            current_index(int): index of the current data.
            portfolio (Optional[Dict], optional): Dictionary with portfolio state
                                        (e.g., cash, position, value). Defaults to None.
        Returns:
            bool: True if the action should be hold, False otherwise.
        """
        return not self.should_buy(current_index, portfolio) and not self.should_sell(current_index, portfolio)

    def select_quantity(self, price: float, portfolio_value: float) -> int:
        """
        Calculates the number of shares/contracts to trade.

        Args:
            price (float): The current price.
            portfolio_value(float): current portfolio value

        Returns:
            int: The quantity to buy or sell.
        """
        raise NotImplementedError("Subclasses must implement select_quantity method.")
