import pandas as pd
from typing import Dict, Optional
import numpy as np


class TechnicalAnalyzer:
    """
    A class to calculate various technical indicators for a given stock data.
    Indicator parameters can be customized during initialization.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initializes the TechnicalAnalyzer with custom or default parameters.

        Args:
            params (Optional[Dict], optional): A dictionary to override default
                indicator parameters. For example:
                {
                    "sma_short": 10, "sma_long": 30, "rsi_period": 10
                }
                Defaults to None, which uses standard values.
        """
        if params is None:
            params = {}

        # Default parameters
        self.params = {
            "sma_short": 20,
            "sma_long": 50,
            "ema_short": 20,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_window": 20,
            "bb_std_dev": 2,
            "coppock_long_roc": 14,
            "coppock_short_roc": 11,
            "coppock_wma_period": 10,
        }
        # Update with any user-provided params
        self.params.update(params)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a suite of technical indicators and adds them to the DataFrame.

        The indicators include:
        - Simple Moving Averages (e.g., SMA_20, SMA_50)
        - Exponential Moving Average (e.g., EMA_20)
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD, MACD_signal)
        - Bollinger Bands (Bollinger_High, Bollinger_Low)
        - Coppock Curve

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data. Must contain a 'Close' column.

        Returns:
            pd.DataFrame: The DataFrame with added indicator columns.
        """
        if "Close" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Close' column.")

        # Find the longest window period needed for calculations
        coppock_lookback = self.params["coppock_long_roc"] + self.params["coppock_wma_period"] - 1
        max_window = max(
            self.params["sma_long"],
            self.params["rsi_period"],
            self.params["macd_slow"],
            self.params["bb_window"],
            coppock_lookback,
        )
        if len(data) < max_window:
            print(
                f"Warning: Data length ({len(data)}) is shorter than the longest "
                f"indicator window ({max_window}). Indicators will be mostly NaN."
            )

        df = data.copy()

        # Simple Moving Averages (SMA)
        sma_short_win = self.params["sma_short"]
        sma_long_win = self.params["sma_long"]
        df[f"SMA_{sma_short_win}"] = df["Close"].rolling(window=sma_short_win).mean()
        df[f"SMA_{sma_long_win}"] = df["Close"].rolling(window=sma_long_win).mean()

        # Exponential Moving Average (EMA)
        ema_short_span = self.params["ema_short"]
        df[f"EMA_{ema_short_span}"] = df["Close"].ewm(span=ema_short_span, adjust=False).mean()

        # Relative Strength Index (RSI)
        rsi_period = self.params["rsi_period"]
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, 1e-9)  # Avoid division by zero
        df["RSI"] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        ema_fast = df["Close"].ewm(span=self.params["macd_fast"], adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.params["macd_slow"], adjust=False).mean()
        df["MACD"] = ema_fast - ema_slow
        df["MACD_signal"] = df["MACD"].ewm(span=self.params["macd_signal"], adjust=False).mean()

        # Bollinger Bands
        bb_window = self.params["bb_window"]
        bb_std = self.params["bb_std_dev"]
        sma_for_bb = df["Close"].rolling(window=bb_window).mean()
        std_for_bb = df["Close"].rolling(window=bb_window).std()
        df["Bollinger_High"] = sma_for_bb + (std_for_bb * bb_std)
        df["Bollinger_Low"] = sma_for_bb - (std_for_bb * bb_std)

        # Coppock Curve
        coppock_long_roc_n = self.params["coppock_long_roc"]
        coppock_short_roc_n = self.params["coppock_short_roc"]
        coppock_wma_n = self.params["coppock_wma_period"]

        # Rate of Change (ROC)
        roc_long = df["Close"].pct_change(coppock_long_roc_n) * 100
        roc_short = df["Close"].pct_change(coppock_short_roc_n) * 100

        # Sum of ROCs
        roc_sum = roc_long + roc_short

        # Weighted Moving Average (WMA) of the sum
        weights = np.arange(1, coppock_wma_n + 1)
        wma_func = lambda x: np.dot(x, weights) / weights.sum()
        df["Coppock_Curve"] = roc_sum.rolling(window=coppock_wma_n).apply(wma_func, raw=True)

        return df