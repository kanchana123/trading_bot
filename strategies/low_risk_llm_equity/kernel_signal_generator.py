import numpy as np
import pandas as pd
from scipy.signal import convolve
from typing import Dict, Optional


class KernelSignalGenerator:
    """
    Generates trading signals using a kernel-based momentum calculation.

    This class adapts the core logic from the `KernelTrader` strategy into a
    reusable module. It can be instantiated and used by other strategies to get
    kernel-based buy/sell signals.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initializes the signal generator.

        Args:
            params (Optional[Dict], optional): Parameters for the kernel method.
                - 'kernel': A list or numpy array for the convolution.
                - 'threshold': The momentum value to trigger a signal.
                Defaults to None, which uses standard values.
        """
        if params is None:
            params = {}
        # Default parameters adapted from KernelTrader
        self.kernel = np.array(params.get("kernel", [-2, -1, 0, 1, 2]))
        self.threshold = params.get("threshold", 1.5)
        self.window = len(self.kernel)
        self.data_with_momentum = pd.DataFrame()

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the momentum signal for the entire dataset and stores it internally.
        This should be called once before generating signals in a loop.

        Args:
            data (pd.DataFrame): The full historical OHLCV data.

        Returns:
            pd.DataFrame: The data with an added 'kernel_momentum' column.
        """
        if data.empty:
            raise ValueError("Input data for kernel processing cannot be empty.")

        required_cols = ["High", "Low", "Close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")

        df = data.copy()
        mid_price = (df["High"] + df["Low"]) / 2
        signal_values = (df["Close"] - mid_price).values

        if len(signal_values) >= self.window:
            momentum = convolve(signal_values, self.kernel, mode="valid")
            # The result of 'valid' convolution is shorter. We pad it at the beginning.
            padding_size = len(df) - len(momentum)
            padded_momentum = np.pad(
                momentum, (padding_size, 0), "constant", constant_values=np.nan
            )
            df["kernel_momentum"] = padded_momentum
        else:
            df["kernel_momentum"] = np.nan

        self.data_with_momentum = df
        return self.data_with_momentum

    def should_buy(self, current_index: int) -> bool:
        """
        Determines if a buy signal is generated at the current index based on
        pre-calculated momentum.

        Args:
            current_index (int): The index of the current data point to check.

        Returns:
            bool: True if a buy signal is generated, False otherwise.
        """
        if self.data_with_momentum.empty or current_index < 1 or current_index >= len(self.data_with_momentum):
            return False

        prev_momentum = self.data_with_momentum["kernel_momentum"].iloc[current_index - 1]
        curr_momentum = self.data_with_momentum["kernel_momentum"].iloc[current_index]

        if pd.isna(prev_momentum) or pd.isna(curr_momentum):
            return False

        return bool(prev_momentum > -self.threshold and curr_momentum <= -self.threshold)

    def should_sell(self, current_index: int) -> bool:
        """
        Determines if a sell signal is generated at the current index.
        """
        if self.data_with_momentum.empty or current_index < 1 or current_index >= len(self.data_with_momentum):
            return False

        prev_momentum = self.data_with_momentum["kernel_momentum"].iloc[current_index - 1]
        curr_momentum = self.data_with_momentum["kernel_momentum"].iloc[current_index]

        if pd.isna(prev_momentum) or pd.isna(curr_momentum):
            return False

        return bool(prev_momentum < self.threshold and curr_momentum >= self.threshold)