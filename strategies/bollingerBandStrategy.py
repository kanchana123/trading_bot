# in bollingerBandStrategy.py

from base_models.strategy import Strategy
import pandas as pd
import numpy as np

class BollingerBandStrategy(Strategy):
    """
    A simple trading strategy based on Bollinger Bands.
    """

    def __init__(self, name: str = "Bollinger Band Strategy", params: dict = None):
        """
        Initializes the BollingerBandStrategy.

        Args:
            name (str, optional): The name of the strategy. Defaults to "Bollinger Band Strategy".
            params (dict, optional): Parameters for the strategy (window, std_multiplier). Defaults to None.
        """
        super().__init__(name, params)
        self.window = self.params.get("window", 20)  # Default window of 20
        self.std_multiplier = self.params.get("std_multiplier", 2)  # Default multiplier of 2
        self.desc = self.generate_desc()

    def generate_desc(self):
        """
        Generates a description of the strategy based on its name and parameters.

        Returns:
            str: A descriptive string of the strategy.
        """
        desc = f"{self.name} with parameters: "
        desc += f"Window = {self.window}, "
        desc += f"Standard Deviation Multiplier = {self.std_multiplier}. "
        desc += "This strategy generates a buy signal when the price crosses below the lower Bollinger Band and a sell signal when the price crosses above the upper Bollinger Band."
        return desc

    def process_data(self):
        """
        Calculates the Bollinger Bands (middle band, upper band, lower band).
        """
        if self.data is None or len(self.data) < self.window:
            return

        # Calculate the middle band (SMA)
        self.data['middle_band'] = self.data['Close'].rolling(window=self.window).mean()

        # Calculate the standard deviation
        std = self.data['Close'].rolling(window=self.window).std()

        # Calculate the upper and lower bands
        self.data['upper_band'] = self.data['middle_band'] + self.std_multiplier * std
        self.data['lower_band'] = self.data['middle_band'] - self.std_multiplier * std
        

    def should_buy(self, current_index: int) -> bool:
        """
        Generates a buy signal when the price crosses below the lower Bollinger Band.
        Args:
            current_index(int): index of the current data.
        Returns:
            bool: True if a buy signal is generated, False otherwise.
        """
        if self.data is None or current_index < self.window:
            return False

        current_data = self.data.iloc[current_index]
        previous_data = self.data.iloc[current_index -1]

        return bool(
            previous_data['Close'] <= previous_data['lower_band']
            and current_data['Close'] > current_data['lower_band']
        )

    def should_sell(self, current_index: int) -> bool:
        """
        Generates a sell signal when the price crosses above the upper Bollinger Band.
        Args:
            current_index(int): index of the current data.
        Returns:
            bool: True if a sell signal is generated, False otherwise.
        """
        if self.data is None or current_index < self.window:
            return False
        
        current_data = self.data.iloc[current_index]
        previous_data = self.data.iloc[current_index -1]

        return bool(
            previous_data['Close'] >= previous_data['upper_band']
            and current_data['Close'] < current_data['upper_band']
        )

    def select_quantity(self, price: float, portfolio_value: float) -> int:
        """
        Selects the quantity to buy or sell.

        Args:
            price (float): The current price.
            portfolio_value(float): current portfolio value

        Returns:
            int: The quantity to buy or sell.
        """
        quantity = int(portfolio_value * 0.2 / price) # buy only 20% of portfolio value for the first time.
        if quantity == 0:
            return 1
        return quantity

# Example Usage:
if __name__ == "__main__":
    # Create sample data
    data = {
        'Close': [10, 12, 15, 14, 16, 18, 20, 19, 22, 25, 23, 26, 28, 27, 30, 29, 32, 35, 33, 36, 38, 37, 40, 39, 42]
    }
    df = pd.DataFrame(data)

    # Create and configure the strategy
    strategy = BollingerBandStrategy(params={"window": 5, "std_multiplier": 2})
    strategy.set_data(df)
    strategy.process_data()

    print(strategy.data)
    print(strategy.select_quantity(10, 1000))

    # Test buy and sell signals
    for i in range(len(df)):
        if strategy.should_buy(i):
            print(f"Buy signal at index {i}, price: {df['Close'][i]}")
        elif strategy.should_sell(i):
            print(f"Sell signal at index {i}, price: {df['Close'][i]}")
        elif strategy.should_hold(i):
            print(f"Hold signal at index {i}, price: {df['Close'][i]}")
