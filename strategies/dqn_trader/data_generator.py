#data_generator.py
import os
import pandas as pd
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
import random

# Get the project root directory (one level up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from base_models.angel_api import AngelAPI


class DataGenerator:
    """
    Handles fetching historical data, generating charts, and saving them.
    """

    def __init__(self, api: AngelAPI, instrument_symbol: str, filepath: str, interval="ONE_MINUTE", chart_candles=375):
        """
        Initializes the DataGenerator.

        Args:
            api (AngelAPI): An instance of the AngelAPI class for fetching data.
            instrument_symbol (str): The symbol of the instrument to fetch data for.
            filepath (str): The path to the token file.
            interval (str): the interval for the data.
        """
        self.api = api
        self.instrument_symbol = instrument_symbol
        self.interval = interval
        self.filepath = filepath
        self.data_dir = "strategies/RL/data"  # Directory to store data and images
        self.charts_dir = "strategies/RL/data/charts"
        self.chart_candles = chart_candles
        self.metadata_file = os.path.join(self.data_dir, "metadata.csv")

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        self._initialize_metadata_csv()

    def _initialize_metadata_csv(self):
        """
        Initializes the metadata CSV file if it doesn't exist.
        """
        if not os.path.exists(self.metadata_file):
            metadata_df = pd.DataFrame(
                columns=["image_path", "timestamp", "Open", "High", "Low", "Close", "Volume", "SMA", "Upper", "Lower", "Open_HA", "High_HA", "Low_HA", "Close_HA", "Open_R", "High_R", "Low_R", "Close_R"]
            )
            metadata_df.to_csv(self.metadata_file, index=False)

    def _process_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the historical data to ensure correct data types and basic validity.

        Args:
            df (pd.DataFrame): The raw historical data.

        Returns:
            pd.DataFrame: The processed historical data.
        """
        if df.empty:
            return df

        # Ensure data types
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        # Remove rows with NaN values after data type conversion
        df.dropna(inplace=True)

        return df

    def _calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculates Bollinger Bands.

        Args:
            data (pd.DataFrame): input dataframe.
            window (int): the lookback window.
            std (int): number of standard deviation.

        Returns:
            pd.DataFrame: dataframe with bollinger bands.
        """
        data["SMA"] = data["Close"].rolling(window=window).mean()
        data["std"] = data["Close"].rolling(window=window).std()
        data["Upper"] = data["SMA"] + (data["std"] * std)
        data["Lower"] = data["SMA"] - (data["std"] * std)

        for col in data.columns.tolist():
           data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)

        return data

    def _calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Heikin Ashi candles.

        Args:
            df (pd.DataFrame): input dataframe.

        Returns:
            pd.DataFrame: dataframe with heikin ashi candles.
        """
        df_ha = df.copy()
        # ensure that the values are correct type
        df_ha = df_ha.astype({'Open': float, 'High': float, 'Low': float, 'Close': float})

        # Calculate Heikin Ashi candles
        df_ha = df_ha.assign(
            Close_HA=lambda x: round((x['Open'] + x['High'] + x['Low'] + x['Close']) / 4, 2),
            Open_HA=lambda x: round((x['Open'].shift(1) + x['Close'].shift(1)) / 2, 2),
            High_HA=lambda x: x[['High', 'Open_HA', 'Close_HA']].max(axis=1).round(2),
            Low_HA=lambda x: x[['Low', 'Open_HA', 'Close_HA']].min(axis=1).round(2)
        )
        # Calculate the first value in Open_HA
        df_ha.loc[df_ha.index[0], "Open_HA"] = (df_ha.loc[df_ha.index[0], "Open"] + df_ha.loc[df_ha.index[0], "Close"]) / 2

        # Remove NaN values
        df_ha.dropna(inplace=True)

        print("df_ha columns", df_ha.columns.tolist())
        df_ha = df_ha[["Open_HA", "High_HA", "Low_HA", "Close_HA"]]
        
        # Rename columns to match mplfinance requirements
        df_ha = df_ha.rename(
            columns={"Open_HA": "Open", "High_HA": "High", "Low_HA": "Low", "Close_HA": "Close"}
        )
        
        print(df_ha)
        # Ensure all columns are float after calculation
        for col in ["Open", "High", "Low", "Close"]:
            df_ha[col] = pd.to_numeric(df_ha[col], errors="coerce").astype(float)

        return df_ha

    def _calculate_renko(self, df: pd.DataFrame, box_size: float = None) -> pd.DataFrame:
        """Calculates Renko bricks.

        Args:
            df (pd.DataFrame): input dataframe.
            box_size (float): the box size for the renko bricks.

        Returns:
            pd.DataFrame: dataframe with renko bricks.
        """
        # If box_size is not specified, calculate it as the average ATR over 14 periods.
        if box_size is None:
            df["High_Low_Range"] = df["High"] - df["Low"]
            df["Previous_Close_Range"] = abs(df["Close"].shift(1) - df["High"])
            df["Previous_Close_Range2"] = abs(df["Close"].shift(1) - df["Low"])
            df["TR"] = df[["High_Low_Range", "Previous_Close_Range", "Previous_Close_Range2"]].max(axis=1)
            df["ATR"] = df["TR"].rolling(window=14).mean()
            box_size = df["ATR"].iloc[-1]  # Use the last calculated ATR as the box size

        df_renko = pd.DataFrame(columns=["Open", "Close", "High", "Low"])
        
        # Store the indices as the same length as the number of rows of the renko data
        renko_indices = []

        last_close = df["Close"].iloc[0]
        brick = 0
        brick_open = last_close
        for i, row in df.iterrows():
            low = row["Low"]
            high = row["High"]
            if high >= last_close + box_size:
                bricks_up = int((high - last_close) // box_size)
                for _ in range(bricks_up):
                    brick_open = last_close
                    last_close += box_size
                    brick += 1
                    df_renko.loc[len(df_renko)] = [brick_open, last_close, last_close, last_close]
                    renko_indices.append(i) # add a new index each time we add a row to the df_renko dataframe.
            if low <= last_close - box_size:
                bricks_down = int((last_close - low) // box_size)
                for _ in range(bricks_down):
                    brick_open = last_close
                    last_close -= box_size
                    brick -= 1
                    df_renko.loc[len(df_renko)] = [brick_open, last_close, last_close, last_close]
                    renko_indices.append(i)# add a new index each time we add a row to the df_renko dataframe.

        for col in df_renko.columns.tolist():
           df_renko[col] = pd.to_numeric(df_renko[col], errors="coerce").astype(float)
        
        # Add a date to the renko data.
        df_renko["Date"] = renko_indices
        df_renko.drop_duplicates(inplace=True)
        
        # Set index to datetime objects
        df_renko.set_index("Date", inplace=True)
        df_renko.index = pd.to_datetime(df_renko.index)

        return df_renko

    def _plot_chart(self, data: pd.DataFrame, data_ha: pd.DataFrame, data_renko:pd.DataFrame, filename: str):
        """
        Plots a candlestick chart with Bollinger Bands and saves it as an image.

        Args:
            data (pd.DataFrame): Data with OHLCV and Bollinger Bands.
            filename (str): Output filename.
        """
        # Check if any rows exist to avoid errors with empty data
        if data.empty:
            print(f"Warning: Data is empty. Skipping chart creation for {filename}.")
            return
        
        # Create subplots
        fig = plt.figure(figsize=(10, 8))  # Increased figure height for volume subplot
        gs = fig.add_gridspec(4, 1, hspace=0, height_ratios=[3,1,1,1])  # create 4 subplots, adjusted height ratios
        axes = gs.subplots(sharex=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0) #remove padding
        
        # Add the volume data to the addplot list of the candlestick chart.
        volume = mpf.make_addplot(data["Volume"], type="bar", panel=3, ax=axes[3])

        apds = [
            mpf.make_addplot(data["Upper"], color="orange", linewidths=1, ax=axes[0]),
            mpf.make_addplot(data["SMA"], color="blue", linewidths=1, ax=axes[0]),
            mpf.make_addplot(data["Lower"], color="orange", linewidths=1, ax=axes[0]),
            volume
        ]
        
        # Plot candlestick chart
        try:
            mpf.plot(data, type="candle", style="yahoo", addplot=apds, warn_too_much_data=500, axtitle='candlestick', ax=axes[0], xrotation=0, show_nontrading=False, volume=False,mav=(20))
        except Exception as e:
            print(f"Exception plotting candlestick chart: {e}")

        # Plot Heikin Ashi chart
        try:
            mpf.plot(data_ha, type="candle", style="yahoo", warn_too_much_data=500, axtitle='Heikin Ashi', ax=axes[1], xrotation=0, show_nontrading=False, volume=False)
        except Exception as e:
            print(f"Exception plotting Heikin Ashi chart: {e}")

        # Plot Renko chart
        try:
            mpf.plot(data_renko, type="candle", style="yahoo", warn_too_much_data=500, axtitle='Renko', ax=axes[2], xrotation=0, show_nontrading=False, volume=False)
        except Exception as e:
            print(f"Exception plotting Renko chart: {e}")
        
            
        plt.savefig(os.path.join(self.charts_dir, filename + ".png"))
        plt.close(fig)
        plt.close("all")

    def generate_data_and_charts(self, days: int = 365, from_date: str = None, to_date: str = None):
        """
        Fetches historical data, generates candlestick, Heikin Ashi and renko charts, and saves them.

        Args:
            days (int): The number of days of historical data to fetch.
        """
        # try:
        # Get data from API for all days
        if days:
            df = self.api.download_historical_data(
                exchange="NSE", instrument_symbol=self.instrument_symbol, interval=self.interval, days=days, filepath=self.filepath
            )
        else:
            df = self.api.download_historical_data(
                exchange="NSE", instrument_symbol=self.instrument_symbol, interval=self.interval, from_date=from_date, to_date=to_date, filepath=self.filepath
            )
        print(f"data:\n{df.head()}")
        if df is None or df.empty:
            print("Error: Could not get historical data from API.")
            return

        df = self._process_historical_data(df)

        # Calculate Bollinger Bands
        df = self._calculate_bollinger_bands(df)

        # Calculate Heikin Ashi
        df_ha = self._calculate_heikin_ashi(df)
        
        # Calculate Renko
        df_renko = self._calculate_renko(df)
   
        metadata = []

        try:
            # Iterate over data in 1-day slices
            for i in range(len(df) - 1):
                start_index = i
                end_index = min(i + self.chart_candles, len(df))  # 375 data points
                if end_index - start_index > 0:
                    one_day_data = df.iloc[start_index:end_index]
                    one_day_data_ha = df_ha.iloc[start_index:end_index]
                    one_day_data_renko = df_renko.iloc[start_index:end_index]

                    # Check if any rows exist in one_day_data
                    if not one_day_data.empty:
                        timestamp = one_day_data.index[-1].strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{self.instrument_symbol}_{timestamp}"

                        # Create and save candlestick chart
                        self._plot_chart(one_day_data, one_day_data_ha, one_day_data_renko, filename)
                        
                        # Save OHLCV, indicators, and image name to CSV
                        last_candle_data = one_day_data.iloc[-1]
                        last_candle_data_ha = one_day_data_ha.iloc[-1]
                        last_candle_data_r = one_day_data_renko.iloc[-1]

                        metadata_row = {
                            "image_path": os.path.join(filename + ".png"),
                            "timestamp": timestamp,
                            "Open": round(last_candle_data["Open"], 2),
                            "High": round(last_candle_data["High"], 2),
                            "Low": round(last_candle_data["Low"], 2),
                            "Close": round(last_candle_data["Close"], 2),
                            "Volume": last_candle_data["Volume"],
                            "SMA": round(last_candle_data["SMA"], 2),
                            "Upper": round(last_candle_data["Upper"], 2),
                            "Lower": round(last_candle_data["Lower"], 2),
                            "Open_HA": round(last_candle_data_ha["Open"], 2),
                            "High_HA": round(last_candle_data_ha["High"], 2),
                            "Low_HA": round(last_candle_data_ha["Low"], 2),
                            "Close_HA": round(last_candle_data_ha["Close"], 2),
                            "Open_R": round(last_candle_data_r["Open"], 2),
                            "High_R": round(last_candle_data_r["High"], 2),
                            "Low_R": round(last_candle_data_r["Low"], 2),
                            "Close_R": round(last_candle_data_r["Close"], 2),
                        }
                        metadata.append(metadata_row)
        except Exception as e:
            print("Errot generating charts and data", e)

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.metadata_file, index=False)

        # except Exception as e:
        #     print(f"An error occurred while generating data and charts: {e}")

    def preprocess_data(self, current_balance=10000, transaction_cost_perc=0.001):
        """
            generate data for training and testing
            output df should have image pixel values, metadata column, target-q
        """
        # read metadata
        metadata = pd.read_csv(self.metadata_file, index=False)

        metadata['action'] = 0
        metadata['reward'] = 0
        current_position = 0
        entry_price = 0
        for i in range(metadata.shape[0]):
            last_balance = current_balance
            valid_action = False
            while not valid_action:
                action = random.randrange(3)
                metadata.iloc[i]['action'] = action
                if action == 1 and current_position == 0: #buy
                    metadata.iloc[i]['trade_price'] = metadata.iloc[i]['Close']
                    entry_price = metadata.iloc[i]['Close']
                    current_position = 1
                    current_balance -= metadata.iloc[i]['Close'] - metadata.iloc[i]['Close']*transaction_cost_perc
                    if current_balance < 0:
                        current_balance = last_balance
                    else:
                        valid_action = True
                        metadata.iloc[i]['reward'] = -metadata.iloc[i]['Close']*transaction_cost_perc
                elif action == 2 and current_position == 1: #sell
                    metadata.iloc[i]['trade_price'] = metadata.iloc[i]['Close']
                    current_position = 0
                    current_balance -= metadata.iloc[i]['Close']*transaction_cost_perc
                    current_balance += metadata.iloc[i]['Close']
                    # reward = metadata.iloc[i]['Close'] - metadata.iloc[i]['trade_price'] - metadata.iloc[i]['Close']*transaction_cost_perc
                    metadata.iloc[i]['reward'] = current_balance
                    valid_action = True
                elif action == 0 and current_position == 1:
                    metadata.iloc[i]['trade_price'] = entry_price
                    valid_action = True
                    reward = current_balance + metadata.iloc[i]['Close']
                    metadata.iloc[i]['reward'] = reward
                elif action == 0 and current_position == 0:
                    entry_price = 0
                    reward = current_balance
                    metadata.iloc[i]['reward'] = reward
                    valid_action = True
        
        metadata['target_q'] = metadata['target_q'] + 0.9*(metadata['reward'])
        metadata['next_state_q'] = metadata['target_q'].shift(-1)
        metadata['target_q'] = metadata['target_q'] + 0.9*(metadata['reward']+metadata['next_state_q']-metadata['target_q'])

        # assign random action, make sure there was buy action before sell action
        # hold = 0, buy = 1, sell = 2
      
        # calculate target-q


# Add main function
if __name__ == "__main__":
    # Initialize the AngelAPI
    api = AngelAPI()

    # Set instrument_symbol, and filepath
    instrument_symbol = "ICICIBANK-EQ"
    filepath = os.path.join(project_root, "base_models", "token_symbol_list.txt")

    # Initialize the DataGenerator
    data_generator = DataGenerator(api, instrument_symbol, filepath, interval="FIVE_MINUTE", chart_candles=75) # chart_candles=375/5

    # Generate data and charts for 365 days
    data_generator.generate_data_and_charts(days=None, from_date="2025-01-01 09:15", to_date="2025-02-01 15:20")


