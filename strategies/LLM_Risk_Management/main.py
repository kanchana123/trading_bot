import pandas as pd
import numpy as np
import pandas_ta as pta  # Import pandas_ta
from datetime import datetime, timedelta
import time
import json
import requests
from typing import Dict, List, Tuple
from model import get_llm_trading_advice_hf as get_llm_trading_advice
import os
import re # Import the regular expression module

import sys
sys.path.append("/Users/kanchannannavare/Documents/TradingBot")

from base_models.angel_api import AngelAPI  # Import AngelAPI

def parse_llm_response(llm_response: str) -> List[Dict]:
    """
    Parses the LLM's response string into a list of order dictionaries.

    Args:
        llm_response: The string response from the LLM.

    Returns:
        A list of dictionaries, where each dictionary represents a trading order.
        Returns an empty list if parsing fails.
    """
    if not llm_response:
        print("LLM response is empty.")
        return []

    print(llm_response)
    # Remove any leading/trailing whitespace or non-JSON characters
    llm_response = llm_response.strip()

    # Use regular expression to find the JSON array within the string
    match = re.search(r"(\[[\s\S]*?\])", llm_response) # Updated this line

    if match:
        json_string = match.group(1)
    else:
        print("No JSON array found in LLM response.")
        return []

    try:
        orders = json.loads(json_string)
        if not isinstance(orders, list):
            raise ValueError("LLM response should be a list of orders.")
        for order in orders:
            if not isinstance(order, dict):
                raise ValueError("Each element in the LLM response should be a dictionary.")
            if not all(
                key in order
                for key in [
                    "stock",
                    "transaction_type",
                    "buy/sell",
                    "entry_price",
                    "stop_loss",
                    "explanation",
                    "quantity",
                    "trend_confidence"
                ]
            ):
                raise ValueError(
                    "Each order should contain 'stock', 'transaction_type', 'buy/sell', 'entry_price', 'stop_loss', 'explanation', 'quantity', and 'trend_confidence'."
                )
            if order["entry_price"] is not None:
                order["entry_price"] = float(order["entry_price"])
            if order["stop_loss"] is not None:
                order["stop_loss"] = float(order["stop_loss"])
            if order["quantity"] is not None:
                order["quantity"] = int(order["quantity"])
            if order["trend_confidence"] is not None:
                order["trend_confidence"] = float(order["trend_confidence"])
        return orders
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        return []


def calculate_indicators(df):
    """
    Calculates technical indicators using pandas_ta and adds them to the DataFrame.

    Args:
        df: pandas DataFrame with OHLCV data.

    Returns:
        pandas DataFrame with added indicator columns.
    """
    # Bollinger Bands
    df.ta.bbands(close=df["Close"], length=20, std=2, append=True)

    # RSI
    df.ta.rsi(close=df["Close"], length=14, append=True)

    # MFI
    df.ta.mfi(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], length=14, append=True)
    df["MFI_14"] = df["MFI_14"].astype("float64") # Added this line

    # Coppock Curve
    df.ta.coppock(close=df["Close"], window_slow=14, window_fast=11, window_lookback=10, append=True)

    # TRIX
    df.ta.trix(close=df["Close"], length=14, append=True)

    # MACD
    df.ta.macd(close=df["Close"], fast=12, slow=26, signal=9, append=True)

    df.dropna(inplace=True)
    # df.reset_index(drop=True, inplace=True)
    print(df.columns.tolist())
    print(df.head())
    return df


def backtest(df, initial_capital=30000):
    """
    Backtests the trading strategy.

    Args:
        df: pandas DataFrame with OHLCV data and indicators.
        initial_capital: The initial capital for the portfolio.

    Returns:
        pandas DataFrame with order history.
    """
    portfolio = {"cash": initial_capital, "holdings": {}}
    current_positions = {}
    order_history = []
    history = 10
    previous_llm_response = ""

    for i in range(len(df)):
        current_data = df.iloc[i]

        # Prepare data for LLM
        current_portfolio = portfolio.copy()
        current_portfolio["holdings"] = current_positions.copy()
        current_news = ["No news available"]  # Replace with actual news if available
        
        current_ohlcv = {
            "ICICIBANK": {
                "datetime": df.index[max(0, i - history):i].tolist(),
                "open": df["Open"].iloc[max(0, i - history):i].tolist(),
                "high": df["High"].iloc[max(0, i - history):i].tolist(),
                "low": df["Low"].iloc[max(0, i - history):i].tolist(),
                "close": df["Close"].iloc[max(0, i - history):i].tolist(),
                "volume": df["Volume"].iloc[max(0, i - history):i].tolist(),
                "BBL_20_2.0": df["BBL_20_2.0"].iloc[max(0, i - history):i].tolist(),
                "BBM_20_2.0": df["BBM_20_2.0"].iloc[max(0, i - history):i].tolist(),
                "BBU_20_2.0": df["BBU_20_2.0"].iloc[max(0, i - history):i].tolist(),
                "BBB_20_2.0": df["BBB_20_2.0"].iloc[max(0, i - history):i].tolist(),
                "BBP_20_2.0": df["BBP_20_2.0"].iloc[max(0, i - history):i].tolist(),
                "RSI_14": df["RSI_14"].iloc[max(0, i - history):i].tolist(),
                "MFI_14": df["MFI_14"].iloc[max(0, i - history):i].tolist(),
                "COPC_11_14_10": df["COPC_11_14_10"].iloc[max(0, i - history):i].tolist(),
                "TRIX_14_9": df["TRIX_14_9"].iloc[max(0, i - history):i].tolist(),
                "TRIXs_14_9": df["TRIXs_14_9"].iloc[max(0, i - history):i].tolist(),
                "MACD_12_26_9": df["MACD_12_26_9"].iloc[max(0, i - history):i].tolist(),
                "MACDh_12_26_9": df["MACDh_12_26_9"].iloc[max(0, i - history):i].tolist(),
                "MACDs_12_26_9": df["MACDs_12_26_9"].iloc[max(0, i - history):i].tolist()
            }
        }

        # Get LLM advice
        llm_response = get_llm_trading_advice(current_portfolio, current_positions, current_news, current_ohlcv)
        orders = parse_llm_response(llm_response)

        # Process orders
        for order in orders:
            if order["stock"] == "ICICIBANK":
                current_price = current_data["Close"]
                
                if "ICICIBANK" in current_positions:
                    current_quantity = current_positions["ICICIBANK"]["quantity"]
                    current_entry_price = current_positions["ICICIBANK"]["avg_price"]
                else:
                    current_quantity = 0
                    current_entry_price = 0
                
                order_type = order["buy/sell"]
                entry_price = order["entry_price"]
                stop_loss = order["stop_loss"]
                quantity = order["quantity"]
                order["explanation"] = "Confidence: " + str(order["trend_confidence"]) + " | " + order["explanation"]

                if entry_price is None: # Added this line
                    print(f"Skipping order due to missing entry_price: {order}")
                    continue

                if order_type == "buy":
                    if portfolio["cash"] >= quantity * entry_price:
                        portfolio["cash"] -= quantity * entry_price
                        if "ICICIBANK" in current_positions:
                            current_positions["ICICIBANK"] = {
                                "quantity": current_quantity + quantity,
                                "avg_price": (current_entry_price*current_quantity + quantity*entry_price)/(current_quantity + quantity),
                                "stop_loss": stop_loss
                            }
                        else:
                            current_positions["ICICIBANK"] = {
                                "quantity": quantity,
                                "avg_price": entry_price,
                                "stop_loss": stop_loss
                            }
                        order_history.append({
                            "Date": df.index[i],
                            "Stock": "ICICIBANK",
                            "Portfolio Value": portfolio["cash"] + (current_positions.get("ICICIBANK", {}).get("quantity", 0) * current_price),
                            "Type": "buy",
                            "Entry Price": entry_price,
                            "Stop Loss": stop_loss,
                            "Quantity": quantity,
                            "Explanation": order["explanation"]
                        })
                    else:
                        print(f"Insufficient cash to buy {quantity} shares of ICICIBANK at {entry_price}")

                elif order_type == "sell":
                    if "ICICIBANK" in current_positions:
                        if quantity <= current_positions["ICICIBANK"]["quantity"]:
                            portfolio["cash"] += quantity * entry_price
                            current_positions["ICICIBANK"]["quantity"] -= quantity
                            order_history.append({
                                "Date": df.index[i],
                                "Stock": "ICICIBANK",
                                "Portfolio Value": portfolio["cash"] + (current_positions.get("ICICIBANK", {}).get("quantity", 0) * current_price),
                                "Type": "sell",
                                "Entry Price": entry_price,
                                "Stop Loss": stop_loss,
                                "Quantity": quantity,
                                "Explanation": order["explanation"]
                            })
                            if current_positions["ICICIBANK"]["quantity"] == 0:
                                del current_positions["ICICIBANK"]
                        else:
                            print(f"Insufficient quantity to sell. Trying to sell {quantity} but only have {current_positions['ICICIBANK']['quantity']}")
                    else:
                        print("No long position in ICICIBANK to sell.")

                # Check stop loss
                if "ICICIBANK" in current_positions: # Added this line
                    if current_positions["ICICIBANK"]["stop_loss"] and current_price <= current_positions["ICICIBANK"]["stop_loss"]:
                        quantity = current_positions["ICICIBANK"]["quantity"]
                        portfolio["cash"] += quantity * current_price
                        order_history.append({
                            "Date": df.index[i],
                            "Stock": "ICICIBANK",
                            "Portfolio Value": portfolio["cash"] + (current_positions.get("ICICIBANK", {}).get("quantity", 0) * current_price),
                            "Type": "stop_loss_sell",
                            "Entry Price": current_price,
                            "Stop Loss": current_positions["ICICIBANK"]["stop_loss"],
                            "Quantity": quantity,
                            "Explanation": "Stop loss triggered"
                        })
                        del current_positions["ICICIBANK"]

        orders_df = pd.DataFrame(order_history)
        orders_df.to_csv("order_history.csv")
        

    return orders_df

def main():
    # Initialize AngelAPI
    angel_api = AngelAPI()

    # Get historical data
    instrument_symbol = "ICICIBANK-EQ"  # Instrument symbol for ICICIBANK
    interval = "FIVE_MINUTE"
    days = 30
    filepath = f'{os.path.abspath(".")}/base_models/token_symbol_list.txt'
    df = angel_api.download_historical_data("NSE", instrument_symbol, interval=interval, days=days, filepath=filepath)
    if df is None:
        print("Failed to get historical data.")
        return

    # Calculate indicators
    df = calculate_indicators(df)

    # Backtest
    order_history_df = backtest(df)

    # Save order history to CSV
    order_history_df.to_csv("order_history.csv", index=False)
    print("Order history saved to order_history.csv")


if __name__ == "__main__":
    main()
