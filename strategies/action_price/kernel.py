import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import yfinance as yf # Removed yfinance
from scipy.signal import convolve
import os
import sys

# Add project root to sys.path to import AngelAPI
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from base_models.angel_api import AngelAPI # Import AngelAPI

# --- 1. Load data using AngelAPI ---
ticker = 'ICICIBANK-EQ'
exchange = "NSE"
interval = "ONE_MINUTE" # Changed interval to match Angel API options
days = 10 # Get last 100 days of data

try:
    angel_api = AngelAPI()
    ohlc = angel_api.download_historical_data(
        exchange=exchange,
        instrument_symbol=ticker,
        interval=interval,
        days=days
    )
    # ohlc = ohlc.head(375)
except Exception as e:
    print(f"Error fetching data using AngelAPI: {e}")
    sys.exit(1) # Exit if data fetching fails

if ohlc is None or ohlc.empty:
    print(f"No data received for {ticker}. Exiting.")
    sys.exit(1)

print(f"Data for {ticker} loaded successfully:")
print(ohlc.head())

ohlc.reset_index(drop=False, inplace=True)
ohlc.rename(columns={'Date': 'date'}, inplace=True)
ohlc.set_index('date', inplace=True)

df = ohlc.copy()
# 2. Momentum kernel
# Note: The effectiveness of this kernel might depend on the data interval
kernel = np.array([-2, -1, 0, 1, 2])

# 3. Compute signal (Close - Midpoint)
mid_price = (df['High'] + df['Low']) / 2
signal = (df['Close'] - mid_price).values
momentum = convolve(signal, kernel, mode='same')

# 4. Define trading logic
threshold = 1.0  # sensitivity threshold - you might need to adjust this based on ICICIBANK's price scale and volatility
orders = []
position = 0  # 0: no position, 1: long

df.reset_index(drop=False, inplace=True)


for i in range(1, len(momentum)):
    date = df.index[i]
    price = df['Close'].iloc[i]
    
    # Buy signal
    if momentum[i-1] < threshold and momentum[i] >= threshold and position == 1: # Sell if momentum crosses above threshold while long
        buy_price = orders[-1][2] if orders else price
        orders.append((date, "SELL", price,  price - buy_price))
        position = 0
        
    # Sell signal
    elif momentum[i-1] > -threshold and momentum[i] <= -threshold and position == 0: # Buy if momentum crosses below -threshold while flat
        orders.append((date, "BUY", price))
        position = 1

# 5. Print trade log
profit = 0
print("\n📝 Trade Log:")
for order in orders:
    print(f"{order[0]} | {order[1]:<4} | Price: ${order[2]:.2f}")
    profit += order[3] if len(order) > 3 else 0

print("Profit: ", round(profit, 2), len(orders)/2, "trades")
# df.reset_index(drop=False, inplace=True)
# 6. Plot
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price', color='blue')
# plt.plot(df.index, momentum, label='Momentum Signal', color='red')

# Mark trades
for order in orders:
    color = 'green' if order[1] == 'BUY' else 'black'
    marker = '^' if order[1] == 'BUY' else 'v'
    plt.plot(order[0], order[2], marker=marker, color=color, markersize=10)

# --- Adjust Y-axis limits ---
price_min = df['Close'].min()
price_max = df['Close'].max()
y_buffer = (price_max - price_min) * 0.1 # Add a 10% buffer
plt.ylim(price_min - y_buffer, price_max + y_buffer)
# --- End Y-axis adjustment ---


plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.5)
plt.title(f"{ticker} - Momentum Breakout Strategy")
plt.legend()
plt.grid(True)
# plt.tight_layout()
plt.show()
