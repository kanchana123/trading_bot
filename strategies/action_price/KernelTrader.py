import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os
import sys
import logging

# Add project root to sys.path to import AngelAPI
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from base_models.strategy import Strategy as BacktestStrategy # Alias for clarity
from base_models.angel_api import AngelAPI # Keep for backtesting part
from base_models.backtest import Backtest # Keep for backtesting part
from strategies.base_strategy_rt import RealTimeStrategy # Import new RT base
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta

# Use the global logger configured in main.py or logger_setup.py
# No need for basicConfig here if main.py sets it up.
logger = logging.getLogger(__name__)
class KernelStrategy(RealTimeStrategy): # Inherit from RealTimeStrategy
    """
    Implements the Kernel Momentum Breakout strategy.
    """
    def __init__(self, name: str = "KernelMomentum", params: dict = None):
        super().__init__(name, params)
        logger.info(f"Initializing KernelStrategy. name='{self.name}', params='{self.params}'")

        # Default parameters
        self.kernel = np.array(self.params.get("kernel", [-2, -1, 0, 1, 2]))
        self.threshold = self.params.get("threshold", 1.5) # Adjust based on price scale/volatility
        self.fixed_quantity = self.params.get("quantity", 1) # Simple fixed quantity for now
        self.window = len(self.kernel) # Window needed for convolution
        self.bar_interval_minutes = self.params.get("bar_interval_minutes", 1) # Interval for aggregating ticks
        self.desc = self.generate_desc()

        # Real-time specific state
        self.current_bar_ticks: List[Dict] = [] # Ticks for the current forming bar
        self.current_bar_start_time: Optional[datetime] = None
        self.historical_bars = pd.DataFrame(columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'momentum'])
        self.max_historical_bars = self.params.get("max_historical_bars", 100) # Keep N recent bars for momentum calc

        # Position tracking (simple, per instance) - could be more sophisticated
        self.current_position = 0 # 0: flat, >0: long qty, <0: short qty (if supported)
        self._last_signal_bar_time = None

    def generate_desc(self):
        return (f"Kernel Momentum Strategy with kernel={list(self.kernel)}, "
                f"threshold={self.threshold}, quantity={self.fixed_quantity}, "
                f"bar_interval={self.bar_interval_minutes}min")

    def _process_historical_bars(self):
        """
        Calculates the momentum signal using the kernel on self.historical_bars.
        This is similar to the original process_data but adapted for the RT historical_bars.
        """
        if self.historical_bars.empty or len(self.historical_bars) < self.window:
            logger.debug(f"Skipping momentum calculation: Not enough historical bars. count={len(self.historical_bars)}, required_window={self.window}, token='{self.token_info.get('symbol', 'N/A') if self.token_info else 'N/A'}'")
            return

        # Ensure required columns exist
        required_cols = ['High', 'Low', 'Close']
        if not all(col in self.historical_bars.columns for col in required_cols): # Corrected from self.data to self.historical_bars
            logger.error(f"Historical bars missing required columns for momentum calculation. required_cols='{required_cols}', available_cols='{list(self.historical_bars.columns)}', token='{self.token_info.get('symbol', 'N/A') if self.token_info else 'N/A'}'")
            return
        
        # Compute signal (Close - Midpoint)
        mid_price = (self.historical_bars['High'] + self.historical_bars['Low']) / 2
        signal_values = (self.historical_bars['Close'] - mid_price).values

        # Calculate momentum using convolution
        if len(signal_values) < len(self.kernel):
            logger.debug(f"Signal length too short for convolution. signal_len={len(signal_values)}, kernel_len={len(self.kernel)}, token='{self.token_info.get('symbol', 'N/A') if self.token_info else 'N/A'}'")
            return
            
        momentum = convolve(signal_values, self.kernel, mode='valid')

        # Causal alignment: pad only at the start so bar i does not use future bars.
        padding = len(self.historical_bars) - len(momentum)
        padding = max(0, padding)
        padded_momentum = np.pad(momentum, (padding, 0), 'constant', constant_values=np.nan)

        self.historical_bars['momentum'] = padded_momentum
        last_momentum_val = self.historical_bars['momentum'].iloc[-1] if not self.historical_bars.empty and 'momentum' in self.historical_bars.columns and not pd.isna(self.historical_bars['momentum'].iloc[-1]) else 'N/A'
        logger.debug(f"Momentum calculated and added to historical_bars. token='{self.token_info.get('symbol', 'N/A') if self.token_info else 'N/A'}', last_momentum={last_momentum_val}, bars_count={len(self.historical_bars)}")


    def on_new_tick(self, tick_data: Dict, token_details: Dict) -> Optional[Dict[str, Any]]:
        """
        Process a new tick: aggregate into bars, calculate momentum, generate signals.
        tick_data: {'tk': '...', 'ltp': '...', 'tt': 'timestamp_ms', ...}
                   or {'token': '...', 'last_traded_price': '...', 'exchange_timestamp': '...'}
        """
        logger.debug(f"Processing new tick. token_symbol='{token_details.get('symbol', 'N/A')}', tick_data='{tick_data}'")
        
        # Extract price and timestamp
        # Angel API tick structure varies. 'ltp' or 'last_traded_price'.
        # Timestamp might be 'tt' (milliseconds), 'exchange_timestamp', or 'feed_timestamp'.
        price_str = tick_data.get('ltp') or tick_data.get('last_traded_price')
        if price_str is None:
            logger.warning(f"Tick missing price field. token_symbol='{token_details.get('symbol', 'N/A')}', tick_data='{tick_data}'")
            return None
        try:
            price = float(price_str)
        except ValueError:
            logger.warning(f"Could not parse price from tick. price_str='{price_str}', token_symbol='{token_details.get('symbol', 'N/A')}'", exc_info=True)
            return None

        # Timestamp handling (prefer exchange_timestamp if available)
        ts_value = tick_data.get('exchange_timestamp') or tick_data.get('feed_timestamp') or tick_data.get('ft')
        tick_time = datetime.now()
        if ts_value is None and 'tt' in tick_data:
            try:
                ts_value = int(tick_data['tt']) / 1000
            except (TypeError, ValueError):
                logger.warning(f"Could not parse timestamp 'tt' from tick. tt_value='{tick_data['tt']}', token_symbol='{token_details.get('symbol', 'N/A')}'", exc_info=True)
                ts_value = None
        if isinstance(ts_value, str):
            try:
                tick_time = pd.to_datetime(ts_value).to_pydatetime()
            except Exception:
                tick_time = datetime.now()
        elif isinstance(ts_value, (int, float)):
            tick_time = datetime.fromtimestamp(ts_value)

        # Initialize current bar if first tick or new bar interval starts
        if not self.current_bar_start_time or \
           tick_time >= self.current_bar_start_time + timedelta(minutes=self.bar_interval_minutes):
            
            # Finalize previous bar if it exists
            if self.current_bar_ticks:
                open_price = self.current_bar_ticks[0]['price']
                high_price = max(t['price'] for t in self.current_bar_ticks)
                low_price = min(t['price'] for t in self.current_bar_ticks)
                close_price = self.current_bar_ticks[-1]['price']
                # Volume might not be in LTP ticks, default to 0 or sum if available
                volume = sum(t.get('v', {}).get('volume', 0) for t in self.current_bar_ticks) # Example if volume is in tick

                new_bar = pd.DataFrame([{
                    'timestamp': self.current_bar_start_time,
                    'Open': open_price, 'High': high_price, 'Low': low_price, 'Close': close_price, 'Volume': volume
                }])
                logger.debug(f"Finalized bar. token_symbol='{token_details.get('symbol', 'N/A')}', bar_time='{self.current_bar_start_time}', O={open_price}, H={high_price}, L={low_price}, C={close_price}, V={volume}")

                self.historical_bars = pd.concat([self.historical_bars, new_bar], ignore_index=True)
                
                # Keep only recent bars
                if len(self.historical_bars) > self.max_historical_bars:
                    self.historical_bars = self.historical_bars.iloc[-self.max_historical_bars:]
                
                self._process_historical_bars() # Recalculate momentum

            # Start new bar
            self.current_bar_start_time = tick_time.replace(second=0, microsecond=0) # Align to minute start
            # If bar_interval_minutes > 1, align to the start of the interval
            minute_offset = self.current_bar_start_time.minute % self.bar_interval_minutes
            self.current_bar_start_time -= timedelta(minutes=minute_offset)
            
            logger.debug(f"Starting new bar. token_symbol='{token_details.get('symbol', 'N/A')}', new_bar_start_time='{self.current_bar_start_time}'")
            self.current_bar_ticks = []

        # Add current tick to the forming bar
        self.current_bar_ticks.append({'price': price, 'time': tick_time, 'raw_tick': tick_data})

        # --- Signal Generation (based on the last COMPLETED bar's momentum) ---
        if self.historical_bars.empty or pd.isna(self.historical_bars['momentum'].iloc[-1]):
            logger.debug(f"No signal: Historical bars empty or last momentum is NaN. token_symbol='{token_details.get('symbol', 'N/A')}'")
            return None # Not enough data or momentum not calculated yet

        # Need at least two momentum values to check for crossover
        if len(self.historical_bars) < 2 or pd.isna(self.historical_bars['momentum'].iloc[-2]):
            logger.debug(f"No signal: Not enough historical bars for momentum comparison (need 2). count={len(self.historical_bars)}, token_symbol='{token_details.get('symbol', 'N/A')}'")
            return None

        prev_momentum = self.historical_bars['momentum'].iloc[-2]
        curr_momentum = self.historical_bars['momentum'].iloc[-1]
        current_close_price = self.historical_bars['Close'].iloc[-1]
        bar_time = self.historical_bars['timestamp'].iloc[-1] if 'timestamp' in self.historical_bars.columns else None
        if bar_time is not None and self._last_signal_bar_time == bar_time:
            return None

        signal = None
        # Buy signal: momentum crosses below negative threshold
        if prev_momentum > -self.threshold and curr_momentum <= -self.threshold:
            if self.current_position <= 0:
                signal = {'action': 'buy', 'quantity': self.fixed_quantity, 'price': current_close_price, 'order_type': 'LIMIT'}
                logger.info(f"BUY signal generated. token_symbol='{token_details.get('symbol', 'N/A')}', prev_momentum={prev_momentum:.4f}, curr_momentum={curr_momentum:.4f}, threshold={-self.threshold}, price={current_close_price}, quantity={self.fixed_quantity}, current_pos_before_trade={self.current_position}")

        # Sell signal: momentum crosses above positive threshold
        elif prev_momentum < self.threshold and curr_momentum >= self.threshold:
            if self.current_position > 0:
                signal = {'action': 'sell', 'quantity': self.fixed_quantity, 'price': current_close_price, 'order_type': 'LIMIT'}
                logger.info(f"SELL signal generated. token_symbol='{token_details.get('symbol', 'N/A')}', prev_momentum={prev_momentum:.4f}, curr_momentum={curr_momentum:.4f}, threshold={self.threshold}, price={current_close_price}, quantity={self.fixed_quantity}, current_pos_before_trade={self.current_position}")

        if signal is not None:
            self._last_signal_bar_time = bar_time
        return signal

    def confirm_fill(self, trade_signal: Dict[str, Any]):
        """Update local position only after the executor reports a successful fill."""
        quantity = int(trade_signal.get('quantity') or 0)
        action = (trade_signal.get('action') or '').lower()
        if action == 'buy':
            self.current_position += quantity
        elif action == 'sell':
            self.current_position -= quantity

    # --- Backtesting related methods (can be kept for separate backtesting if needed) ---
    # These methods would use self.data (a DataFrame passed during backtest)
    # instead of self.historical_bars.

    def process_data_for_backtest(self, data: pd.DataFrame):
        """Calculates momentum for backtesting. self.data should be set."""
        if data is None or data.empty:
            logger.error("Backtest: No data provided for momentum calculation.")
            return data
        
        temp_data = data.copy()
        required_cols = ['High', 'Low', 'Close']
        if not all(col in temp_data.columns for col in required_cols):
            logger.error(f"Backtest data missing required columns. required='{required_cols}', available='{list(temp_data.columns)}'")
            return data

        mid_price = (temp_data['High'] + temp_data['Low']) / 2
        signal_values = (temp_data['Close'] - mid_price).values
        
        if len(signal_values) < len(self.kernel):
            temp_data['momentum'] = np.nan
            logger.debug(f"Backtest: Signal length too short for convolution. signal_len={len(signal_values)}, kernel_len={len(self.kernel)}")
            return temp_data

        momentum = convolve(signal_values, self.kernel, mode='valid')
        padding = len(temp_data) - len(momentum)
        padding = max(0, padding)
        padded_momentum = np.pad(momentum, (padding, 0), 'constant', constant_values=np.nan)
        temp_data['momentum'] = padded_momentum
        logger.debug("Backtest: Momentum calculated for historical data.")
        return temp_data

    def should_buy_backtest(self, data: pd.DataFrame, current_index: int) -> bool:
        """
        Buy if momentum crosses below negative threshold.
        """
        if current_index < 1 or 'momentum' not in data.columns:
            return False
        # Ensure indices are valid and data exists
        if current_index >= len(data) or pd.isna(data['momentum'].iloc[current_index]) or pd.isna(data['momentum'].iloc[current_index - 1]):
             return False

        prev_momentum = data['momentum'].iloc[current_index - 1]
        curr_momentum = data['momentum'].iloc[current_index]

        return bool(prev_momentum > -self.threshold and curr_momentum <= -self.threshold)

    def should_sell_backtest(self, data: pd.DataFrame, current_index: int) -> bool:
        """
        Sell if momentum crosses above positive threshold.
        """
        if current_index < 1 or 'momentum' not in data.columns:
            return False
        # Ensure indices are valid and data exists
        if current_index >= len(data) or pd.isna(data['momentum'].iloc[current_index]) or pd.isna(data['momentum'].iloc[current_index - 1]):
             return False

        prev_momentum = data['momentum'].iloc[current_index - 1]
        curr_momentum = data['momentum'].iloc[current_index]

        return bool(prev_momentum < self.threshold and curr_momentum >= self.threshold)

    def select_quantity(self, price: float, portfolio_value: float) -> int:
        """
        Selects a fixed quantity to trade.
        """
        # Simple fixed quantity logic
        return self.fixed_quantity

# --- Adapter for Backtesting Framework ---
# This class can be used if you want to use the same KernelStrategy logic
# with your existing Backtest class that expects a BacktestStrategy interface.
class KernelBacktestAdapter(BacktestStrategy):
    def __init__(self, name: str = "KernelMomentum", params: dict = None):
        super().__init__(name, params) # Calls BacktestStrategy.__init__
        logger.info(f"Initializing KernelBacktestAdapter. name='{name}', params='{params}'")
        self.rt_strategy = KernelStrategy(name, params) # Contains the actual logic
        self.desc = self.rt_strategy.generate_desc()
        self.window = self.rt_strategy.window

    def process_data(self): # Conforms to BacktestStrategy
        self.data = self.rt_strategy.process_data_for_backtest(self.data)

    def should_buy(self, current_index: int) -> bool: # Conforms to BacktestStrategy
        return self.rt_strategy.should_buy_backtest(self.data, current_index)

    def should_sell(self, current_index: int) -> bool: # Conforms to BacktestStrategy
        return self.rt_strategy.should_sell_backtest(self.data, current_index)

    def select_quantity(self, price: float, portfolio_value: float) -> int: # Conforms to BacktestStrategy
        return self.rt_strategy.select_quantity(price, portfolio_value)

# --- Main Execution Block for Backtesting ---
if __name__ == "__main__":
    # --- Configuration ---
    ticker = 'RELIANCE-EQ'
    exchange = "NSE"
    interval = "ONE_MINUTE" # Use minute data for this strategy type
    start_date_str = "2024-01-01" # Example start date
    end_date_str = "2024-01-10"   # Example end date
    initial_portfolio = 100000.0
    strategy_params = {"threshold": 2.0, "quantity": 50} # Example params

    # --- 1. Load Data ---
    try:
        logger.info("Backtest Script: Initializing AngelAPI...")
        angel_api = AngelAPI()
        logger.info(f"Backtest Script: Fetching data for ticker='{ticker}', start='{start_date_str}', end='{end_date_str}'...")
        # Note: AngelAPI's date format might need HH:MM, adjust if necessary
        from_date_api = f"{start_date_str} 09:15"
        to_date_api = f"{end_date_str} 15:30"
        ohlc_data = angel_api.download_historical_data(
            exchange=exchange,
            instrument_symbol=ticker,
            interval=interval,
            from_date=from_date_api,
            to_date=to_date_api
        )
    except Exception as e:
        logger.error(f"Backtest Script: Error fetching data using AngelAPI. error='{e}'", exc_info=True)
        sys.exit(1)

    if ohlc_data is None or ohlc_data.empty:
        logger.error(f"Backtest Script: No data received for ticker='{ticker}' between {start_date_str} and {end_date_str}. Exiting.")
        sys.exit(1)

    logger.info(f"Backtest Script: Data for ticker='{ticker}' loaded successfully. rows={len(ohlc_data)}.")
    # Ensure index is datetime
    ohlc_data.index = pd.to_datetime(ohlc_data.index)

    print(ohlc_data.head())
    # --- 2. Initialize Strategy and Backtest ---
    # Use the adapter for the old backtesting framework
    kernel_strategy_adapter = KernelBacktestAdapter(params=strategy_params)
    
    backtester = Backtest(
        strategy=kernel_strategy_adapter, # Use the adapter
        stock=ticker,
        start_date=start_date_str,
        end_date=end_date_str,
        data=ohlc_data, # Pass the fetched data
        initial_portfolio_value=initial_portfolio
    )

    # --- 3. Run Backtest ---
    orders = backtester.run()

    # --- 4. Display Results ---
    logger.info("\n--- Backtest Script Results ---")
    if not orders:
        logger.info("Backtest Script: No trades were executed.")
    else:
        profit = 0
        buy_cost = 0
        position = 0
        print("\n📝 Trade Log:")
        for order in orders:
            print(order)
            if order.transaction_type == 'buy':
                buy_cost += order.price * order.quantity
                position += order.quantity
            elif order.transaction_type == 'sell':
                # Calculate profit for this sell trade based on average buy cost if needed
                # Simple calculation: proceeds - cost basis (needs tracking cost basis)
                # For simplicity, let's track total P/L based on portfolio value change
                pass # Profit is implicitly calculated in portfolio value

        final_value = backtester.portfolio_value_history[-1]
        total_profit = final_value - initial_portfolio
        num_trades = len(orders) // 2 # Approx number of round trips

        print(f"\nInitial Portfolio Value: ${initial_portfolio:.2f}")
        print(f"Final Portfolio Value:   ${final_value:.2f}")
        print(f"Total Profit/Loss:       ${total_profit:.2f}")
        print(f"Number of Trades:        {len(orders)} ({num_trades} round trips approx.)")

        print(backtester.data.head()) # Display data with momentum

        # --- 5. Plotting ---
        plt.figure(figsize=(14, 8))

        # Plot Close Price
        plt.plot(backtester.data.index, backtester.data['Close'], label='Close Price', color='gray', alpha=0.7)

        # Plot Momentum (on secondary axis if scale differs significantly)
        # ax2 = plt.twinx()
        # ax2.plot(backtester.data.index, backtester.data['momentum'], label='Momentum', color='orange', alpha=0.5)
        # ax2.set_ylabel('Momentum')
        # ax2.legend(loc='upper right')

        # Mark trades on the primary axis (ensure 'timestamp' column exists in backtester.data if used like this)
        buy_markers = [(backtester.data[backtester.data["timestamp"] == o.timestamp].index[0], o.price) for o in orders if o.transaction_type == 'buy']
        sell_markers = [(backtester.data[backtester.data["timestamp"] == o.timestamp].index[0], o.price) for o in orders if o.transaction_type == 'sell']

        if buy_markers:
            buy_times, buy_prices = zip(*buy_markers)
            plt.plot(buy_times, buy_prices, '^', color='green', markersize=10, label='Buy')
        if sell_markers:
            sell_times, sell_prices = zip(*sell_markers)
            plt.plot(sell_times, sell_prices, 'v', color='black', markersize=10, label='Sell')

        plt.title(f"{ticker} - {kernel_strategy_adapter.name} Backtest")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
