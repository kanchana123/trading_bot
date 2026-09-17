# Note: This script requires yfinance. Please install it using:
# pip install yfinance

import os
import sys
import pandas as pd
import pandas_ta as ta
import argparse
from datetime import datetime
import plotly.graph_objects as go
import logging

# Add project root to sys.path to import AngelAPI
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from llm_handler import LLMHandler
from news_analyzer import NewsAnalyzer
from technical_analyzer import TechnicalAnalyzer
from kernel_signal_generator import KernelSignalGenerator
from strategy_sandbox import compile_strategy_code
from base_models.angel_api import AngelAPI

# --- Setup Logging ---
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "low_risk_llm_backtest.log")
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'), # 'w' to overwrite the file on each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

TAKE_PROFIT_PERCENT = 0.5  # e.g., 0.5% profit target
STOP_LOSS_PERCENT = 0.3    # e.g., 0.3% stop loss
MIN_PROFIT_FOR_LLM_SELL_PERCENT = 0.1 # Min profit required before an LLM sell signal is considered

def plot_backtest_results(stock_data: pd.DataFrame, orders_df: pd.DataFrame, stock_symbol: str):
    """
    Generates an interactive candlestick chart with buy/sell markers and saves it as an HTML file.
    """
    if stock_data.empty:
        logger.warning("Cannot plot results: stock data is empty.")
        return

    fig = go.Figure()

    # 1. Add the candlestick trace for OHLC data
    fig.add_trace(go.Candlestick(x=stock_data.index,
                               open=stock_data['Open'],
                               high=stock_data['High'],
                               low=stock_data['Low'],
                               close=stock_data['Close'],
                               name='Candlestick'))

    if not orders_df.empty:
        buy_orders = orders_df[orders_df['action'] == 'BUY']
        sell_orders = orders_df[orders_df['action'] == 'SELL']

        # 2. Add markers for BUY orders
        fig.add_trace(go.Scatter(
            x=buy_orders['timestamp'],
            y=buy_orders['price'],
            mode='markers',
            marker=dict(color='purple', symbol='triangle-up', size=20, line=dict(width=1, color='DarkSlateGrey')),
            name='Buy Order'
        ))

        # 3. Add markers for SELL orders
        fig.add_trace(go.Scatter(
            x=sell_orders['timestamp'],
            y=sell_orders['price'],
            mode='markers',
            marker=dict(color='aqua', symbol='triangle-down', size=20, line=dict(width=1, color='DarkSlateGrey')),
            name='Sell Order'
        ))

    # Update layout to make the chart continuous by hiding non-trading periods
    fig.update_layout(
        title=f'Backtest Results for {stock_symbol}',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        xaxis_rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Hide weekends
            dict(bounds=[15.5, 9.25], pattern="hour"),  # Hide non-trading hours (3:30 PM to 9:15 AM)
        ]
    )
    chart_filename = "backtest_chart.html"
    fig.write_html(chart_filename)
    logger.info(f"Interactive chart with trades saved to '{chart_filename}'")

def find_trend_data(data: pd.DataFrame, window_size=100, trend_threshold=0.05, neutral_threshold=0.02) -> dict:
    """
    Finds sample windows of data representing bullish, bearish, and neutral trends.
    A window size of ~60 candles provides a good balance of context for the LLM
    (e.g., 5 hours of data on a 5-minute interval) for it to analyze raw price
    action and decide on appropriate indicators.
    """
    samples = {"bullish": None, "bearish": None, "neutral": None}
    if len(data) < window_size:
        return samples
    for i in range(len(data) - window_size):
        window = data.iloc[i:i + window_size]
        price_change = (window["Close"].iloc[-1] - window["Close"].iloc[0]) / window["Close"].iloc[0]
        if samples["bullish"] is None and price_change > trend_threshold:
            samples["bullish"] = window
        elif samples["bearish"] is None and price_change < -trend_threshold:
            samples["bearish"] = window
        elif samples["neutral"] is None and abs(price_change) < neutral_threshold:
            samples["neutral"] = window
        if all(v is not None for v in samples.values()):
            break
    return samples


def run_llm_backtest(stock_to_trade: str, start_date: str, end_date: str, initial_portfolio_value: float, candle_size: str, generate_new_strategy: bool, train_test_split: float = 0.7):
    logger.info("--- Starting Low-Risk LLM Equity Backtest ---")
    
    news_analyzer = NewsAnalyzer()
    llm_handler = LLMHandler(logger=logger)
    angel_api = AngelAPI()
    all_orders = []

    logger.info(f"Running backtest for: {stock_to_trade}")

    full_data = angel_api.download_historical_data(
        exchange="NSE",
        instrument_symbol=stock_to_trade,
        interval=candle_size,
        from_date=f"{start_date} 09:15",
        to_date=f"{end_date} 15:30",
    )
    if full_data is None or full_data.empty:
        logger.error(f"Could not download data for {stock_to_trade}. Aborting backtest.")
        return

    strategy_file_path = os.path.join(os.path.dirname(__file__), "llm_generated_strategy.py")
    strategy_code = ""

    kernel_generator = KernelSignalGenerator()
    full_data = kernel_generator.process_data(full_data)

    if generate_new_strategy or not os.path.exists(strategy_file_path):
        if not os.path.exists(strategy_file_path):
            logger.warning(f"Strategy file not found at {strategy_file_path}. Generating a new one as fallback.")
        
        # --- Generate new strategy ---
        train_size = int(len(full_data) * train_test_split)
        sampling_data = full_data.iloc[:train_size]
        logger.info(f"Finding trend samples from the training dataset ({len(sampling_data)} rows) for the LLM prompt...")
        trend_samples = find_trend_data(sampling_data)

        logger.info(f"\n--- Generating Python strategy code for {stock_to_trade} ---")
        strategy_code = llm_handler.get_strategy_code(
            stock_to_trade,
            candle_size,
            trend_samples["bullish"],
            trend_samples["bearish"],
            trend_samples["neutral"]
        )
        if strategy_code:
            try:
                with open(strategy_file_path, "w") as f:
                    f.write(strategy_code)
                logger.info(f"LLM-generated strategy code saved to: {strategy_file_path}")
            except IOError as e:
                logger.error(f"Could not write strategy to file {strategy_file_path}: {e}")
    else:
        logger.info(f"Using existing strategy from: {strategy_file_path}")
        try:
            with open(strategy_file_path, "r") as f:
                strategy_code = f.read()
        except IOError as e:
            logger.error(f"Could not read strategy from file {strategy_file_path}: {e}. Aborting.")
            return

    # --- Compile the generated/loaded code ---
    calculate_indicators_generated, should_buy_generated, should_sell_generated = None, None, None
    if strategy_code:
        try:
            strategy_scope = {}
            _, strategy_scope = compile_strategy_code(
                strategy_code, {"pd": pd, "ta": ta}
            )
            calculate_indicators_generated = strategy_scope.get("calculate_indicators")
            should_buy_generated = strategy_scope.get("should_buy")
            should_sell_generated = strategy_scope.get("should_sell")
            logger.info("Successfully compiled strategy functions from code.")
        except Exception as e:
            logger.error(f"Failed to compile strategy code: {e}")

    # --- Prepare data for backtesting (out-of-sample) ---
    train_size = int(len(full_data) * train_test_split)
    backtesting_data_raw = full_data.iloc[train_size:]
    logger.info(f"Using {len(backtesting_data_raw)} rows for backtesting.")
    
    data_with_indicators = pd.DataFrame()
    # Apply the LLM's indicator calculation function to the raw data
    if calculate_indicators_generated:
        logger.info("Applying LLM-generated 'calculate_indicators' function to the backtesting dataset...")
        data_with_indicators = calculate_indicators_generated(backtesting_data_raw.copy())
    else:
        logger.warning("LLM did not provide 'calculate_indicators'. Using default analyzer as a fallback.")
        tech_analyzer = TechnicalAnalyzer() # Fallback
        data_with_indicators = tech_analyzer.calculate_indicators(backtesting_data_raw.copy())

    # --- Data Validation & Enrichment ---
    # The LLM might forget to calculate all necessary indicators or name them correctly.
    # This block ensures the data is in the expected state, making the system more robust.
    logger.info("Validating and enriching indicators post-LLM generation...")
    if 'ATRr_14' in data_with_indicators.columns and 'ATR_14' not in data_with_indicators.columns:
        logger.warning("LLM-generated strategy created 'ATRr_14' but not 'ATR_14'. Calculating ATR_14 as a fallback.")
        # Calculate the 14-period RMA of the True Range to get the final ATR value
        data_with_indicators['ATR_14'] = ta.rma(data_with_indicators['ATRr_14'], length=14)
        data_with_indicators['ATR_14'].bfill(inplace=True) # Back-fill NaNs from the start of the series
    

    logger.info("\n--- Starting Backtesting Loop ---")
    portfolio = {
        "cash": initial_portfolio_value,
        "total_value": initial_portfolio_value,
        "positions": {
            stock_to_trade: {"shares": 0, "value": 0, "avg_price": 0.0, "stop_loss_price": 0.0}
        },
    }

    data_with_indicators.to_csv("backtest_data_with_indicators.csv", index=True)
    stock_data = data_with_indicators.copy()
    backtest_range = stock_data.index
    news_cache = {}

    for current_date in backtest_range:
        # logger.info(f"\n--- Date: {current_date.strftime('%Y-%m-%d %H:%M')} ---") # Too verbose for minute data

        current_total_value = portfolio["cash"]
        position = portfolio["positions"][stock_to_trade]
        if position["shares"] > 0:
            current_price = stock_data.loc[current_date, "Close"]
            position["value"] = position["shares"] * current_price
            current_total_value += position["value"]
        portfolio["total_value"] = current_total_value

        current_data_point = stock_data.loc[current_date]
        current_index = stock_data.index.get_loc(current_date)

        kernel_buy_signal = kernel_generator.should_buy(current_index)
        kernel_sell_signal = kernel_generator.should_sell(current_index)
        kernel_signal = "BUY" if kernel_buy_signal else "SELL" if kernel_sell_signal else "HOLD"

        if should_buy_generated and should_sell_generated:
            current_price = current_data_point["Close"]
            has_position = position["shares"] > 0
            try:
                analysis_date = current_date.to_pydatetime() if hasattr(current_date, "to_pydatetime") else current_date
                date_key = analysis_date.date() if hasattr(analysis_date, "date") else analysis_date
                if date_key not in news_cache:
                    try:
                        news_cache[date_key] = news_analyzer.get_news_sentiment(
                            stock_to_trade, analysis_date
                        )
                    except Exception as news_error:
                        logger.warning("News lookup failed for %s: %s", date_key, news_error)
                        news_cache[date_key] = {
                            "overall_sentiment": "Neutral",
                            "sentiment_score": 0.0,
                            "top_headlines": [],
                        }
                news_context = news_cache[date_key]
                buy_signal = should_buy_generated(current_data_point, kernel_signal, news_context)

                if not has_position and buy_signal:
                    trade_value = portfolio["cash"] * 0.25
                    quantity = int(trade_value / current_price)
                    if quantity > 0 and portfolio["cash"] >= (quantity * current_price):
                        cost = quantity * current_price
                        portfolio["cash"] -= cost
                        position["shares"] += quantity
                        position["avg_price"] = current_price
                        # Set the initial stop-loss price when buying
                        position["stop_loss_price"] = current_price * (1 - STOP_LOSS_PERCENT / 100.0)
                        logger.info(f"Executed BUY (LLM Signal): {quantity} shares @ INR {current_price:.2f}. Initial Stop-Loss set to {position['stop_loss_price']:.2f}")
                        all_orders.append({"stock_symbol": stock_to_trade, "action": "BUY", "quantity": quantity, "price": current_price, "timestamp": current_date, "status": "EXECUTED", "reason": "LLM Signal"})

                elif has_position:
                    # --- Sell logic combines risk management rules with the LLM signal ---
                    avg_price = position["avg_price"]
                    current_price = current_data_point["Close"]

                    # --- Trailing Stop-Loss Logic ---
                    # Calculate a potential new stop-loss price based on the current price.
                    original_stop_loss = current_price <= avg_price * (1 - STOP_LOSS_PERCENT / 100.0)

                    potential_new_stop_loss = current_price * (1 - STOP_LOSS_PERCENT / 100.0)
                    # The stop-loss only moves up. If the new potential stop-loss is higher, update it.
                    if potential_new_stop_loss > position["stop_loss_price"]:
                        position["stop_loss_price"] = potential_new_stop_loss

                    # 1. Check for LLM-driven sell signal
                    llm_sell_signal = should_sell_generated(current_data_point, kernel_signal, news_context)

                    # 2. Check for trailing stop-loss
                    stop_loss_triggered = current_price <= position["stop_loss_price"]

                    # 3. Check for hard take-profit
                    take_profit_triggered = current_price >= avg_price * (1 + TAKE_PROFIT_PERCENT / 100.0)
                    
                    # 4. LLM sell is only considered if a minimum profit is met
                    min_profit_target_met = current_price >= avg_price * (1 + MIN_PROFIT_FOR_LLM_SELL_PERCENT / 100.0)

                    # Risk exits fire regardless of profit; LLM sells still require a minimum gain.
                    if original_stop_loss or stop_loss_triggered or take_profit_triggered or (llm_sell_signal and min_profit_target_met):
                        if original_stop_loss or stop_loss_triggered:
                            reason = "Stop-Loss"
                        elif take_profit_triggered:
                            reason = "Take-Profit"
                        else:
                            reason = "LLM Signal"
                        profit_per_share = current_price - avg_price
                        logger.info(f"Executed SELL ({reason}): {position['shares']} shares @ INR {current_price:.2f} (Profit/share: {profit_per_share:.2f})")
                        
                        quantity = position["shares"]
                        proceeds = quantity * current_price
                        portfolio["cash"] += proceeds
                        position["shares"] = 0
                        position["avg_price"] = 0.0
                        position["stop_loss_price"] = 0.0 # Reset stop loss price
                        all_orders.append({"stock_symbol": stock_to_trade, "action": "SELL", "quantity": quantity, "price": current_price, "timestamp": current_date, "status": "EXECUTED", "reason": reason})

            except Exception as e:
                logger.error(f"Error executing LLM-generated function on {current_date}: {e}")

        # logger.info(f"End of Tick Portfolio Value: INR {portfolio['total_value']:.2f}")

    logger.info("\n--- Backtest Finished ---")
    final_value = portfolio["total_value"]
    logger.info(f"Initial Portfolio Value: INR {initial_portfolio_value:,.2f}")
    logger.info(f"Final Portfolio Value:   INR {final_value:,.2f}")
    performance = (final_value - initial_portfolio_value) / initial_portfolio_value
    logger.info(f"Total Performance: {performance:.2%}")

    # --- Save Orders and Plot Results ---
    if all_orders:
        orders_df = pd.DataFrame(all_orders)
        # Ensure timestamp is in datetime format for plotting
        orders_df["timestamp"] = pd.to_datetime(orders_df["timestamp"])
        orders_df[["timestamp", "stock_symbol", "action", "reason", "quantity", "price", "status"]].to_csv("backtest_orders.csv", index=False)
        logger.info("Executed orders saved to 'backtest_orders.csv'")
        plot_backtest_results(stock_data, orders_df, stock_to_trade)
    else:
        logger.info("No orders were executed during the backtest.")


if __name__ == "__main__":
    STOCK_TO_TEST = "ICICIBANK-EQ"
    # Note: Today's date is 2025-07-29
    START_DATE = "2025-03-20" 
    END_DATE = "2025-04-30"
    PORTFOLIO_VALUE = 100000.0
    CANDLE_SIZE = "ONE_MINUTE"
    GENERATE_NEW_STRATEGY = False  # Set to True to generate a new, improved strategy
    TRAIN_TEST_SPLIT = 0.3  # 30% training data, 70% backtesting
    run_llm_backtest(STOCK_TO_TEST, START_DATE, END_DATE, PORTFOLIO_VALUE, CANDLE_SIZE, GENERATE_NEW_STRATEGY, TRAIN_TEST_SPLIT)
