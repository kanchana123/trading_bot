# Note: This module requires the openai library. Please install it using:
# pip install openai
#
# You also need to set your OpenAI API key as an environment variable.
# For example, in your terminal:
# export OPENAI_API_KEY='your_api_key_here'

import os
import json
from typing import Dict, List, Any
import logging

from openai import OpenAI, OpenAIError
import pandas as pd


class LLMHandler:
    """
    Handles interaction with a Large Language Model (LLM) like GPT-4 to get
    trading decisions.
    """

    def __init__(self, api_key: str = None, logger=None):
        """
        Initializes the LLM handler with an OpenAI API key.

        Args:
            api_key (str, optional): OpenAI API key. If None, it will be
                                     read from the OPENAI_API_KEY environment variable.
        """
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Please provide it or set the "
                "OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=key)
        # If no logger is provided, create a basic one that just prints
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(logging.StreamHandler())

    def generate_prompt(
        self, portfolio_state: Dict, market_data_by_stock: Dict[str, Any]
    ) -> str:
        """
        Generates a detailed and token-optimized prompt for the LLM.
        """
        # System message: concise and direct
        prompt_parts = [
            (
                "You are a low-risk trading analyst focused on capital preservation. "
                "Analyze the provided data and decide to BUY, SELL, or HOLD. "
                "Your response MUST be a single JSON object with an 'orders' list. "
                "Do not add any text outside the JSON."
            ),
            "\n--- PORTFOLIO STATE ---",
            f"Cash: INR {portfolio_state.get('cash', 0):,.2f}",
            f"Total Value: INR {portfolio_state.get('total_value', 0):,.2f}",
            f"Current Positions: {json.dumps(portfolio_state.get('positions', {}), indent=2)}",
            "\n--- MARKET DATA ANALYSIS ---",
        ]

        for stock, data in market_data_by_stock.items():
            prompt_parts.append(f"\n--- Stock: {stock} ---")

            # Format all numbers to 2 decimal places to save tokens
            current_price = data.get('current_price', 0.0)
            prompt_parts.append(f"Current Price: INR {current_price:.2f}")

            recent_closes_raw = data.get('recent_closes', [])
            recent_closes_str = [f"{p:.2f}" for p in recent_closes_raw] if isinstance(recent_closes_raw, list) else 'N/A'
            prompt_parts.append(f"Recent Closes: {recent_closes_str}")

            prompt_parts.append("\nTechnical Indicators:")
            for key, val in data.get("indicators", {}).items():
                if isinstance(val, (int, float)):
                    prompt_parts.append(f"- {key}: {val:.2f}")
                else:
                    prompt_parts.append(f"- {key}: {val}")

            prompt_parts.append("\nSignals:")
            prompt_parts.append(f"- Kernel Signal: {data.get('kernel_signal', 'N/A')}")

            news_data = data.get('news', {})
            sentiment_score = news_data.get('sentiment_score')
            sentiment_str = f"{sentiment_score:.2f}" if isinstance(sentiment_score, float) else "N/A"
            prompt_parts.append(f"- News Sentiment: {news_data.get('overall_sentiment', 'N/A')} (Score: {sentiment_str})")

            prompt_parts.append("- Top Headlines:")
            for headline in news_data.get('top_headlines', []):
                prompt_parts.append(f"  - {headline.get('title')}")

        # Instructions: concise, with a clear example defining the schema
        prompt_parts.extend([
            "\n--- INSTRUCTIONS & JSON FORMAT ---",
            (
                "Based on the data, decide whether to BUY, SELL, or HOLD for each stock. "
                "Your response MUST be a single JSON object with an 'orders' key, containing a list of order objects. "
                "IMPORTANT RULE: Only generate a 'SELL' order for a stock if its 'Current Price' is at least INR 5.00 higher "
                "than its 'avg_price' in the 'Current Positions'. Do not sell at a loss or for a small profit. "
                "The 'avg_price' for a stock is available in the 'Current Positions' section of the portfolio state. "
                "An empty list `[]` means hold all positions. "
                "Each order must follow the schema in this example:"
            ),
            # Example serves as the schema definition
            """{
  "orders": [
    {
      "stock_symbol": "RELIANCE-EQ",
      "action": "BUY",
      "quantity": 10,
      "order_type": "MARKET",
      "price": null,
      "stop_loss": 2750.00,
      "take_profit": 2900.00,
      "reasoning": "Positive news and bullish MACD suggest upward momentum."
    }
  ]
}"""
        ])
        return "\n".join(prompt_parts)

    def get_trading_decision(self, prompt: str) -> str:
        """
        Calls the GPT-4 API to get a trading decision.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Or another suitable model
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,  # Lower temperature for more deterministic, fact-based output
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            self.logger.error(f"An error occurred with the OpenAI API: {e}")
            return "{}" # Return empty JSON object on error

    def parse_llm_response(self, response_content: str) -> List[Dict]:
        """
        Parses the JSON response from the LLM and validates its structure.
        """
        try:
            # The response might be wrapped in markdown ```json ... ```
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0]

            data = json.loads(response_content)
            orders = data.get("orders")

            if isinstance(orders, list):
                return orders
            else:
                self.logger.warning(f"LLM response 'orders' key is not a list. Response: {response_content}")
                return []
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode JSON from LLM response. Response: {response_content}")
            return []
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while parsing LLM response: {e}")
            return []

    def get_llm_orders(
        self, portfolio_state: Dict, market_data_by_stock: Dict[str, Any]
    ) -> List[Dict]:
        """
        Orchestrates the process of generating a prompt, calling the LLM,
        and parsing the response to get trading orders.
        """
        prompt = self.generate_prompt(portfolio_state, market_data_by_stock)
        self.logger.info(f"--- LLM Trading Prompt ---\n{prompt}\n--------------------------")
        
        llm_response = self.get_trading_decision(prompt)
        self.logger.info(f"--- Received LLM Trading Response ---\n{llm_response}\n---------------------------")

        orders = self.parse_llm_response(llm_response)
        return orders

    def generate_parameter_optimization_prompt(
        self, stock_symbol: str, data_sample: pd.DataFrame
    ) -> str:
        """
        Generates a prompt to ask the LLM to optimize technical indicator parameters.
        """
        # To save tokens, we'll just pass the closing prices
        close_prices = [f"{p:.2f}" for p in data_sample["Close"].tolist()]

        prompt_parts = [
            (
                "You are an expert quantitative analyst. Your task is to find optimal parameters "
                "for a set of technical indicators for a short-to-medium term trading strategy "
                f"on the stock '{stock_symbol}'."
            ),
            (
                "Based on the following sample of recent closing prices, suggest integer "
                "parameters that would be effective at identifying momentum shifts and "
                "mean-reversion opportunities. The stock is a large-cap equity."
            ),
            f"\n--- DATA SAMPLE ({len(close_prices)} Recent Close Prices) ---",
            f"[{', '.join(close_prices)}]",
            "\n--- PARAMETERS TO OPTIMIZE ---",
            "Provide optimal integer values for the following parameters. Your response MUST be a single JSON object.",
            "- `sma_short`: Short-term moving average window (suggest between 5-20).",
            "- `sma_long`: Long-term moving average window (suggest between 30-60).",
            "- `rsi_period`: Lookback period for RSI (suggest between 10-20).",
            "- `macd_fast`: Fast EMA for MACD (suggest between 8-15).",
            "- `macd_slow`: Slow EMA for MACD (suggest between 20-30).",
            "- `macd_signal`: Signal line for MACD (suggest between 7-10).",
            "- `bb_window`: Window for Bollinger Bands (suggest between 15-25).",
            "- `coppock_wma_period`: WMA period for Coppock Curve (suggest between 8-12).",
            "\n--- REQUIRED JSON FORMAT ---",
            """{
  "sma_short": 10,
  "sma_long": 50,
  "rsi_period": 14,
  "macd_fast": 12,
  "macd_slow": 26,
  "macd_signal": 9,
  "bb_window": 20,
  "coppock_wma_period": 10
}""",
        ]
        return "\n".join(prompt_parts)

    def get_optimized_parameters(
        self, stock_symbol: str, data_sample: pd.DataFrame
    ) -> Dict:
        """
        Orchestrates calling the LLM to get optimized technical analysis parameters.

        Args:
            stock_symbol (str): The stock symbol for context.
            data_sample (pd.DataFrame): A sample of historical data (e.g., 200 candles).

        Returns:
            Dict: A dictionary of optimized parameters, or an empty dict on error.
        """
        if data_sample.empty:
            self.logger.warning("Data sample is empty. Cannot optimize parameters.")
            return {}

        prompt = self.generate_parameter_optimization_prompt(stock_symbol, data_sample)
        self.logger.info(f"--- LLM Parameter Optimization Prompt for {stock_symbol} ---\n{prompt}\n--------------------------")

        llm_response = self.get_trading_decision(prompt)
        self.logger.info(f"--- Received LLM Parameter Response for {stock_symbol} ---\n{llm_response}\n---------------------------")

        try:
            params = json.loads(llm_response)
            if isinstance(params, dict) and "sma_short" in params:
                return params
            return {}
        except (json.JSONDecodeError, TypeError):
            self.logger.error(f"Error: Failed to decode or validate JSON from LLM parameter response. Response: {llm_response}")
            return {}

    def generate_strategy_code_prompt(
        self,
        stock_symbol: str,
        candle_size: str,
        bullish_sample: pd.DataFrame,
        bearish_sample: pd.DataFrame,
        neutral_sample: pd.DataFrame,
    ) -> str:
        """
        Generates a prompt asking the LLM to write Python trading logic functions.
        """
        # Helper to format dataframe samples for the prompt
        def format_sample(df: pd.DataFrame, trend_name: str) -> str:
            if df is None or df.empty:
                return f"\n--- No data sample available for {trend_name.upper()} trend. ---"
            # Show OHLCV and the pre-calculated kernel_momentum to the LLM
            cols_to_show = ["Open", "High", "Low", "Close", "Volume", "kernel_momentum"]
            return (
                f"\n--- Data Sample for a {trend_name.upper()} Trend ---\n"
                f"A good strategy would likely generate {'BUY' if trend_name == 'bullish' else 'SELL/HOLD'} signals in this context.\n"
                f"{df[cols_to_show].to_string()}\n"
            )

        prompt_parts = [
            (
                "You are an expert quantitative analyst and Python programmer. Your task is to create a robust intraday trading strategy. "
                f"The strategy will be for the stock '{stock_symbol}' using a '{candle_size}' candlestick interval. "
                "You will act as both a feature engineer and a strategy developer. Based on the provided raw OHLCV data, you will first decide which technical indicators are most appropriate, "
                "and then write a complete, self-contained trading strategy using them."
            ),
            (
                "\nYour output MUST be a single Python code block containing three functions with these exact signatures:\n"
                "`def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:`\n"
                "`def should_buy(data: pd.Series, kernel_signal: str, news: Dict, trading_env: str) -> bool:`\n"
                "`def should_sell(data: pd.Series, kernel_signal: str, news: Dict, trading_env: str) -> bool:`"
            ),
            "\n--- YOUR THOUGHT PROCESS ---",
            "1. **Analyze the Data Samples**: Look at the raw OHLCV data for the bullish, bearish, and neutral trends provided below.",
            "2. **Select Indicators**: Based on the patterns you see, decide which technical indicators from the `pandas_ta` library would be most effective at identifying these trends and their reversals. Consider momentum, trend, and volatility.",
            "3. **Implement `calculate_indicators`**: Write the code to calculate your chosen indicators. Use `df.ta.indicator_name(append=True)` to add them to the DataFrame.",
            "4. **Implement `should_buy` and `should_sell`**: Write the logic for the buy and sell signals using the indicators you just created, plus the provided `kernel_signal` and `news` sentiment.",
            "\n--- INPUT PARAMETER DETAILS ---",
            (
                "- `calculate_indicators` receives a DataFrame that already contains OHLCV and a pre-calculated `kernel_momentum` column. "
                "Your function should add *new* technical indicator columns to this DataFrame. The original columns must be preserved."
            ),
            (
                "- `should_buy` and `should_sell` receive a `data` pd.Series for the current time step. This Series will contain the columns you added in `calculate_indicators`. "
                "The timestamp for the current candle is available via `data.name`."
            ),
            "- `kernel_signal` (str): A momentum signal, which can be 'BUY', 'SELL', or 'HOLD'.",
            (
                "- `kernel_momentum` (float): **CRITICAL SIGNAL**. This is a powerful pre-calculated score designed to spot **potential trend reversals**. "
                "Your strategy should prioritize using this signal. A common and effective pattern is:\n"
                "  - **BUY on Reversal**: `kernel_momentum` drops *below* a negative threshold (e.g., -1.5), indicating the stock is oversold and may reverse upwards. Confirm with another indicator (like MACD turning bullish).\n"
                "  - **SELL on Reversal**: `kernel_momentum` rises *above* a positive threshold (e.g., 1.5), indicating the stock is overbought and may reverse downwards. Confirm with another indicator (like RSI being high).\n"
                "The example code below demonstrates the correct way to use this."
            ),
            (
                "- `news` (Dict): A dictionary with news sentiment. Access sentiment with `news['overall_sentiment']` "
                "('Positive', 'Negative', 'Neutral') and `news['sentiment_score']` (a float from -1 to 1)."
            ),
            (
                "- `trading_env` (str): The environment the strategy is running in. It will be 'backtesting' during simulations. "
                "**Your code MUST check this variable and should NOT use the `news` data if `trading_env` is 'backtesting'.**"
            ),
            "\n--- CONTEXT: RAW DATA SAMPLES FOR DIFFERENT MARKET TRENDS ---",
            "Analyze these raw OHLCV data samples to decide on the best indicators and create your strategy.",
            format_sample(bullish_sample, "bullish"),
            format_sample(bearish_sample, "bearish"),
            format_sample(neutral_sample, "neutral"),
            "\n--- INSTRUCTIONS & REQUIRED OUTPUT FORMAT ---",
            (
                "1. Your response MUST be a single, clean Python code block starting with `import pandas as pd`.\n"
                "2. Do NOT include any explanations, comments outside the functions, or any text before or after the code block.\n"
                "3. In `calculate_indicators`, use the `pandas_ta` library. The function must return the DataFrame with the new indicator columns.\n"
                "4. The `should_buy` and `should_sell` functions must use the columns you created. Use `.get('COLUMN_NAME', default_value)` for safe access.\n"
                "5. **CRITICAL TIME-BASED RULES FOR `should_buy`**:\n"
                "   - **Morning Volatility**: Do NOT generate a buy signal between 09:15 AM and 09:45 AM. The market is too volatile. You can check this with `data.name.time()`.\n"
                "   - **End-of-Day Closing**: Do NOT generate a buy signal after 02:45 PM to avoid holding positions overnight."
            ),
            "\n--- EXAMPLE OF A COMPLETE RESPONSE ---",
            "\n```python",
            "import pandas as pd",
            "import pandas_ta as ta",
            "from typing import Dict",
            "\n# --- LLM Generated Strategy ---",
            "\ndef calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:",
            "    # Analysis: The data shows periods of strong trends and some volatility.",
            "    # A combination of a trend indicator (MACD) and a momentum/volatility indicator (RSI, Bollinger Bands) seems appropriate.",
            "    df = data.copy()",
            "    # Calculate MACD, RSI, and Bollinger Bands and append them to the DataFrame.",
            "    df.ta.macd(fast=12, slow=26, signal=9, append=True)",
            "    df.ta.rsi(length=14, append=True)",
            "    df.ta.bbands(length=20, std=2, append=True)",
            "    return df",
            "",
            "\ndef should_buy(data: pd.Series, kernel_signal: str, news: Dict, trading_env: str = 'backtesting') -> bool:",
            "    # Using the indicators calculated above.",
            "    # Note: pandas_ta names columns like 'MACD_12_26_9', 'RSI_14', 'BBL_20_2.0'.",
            "    try:",
            "        # Time-based rules to avoid high volatility and overnight risk",
            "        current_time = data.name.time()",
            "        if current_time < pd.to_datetime('09:45').time() or current_time > pd.to_datetime('14:45').time():",
            "            return False",
            "",
            "        # --- Primary Buy Signal: Kernel Momentum Reversal ---",
            "        is_reversal_buy = data.get('kernel_momentum', 0) <= -1.5",
            "        # --- Confirmation Signal: MACD ---",
            "        macd_bullish = data.get('MACD_12_26_9', 0) > data.get('MACDs_12_26_9', 0)",
            "",
            "        # Buy if the primary kernel reversal signal is confirmed by a bullish MACD.",
            "        if is_reversal_buy and macd_bullish:",
            "            return True",
            "",
            "        # Also consider buying on strong positive news if confirmed by momentum",
            "        if trading_env != 'backtesting':",
            "            if news.get('sentiment_score', 0) > 0.5 and macd_bullish:",
            "                return True",
            "",
            "    except Exception:",
            "        return False",
            "    return False",
            "\ndef should_sell(data: pd.Series, kernel_signal: str, news: Dict, trading_env: str = 'backtesting') -> bool:",
            "    # Example using the calculated indicators.",
            "    # --- Primary Sell Signal: Kernel Momentum Reversal ---",
            "    is_reversal_sell = data.get('kernel_momentum', 0) >= 1.5",
            "    # --- Confirmation Signal: RSI ---",
            "    is_rsi_overbought = data.get('RSI_14', 50) > 70",
            "",
            "    # Sell if the primary kernel reversal signal is confirmed by a high RSI.",
            "    if is_reversal_sell and is_rsi_overbought:",
            "        return True",
            "",
            "    return False",
            "```",
        ]
        return "\n".join(prompt_parts)

    def get_strategy_code(
        self,
        stock_symbol: str,
        candle_size: str,
        bullish_sample: pd.DataFrame,
        bearish_sample: pd.DataFrame,
        neutral_sample: pd.DataFrame,
    ) -> str:
        """
        Orchestrates calling the LLM to get Python code for the trading strategy.
        """
        prompt = self.generate_strategy_code_prompt(
            stock_symbol, candle_size, bullish_sample, bearish_sample, neutral_sample
        )
        self.logger.info(f"--- LLM Strategy Code Generation Prompt for {stock_symbol} ---\n{prompt}\n--------------------------")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, # A bit of creativity for code generation
            )
            response_content = response.choices[0].message.content
            self.logger.info(f"--- Received LLM Strategy Code ---\n{response_content}\n---------------------------")

            # Extract code from markdown block
            if "```python" in response_content:
                return response_content.split("```python")[1].split("```")[0].strip()
            return response_content # Fallback if no markdown

        except OpenAIError as e:
            self.logger.error(f"An error occurred with the OpenAI API: {e}")
            return "" # Return empty string on error