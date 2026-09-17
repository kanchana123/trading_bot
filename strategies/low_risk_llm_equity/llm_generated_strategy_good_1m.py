import pandas as pd
import pandas_ta as ta
from typing import Dict

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # Adding MACD for trend following, RSI for momentum, and ATR for volatility
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    df.ta.rsi(close='Close', length=14, append=True)
    df.ta.atr(length=14, append=True)
    return df

def should_buy(data: pd.Series, kernel_signal: str, news: Dict, trading_env: str = 'backtesting') -> bool:
    current_time = data.name.time()
    # Avoiding early morning volatility and end-of-day trading
    if current_time < pd.to_datetime('09:45').time() or current_time > pd.to_datetime('14:45').time():
        return False
    # Primary signal: kernel_momentum for potential reversal
    kernel_momentum_buy_signal = data.get('kernel_momentum', 0) <= -1.5
    # Confirmatory signals: MACD for trend reversal and RSI for momentum
    macd_bullish = data.get('MACD_12_26_9', 0) > data.get('MACDs_12_26_9', 0)
    rsi_not_overbought = data.get('RSI_14', 0) < 70

    # Considering news sentiment if not backtesting
    if trading_env != 'backtesting':
        positive_news = news.get('overall_sentiment', 'Neutral') == 'Positive' and news.get('sentiment_score', 0) > 0.5
        if positive_news and macd_bullish:
            return True
        
    # Combining signals for a buy decision
    if kernel_momentum_buy_signal and macd_bullish and rsi_not_overbought:
        return True
    
    return False

def should_sell(data: pd.Series, kernel_signal: str, news: Dict, trading_env: str = 'backtesting') -> bool:
    # Primary signal: kernel_momentum for potential reversal
    kernel_momentum_sell_signal = data.get('kernel_momentum', 0) >= 1.5
    # Confirmatory signals: MACD for trend continuation and RSI for overbought condition
    macd_bearish = data.get('MACD_12_26_9', 0) < data.get('MACDs_12_26_9', 0)
    rsi_overbought = data.get('RSI_14', 0) > 70
    # Combining signals for a sell decision
    if kernel_momentum_sell_signal and (rsi_overbought or macd_bearish):
        return True
    return False