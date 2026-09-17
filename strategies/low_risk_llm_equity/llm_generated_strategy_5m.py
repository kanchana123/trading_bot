import pandas as pd
import pandas_ta as ta
from typing import Dict

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    df.ta.rsi(close='Close', length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.sma(close='Close', length=20, append=True)
    df.ta.ema(close='Close', length=9, append=True)
    return df

def should_buy(data: pd.Series, kernel_signal: str, news: Dict) -> bool:
    is_positive_news = news.get('sentiment_score', 0) > 0
    macd_above_signal = data.get('MACD_12_26_9', 0) > data.get('MACDs_12_26_9', 0)
    rsi_below_30 = data.get('RSI_14', 100) < 30
    price_above_ema = data.get('Close', 0) > data.get('EMA_9', 0)
    if (macd_above_signal and rsi_below_30 and price_above_ema) or (is_positive_news and macd_above_signal):
        return True
    return False

def should_sell(data: pd.Series, kernel_signal: str, news: Dict) -> bool:
    is_negative_news = news.get('sentiment_score', 0) < 0
    macd_below_signal = data.get('MACD_12_26_9', 0) < data.get('MACDs_12_26_9', 0)
    rsi_above_70 = data.get('RSI_14', 0) > 70
    price_below_sma = data.get('Close', 0) < data.get('SMA_20', 0)
    if (macd_below_signal and rsi_above_70) or (is_negative_news and price_below_sma):
        return True
    return False