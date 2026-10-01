import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(close: pd.Series, window: int = 20, std_multiplier: float = 2.0):
    middle = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + std_multiplier * std
    lower = middle - std_multiplier * std
    return middle, upper, lower


def kernel_momentum(high: pd.Series, low: pd.Series, close: pd.Series, kernel) -> pd.Series:
    from scipy.signal import convolve

    kernel = np.asarray(kernel, dtype=float)
    mid = (high + low) / 2
    signal_values = (close - mid).values
    if len(signal_values) < len(kernel):
        return pd.Series(np.nan, index=close.index)
    momentum = convolve(signal_values, kernel, mode="valid")
    padding = len(close) - len(momentum)
    padded = np.pad(momentum, (max(0, padding), 0), "constant", constant_values=np.nan)
    return pd.Series(padded, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window=period, min_periods=period).mean()
