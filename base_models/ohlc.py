from typing import Optional

import pandas as pd

_COLUMN_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "date": "Date",
    "timestamp": "timestamp",
}


def normalize_ohlc_columns(data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Rename common OHLC column variants to Open/High/Low/Close/Volume."""
    if data is None or data.empty:
        return data

    rename = {}
    for col in data.columns:
        mapped = _COLUMN_MAP.get(str(col).strip().lower())
        if mapped and col != mapped and mapped not in data.columns:
            rename[col] = mapped
    if rename:
        data = data.rename(columns=rename)
    return data
