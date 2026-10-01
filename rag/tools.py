import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class MarketQuote(BaseModel):
    symbol: str
    last_price: float = Field(gt=0)
    source: str = "fixture"


class HistoricalBarsRequest(BaseModel):
    symbol: str
    limit: int = Field(default=40, ge=5, le=500)

    @field_validator("symbol")
    @classmethod
    def symbol_ok(cls, value: str) -> str:
        if not value or len(value) < 3:
            raise ValueError("symbol required")
        return value.strip().upper()


def _retry(fn: Callable, attempts: int = 3, delay: float = 0.05):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            time.sleep(delay * (i + 1))
    raise last


class MarketTools:
    """Validated wrappers around Angel or synthetic bars. No order placement."""

    def __init__(self, angel_api=None):
        self.angel_api = angel_api

    def get_quote(self, symbol: str, last_price: Optional[float] = None) -> MarketQuote:
        if last_price and last_price > 0:
            return MarketQuote(symbol=symbol, last_price=float(last_price), source="provided")
        return MarketQuote(symbol=symbol, last_price=100.0, source="fixture")

    def load_bars(self, request: HistoricalBarsRequest, rows: Optional[List[Dict[str, Any]]] = None):
        if rows:
            return rows

        def _download():
            api = self.angel_api
            if api is None or not getattr(api, "is_session_active", lambda: False)():
                raise RuntimeError("angel_unavailable")
            df = api.download_historical_data(
                exchange="NSE",
                instrument_symbol=request.symbol,
                interval="ONE_DAY",
                days=min(request.limit, 120),
            )
            if df is None or getattr(df, "empty", True):
                raise RuntimeError("empty_history")
            return df.tail(request.limit).to_dict(orient="records")

        try:
            return _retry(_download)
        except Exception:
            return None
