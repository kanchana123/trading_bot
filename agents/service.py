import uuid
from threading import RLock
from typing import Dict, List, Optional

import pandas as pd

from agents.graph import resume_graph, run_graph
from agents.state import AgentState, Bar, Position, TradingMode
from db.audit_manager import AuditLedger
from rag.tools import HistoricalBarsRequest, MarketTools


class InvestmentGraphService:
    """API facade. Agents never receive TradeExecutor except at execute_order."""

    def __init__(self, trade_executor, audit: AuditLedger, angel_api=None):
        self.trade_executor = trade_executor
        self.audit = audit
        self.market = MarketTools(angel_api=angel_api)
        self._lock = RLock()
        self._live: Dict[str, AgentState] = {}

    def start_run(
        self,
        *,
        symbol: str,
        cash: float = 100000.0,
        trading_mode: str = "virtual",
        instrument_token: str = "",
        exchange: str = "NSE_EQ",
        portfolio_id: Optional[int] = None,
        strategy_id: Optional[int] = None,
        bars: Optional[List[dict]] = None,
        positions: Optional[Dict] = None,
        max_name_weight: float = 0.05,
        auto_approve_confidence: float = 0.92,
        last_price: Optional[float] = None,
    ) -> AgentState:
        mode = TradingMode(trading_mode)
        run_id = str(uuid.uuid4())
        parsed_bars = self._resolve_bars(symbol, bars)
        price = last_price or (parsed_bars[-1].close if parsed_bars else 100.0)
        pos_models = {}
        for key, val in (positions or {}).items():
            if isinstance(val, Position):
                pos_models[key] = val
            else:
                pos_models[key] = Position.model_validate(val)
        state = AgentState(
            run_id=run_id,
            thread_id=run_id,
            symbol=symbol,
            instrument_token=instrument_token,
            exchange=exchange,
            trading_mode=mode,
            portfolio_id=portfolio_id,
            strategy_id=strategy_id,
            cash=cash,
            nav=cash + sum(p.market_value for p in pos_models.values()),
            max_name_weight=max_name_weight,
            auto_approve_confidence=auto_approve_confidence,
            positions=pos_models,
            bars=parsed_bars,
            last_price=price,
        )
        state = run_graph(
            state,
            trade_executor=self.trade_executor,
            audit=self.audit,
            stop_before_execute=True,
        )
        with self._lock:
            self._live[run_id] = state
        return state

    def approve(self, run_id: str, operator_id: str) -> AgentState:
        return self._resume(run_id, "approve", operator_id)

    def reject(self, run_id: str, operator_id: str) -> AgentState:
        return self._resume(run_id, "reject", operator_id)

    def get(self, run_id: str) -> Optional[AgentState]:
        with self._lock:
            if run_id in self._live:
                return self._live[run_id]
        row = self.audit.get_run(run_id)
        if not row:
            return None
        return AgentState.model_validate(row["state"])

    def pending(self) -> List[Dict]:
        return self.audit.list_pending()

    def history(self, limit: int = 50) -> List[Dict]:
        return self.audit.list_runs(limit=limit)

    def _resume(self, run_id: str, decision: str, operator_id: str) -> AgentState:
        state = self.get(run_id)
        if state is None:
            raise KeyError(run_id)
        if not state.interrupted:
            raise RuntimeError("run is not awaiting human approval")
        state = resume_graph(
            state,
            decision,
            operator_id,
            trade_executor=self.trade_executor,
            audit=self.audit,
        )
        with self._lock:
            self._live[run_id] = state
        return state

    def _resolve_bars(self, symbol: str, bars: Optional[List[dict]]) -> List[Bar]:
        if bars:
            return [Bar.model_validate(_normalize_bar(row)) for row in bars]
        loaded = self.market.load_bars(HistoricalBarsRequest(symbol=symbol, limit=40), rows=None)
        if loaded:
            return [Bar.model_validate(_normalize_bar(row)) for row in loaded]
        return []


def _normalize_bar(row: dict) -> dict:
    if {"open", "high", "low", "close"} <= {k.lower() for k in row}:
        lowered = {k.lower(): v for k, v in row.items()}
        return {
            "open": lowered["open"],
            "high": lowered["high"],
            "low": lowered["low"],
            "close": lowered["close"],
            "volume": lowered.get("volume") or 0,
        }
    return {
        "open": row.get("Open", row.get("open")),
        "high": row.get("High", row.get("high")),
        "low": row.get("Low", row.get("low")),
        "close": row.get("Close", row.get("close")),
        "volume": row.get("Volume", row.get("volume") or 0),
    }


def bars_from_frame(df: pd.DataFrame) -> List[dict]:
    out = []
    for _, row in df.iterrows():
        out.append(
            {
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": float(row.get("Volume") or 0),
            }
        )
    return out
