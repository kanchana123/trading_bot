from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from agents.state import AgentState, Bar, CandidateSignal, Side
from observability.tracing import TRACER
from strategies.action_price.KernelTrader import KernelBacktestAdapter
from strategies.bollingerBandStrategy import BollingerBandStrategy


def _bars_to_frame(bars: List[Bar]) -> Optional[pd.DataFrame]:
    if not bars:
        return None
    return pd.DataFrame(
        {
            "Open": [b.open for b in bars],
            "High": [b.high for b in bars],
            "Low": [b.low for b in bars],
            "Close": [b.close for b in bars],
            "Volume": [b.volume for b in bars],
        }
    )


def _synthetic_trend(n: int = 40, start: float = 100.0) -> List[Bar]:
    bars: List[Bar] = []
    price = start
    for i in range(n):
        price += 1 if i % 7 != 0 else -2
        bars.append(
            Bar(
                open=price - 0.5,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000 + i,
            )
        )
    return bars


def quant_node(state: AgentState) -> AgentState:
    with TRACER.start_as_current_span("quant_node", symbol=state.symbol):
        if len(state.bars) < 10:
            state.bars = _synthetic_trend()
        frame = _bars_to_frame(state.bars)
        state.last_price = float(state.bars[-1].close)
        signals: List[CandidateSignal] = []
        idx = len(frame) - 1
        qty_budget = max(int(state.nav * state.max_name_weight / state.last_price), 1)

        bb = BollingerBandStrategy(params={"window": min(20, max(5, len(frame) // 2)), "std_multiplier": 2})
        bb.set_data(frame.copy())
        bb.process_data()
        if bb.should_buy(idx):
            signals.append(
                CandidateSignal(
                    source="BollingerBand",
                    side=Side.BUY,
                    symbol=state.symbol,
                    strength=0.55,
                    price=state.last_price,
                    quantity=qty_budget,
                    reason="Close reclaimed the lower Bollinger band.",
                    indicators={"close": state.last_price},
                )
            )
        elif bb.should_sell(idx):
            signals.append(
                CandidateSignal(
                    source="BollingerBand",
                    side=Side.SELL,
                    symbol=state.symbol,
                    strength=0.5,
                    price=state.last_price,
                    quantity=qty_budget,
                    reason="Close rejected the upper Bollinger band.",
                    indicators={"close": state.last_price},
                )
            )

        kernel = KernelBacktestAdapter(
            params={"kernel": [-2, -1, 0, 1, 2], "threshold": 1.5, "quantity": qty_budget}
        )
        kernel.set_data(frame.copy())
        kernel.process_data()
        if kernel.should_buy(idx):
            signals.append(
                CandidateSignal(
                    source="KernelMomentum",
                    side=Side.BUY,
                    symbol=state.symbol,
                    strength=0.6,
                    price=state.last_price,
                    quantity=qty_budget,
                    reason="Kernel momentum crossed below the negative threshold.",
                    indicators={"close": state.last_price},
                )
            )
        elif kernel.should_sell(idx):
            signals.append(
                CandidateSignal(
                    source="KernelMomentum",
                    side=Side.SELL,
                    symbol=state.symbol,
                    strength=0.58,
                    price=state.last_price,
                    quantity=qty_budget,
                    reason="Kernel momentum crossed above the positive threshold.",
                    indicators={"close": state.last_price},
                )
            )

        if not signals:
            signals.append(
                CandidateSignal(
                    source="QuantHold",
                    side=Side.BUY,
                    symbol=state.symbol,
                    strength=0.15,
                    price=state.last_price,
                    quantity=qty_budget,
                    reason="No technical trigger; hold bias only.",
                    indicators={"close": state.last_price},
                )
            )

        state.signals = signals
        state.append_trace(
            "quant",
            "Combined Bollinger and kernel signals on completed bars.",
            signal_count=len(signals),
            last_price=state.last_price,
            sources=[s.source for s in signals],
        )
        return state
