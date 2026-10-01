from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from agents.state import Bar, OrderProposal, PolicyDecision, Position


@dataclass(frozen=True)
class PolicyLimits:
    max_name_weight: float = 0.05
    max_stop_loss_pct: float = 2.0
    var_limit_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    circuit_breaker_return: float = -0.05
    min_quantity: int = 1


class PolicyEngine:
    """Rule-based interceptor. Runs before any TradeExecutor call."""

    def __init__(self, limits: Optional[PolicyLimits] = None):
        self.limits = limits or PolicyLimits()

    def evaluate(
        self,
        *,
        nav: float,
        cash: float,
        last_price: float,
        bars: Sequence[Bar],
        positions: dict,
        proposal: Optional[OrderProposal],
        trading_mode: str,
        max_name_weight: Optional[float] = None,
    ) -> PolicyDecision:
        limits = self.limits
        weight_cap = max_name_weight if max_name_weight is not None else limits.max_name_weight
        violations: List[str] = []
        if proposal is None:
            return PolicyDecision(
                allowed=False,
                violations=["no_order_proposal"],
                requires_human=True,
            )

        if last_price <= 0:
            violations.append("invalid_price")

        returns = _close_returns(bars)
        vol = float(np.std(returns)) if len(returns) else 0.0
        hist_var = _historical_var(returns, 0.05)
        drawdown = _max_drawdown(bars)
        last_return = returns[-1] if len(returns) else 0.0
        circuit = last_return <= limits.circuit_breaker_return

        if circuit:
            violations.append("circuit_breaker")
        if drawdown > limits.max_drawdown_pct:
            violations.append("max_drawdown")
        if proposal.stop_loss_pct > limits.max_stop_loss_pct:
            violations.append("stop_loss_bound")

        position = positions.get(proposal.symbol)
        if isinstance(position, Position):
            current_qty = position.quantity
        elif isinstance(position, dict):
            current_qty = int(position.get("quantity") or 0)
        else:
            current_qty = 0

        signed = proposal.quantity if proposal.side.value == "buy" else -proposal.quantity
        new_qty = current_qty + signed
        notional = abs(new_qty) * last_price
        max_notional = max(nav * weight_cap, 0.0)
        capped_qty = proposal.quantity
        if nav > 0 and notional > max_notional + 1e-9:
            capped_qty = max(int(max_notional // last_price), 0)
            if capped_qty < limits.min_quantity:
                violations.append("position_cap")
            else:
                violations.append("position_cap_resized")
        if proposal.side.value == "buy" and proposal.quantity * last_price > cash + 1e-9:
            violations.append("insufficient_cash")

        var_cash = abs(hist_var) * last_price * proposal.quantity
        if nav > 0 and var_cash > nav * limits.var_limit_pct:
            violations.append("var_limit")

        blocking = [
            v
            for v in violations
            if v
            not in (
                "position_cap_resized",
            )
        ]
        allowed = not blocking and capped_qty >= limits.min_quantity
        requires_human = trading_mode == "real" or not allowed or vol > 0.03
        return PolicyDecision(
            allowed=allowed,
            violations=violations,
            capped_quantity=capped_qty if allowed else None,
            requires_human=requires_human,
            max_notional=max_notional,
            var_estimate=var_cash,
        )


def _close_returns(bars: Sequence[Bar]) -> np.ndarray:
    closes = np.array([b.close for b in bars], dtype=float)
    if len(closes) < 2:
        return np.array([])
    prev = closes[:-1]
    prev = np.where(prev == 0, np.nan, prev)
    rets = (closes[1:] - prev) / prev
    return rets[np.isfinite(rets)]


def _historical_var(returns: np.ndarray, tail: float) -> float:
    if returns.size == 0:
        return 0.0
    return float(np.quantile(returns, tail))


def _max_drawdown(bars: Sequence[Bar]) -> float:
    closes = np.array([b.close for b in bars], dtype=float)
    if closes.size == 0:
        return 0.0
    peak = np.maximum.accumulate(closes)
    dd = (peak - closes) / np.where(peak == 0, 1, peak)
    return float(np.max(dd)) if dd.size else 0.0
